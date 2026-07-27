import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")

with open(os.path.join(MODEL_DIR, "id2label.json")) as f:
    ID2LABEL = json.load(f)

with open(os.path.join(MODEL_DIR, "preprocessor_config.json")) as f:
    PREPROC = json.load(f)

IMAGE_MEAN = np.array(PREPROC["image_mean"], dtype=np.float32)
IMAGE_STD = np.array(PREPROC["image_std"], dtype=np.float32)
INPUT_SIZE = (PREPROC["size"]["height"], PREPROC["size"]["width"])

_session_options = ort.SessionOptions()
_session_options.intra_op_num_threads = 1
_session_options.inter_op_num_threads = 1
_session_options.enable_cpu_mem_arena = False
_session_options.enable_mem_pattern = False
_session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "segmentation.onnx"),
    sess_options=_session_options,
    providers=["CPUExecutionProvider"],
)

_CLASS_NAME_TO_INDEX = {}
for idx_str, label in ID2LABEL.items():
    for part in label.split(","):
        _CLASS_NAME_TO_INDEX[part.strip().lower()] = int(idx_str)

MIN_MASK_AREA = 0.08
ENTROPY_LOW_THRESHOLD = 0.55

# ADE20K classes that should never be painted over, even if a few pixels near their
# edges get misclassified as "wall"/"building" by the model. Every pixel the model is
# even moderately confident belongs to one of these gets carved out of the paint mask,
# so furniture, frames, windows, plants, etc. keep their real color.
_FURNITURE_KEYWORDS = [
    "bed", "cabinet", "table", "chair", "sofa", "couch", "shelf", "armchair", "seat",
    "desk", "wardrobe", "closet", "lamp", "cushion", "chest of drawers", "dresser",
    "counter", "sink", "refrigerator", "icebox", "bookcase", "pillow", "bench",
    "countertop", "stove", "swivel chair", "television", "ottoman", "pouf", "buffet",
    "sideboard", "stool", "oven", "microwave", "dishwasher", "mirror", "painting",
    "picture", "curtain", "drape", "plant", "flora", "vase", "clock", "sconce",
    "radiator", "fan", "door", "windowpane", "window", "person", "animal", "box",
    "flowerpot", "poster", "screen", "monitor", "computer", "fireplace", "washer",
    "basket", "bag", "toy", "book", "bottle", "tray", "chandelier", "rug", "carpet",
    "blanket", "sculpture", "vase", "glass",
]

# Safety margin (in px, at model output resolution before upsampling) grown around
# every excluded object so paint never bleeds right up to a furniture silhouette.
_EXCLUDE_DILATE_PX = 2


def _resolve_furniture_indices():
    indices = set()
    for idx_str, label in ID2LABEL.items():
        lname = label.lower()
        if any(kw in lname for kw in _FURNITURE_KEYWORDS):
            indices.add(int(idx_str))
    return indices


_FURNITURE_INDICES = _resolve_furniture_indices()


def resolve_class_index(name: str) -> int:
    key = name.strip().lower()
    if key not in _CLASS_NAME_TO_INDEX:
        raise ValueError(f"Clase de segmentación desconocida: {name}")
    return _CLASS_NAME_TO_INDEX[key]


def _preprocess(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, :, :, :]


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _upsample_prob(prob_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear-upsample a single-channel float probability map to `size` (w, h).

    Using bilinear here (instead of nearest-neighbor on an already-argmaxed mask)
    means the wall silhouette follows furniture edges much more closely, since the
    decision boundary is interpolated smoothly rather than blown up in blocky steps.
    """
    img = Image.fromarray(prob_map.astype(np.float32), mode="F")
    return np.asarray(img.resize(size, Image.BILINEAR))


def run_segmentation(image: Image.Image, target_class: str):
    """Returns (mask_img: PIL "L" image, mask_area: float, confidence: str)."""
    class_idx = resolve_class_index(target_class)
    original_size = image.size

    input_tensor = _preprocess(image)
    (logits,) = _session.run(["logits"], {"pixel_values": input_tensor})
    logits = logits[0]
    probs = _softmax(logits, axis=0)
    num_classes = probs.shape[0]

    class_map = np.argmax(probs, axis=0)

    # Confidence/entropy stats are computed at model resolution, on the raw argmax
    # decision, before the furniture-exclusion refinement below.
    mask_area_raw = float(np.mean(class_map == class_idx))
    flat_probs = probs.reshape(num_classes, -1)
    flat_target = (class_map == class_idx).reshape(-1)
    region_probs = flat_probs[:, flat_target] if flat_target.any() else flat_probs
    eps = 1e-8
    entropy = -np.sum(region_probs * np.log(region_probs + eps), axis=0)
    mean_entropy = float(np.mean(entropy)) / np.log(num_classes)

    if mask_area_raw < MIN_MASK_AREA:
        confidence = "low"
    elif mean_entropy > ENTROPY_LOW_THRESHOLD:
        confidence = "low"
    elif mean_entropy > ENTROPY_LOW_THRESHOLD * 0.65:
        confidence = "medium"
    else:
        confidence = "high"

    # Wall probability, upsampled smoothly to the original photo resolution.
    wall_prob_full = _upsample_prob(probs[class_idx], original_size)

    # Combined probability of any furniture/object class, so the paint mask can
    # explicitly carve those regions out even where the wall/furniture boundary is
    # ambiguous at model resolution.
    if _FURNITURE_INDICES:
        furniture_prob = np.zeros_like(probs[0])
        for idx in _FURNITURE_INDICES:
            if idx != class_idx:
                furniture_prob += probs[idx]
        furniture_mask_small = (furniture_prob > 0.25).astype(np.uint8) * 255
        furniture_img = Image.fromarray(furniture_mask_small, mode="L")
        if _EXCLUDE_DILATE_PX > 0:
            furniture_img = furniture_img.filter(ImageFilter.MaxFilter(2 * _EXCLUDE_DILATE_PX + 1))
        furniture_full = np.asarray(furniture_img.resize(original_size, Image.BILINEAR)) > 127
    else:
        furniture_full = np.zeros(original_size[::-1], dtype=bool)

    binary_mask = ((wall_prob_full > 0.5) & (~furniture_full)).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask, mode="L")
    mask_arr = np.asarray(mask_img)
    mask_area = float(np.mean(mask_arr > 127))

    # erode a couple more px to avoid halos around frames/skirting boards/furniture
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))

    return mask_img, mask_area, confidence
