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
    binary_mask = (class_map == class_idx).astype(np.uint8) * 255

    mask_img = Image.fromarray(binary_mask, mode="L").resize(original_size, Image.NEAREST)
    mask_arr = np.asarray(mask_img)
    mask_area = float(np.mean(mask_arr > 127))

    flat_probs = probs.reshape(num_classes, -1)
    flat_target = (class_map == class_idx).reshape(-1)
    region_probs = flat_probs[:, flat_target] if flat_target.any() else flat_probs
    eps = 1e-8
    entropy = -np.sum(region_probs * np.log(region_probs + eps), axis=0)
    mean_entropy = float(np.mean(entropy)) / np.log(num_classes)

    if mask_area < MIN_MASK_AREA:
        confidence = "low"
    elif mean_entropy > ENTROPY_LOW_THRESHOLD:
        confidence = "low"
    elif mean_entropy > ENTROPY_LOW_THRESHOLD * 0.65:
        confidence = "medium"
    else:
        confidence = "high"

    # erode 1-2px to avoid halos around frames/skirting boards
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))

    return mask_img, mask_area, confidence
