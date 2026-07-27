import base64
import io
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from recolor import recolor
from segment import run_segmentation

app = FastAPI(title="ThePainters CV Service")

MAX_LONG_EDGE = 1600
MIN_MASK_AREA = 0.08


def _resize_long_edge(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image
    scale = max_edge / long_edge
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/simulate")
async def simulate(
    photo: UploadFile = File(...),
    surface: str = Form(...),
    target_l: float = Form(...),
    target_a: float = Form(...),
    target_b: float = Form(...),
    mask: Optional[UploadFile] = File(None),
):
    photo_bytes = await photo.read()
    try:
        image = Image.open(io.BytesIO(photo_bytes))
        image.load()
    except Exception:
        return JSONResponse({"ok": False, "message": "No se ha podido leer la imagen."}, status_code=400)

    image = _resize_long_edge(image, MAX_LONG_EDGE)
    target_class = "wall" if surface == "interior_wall" else "building"

    used_cached_mask = mask is not None
    mask_png_base64 = None
    mask_area = None
    confidence = None

    if used_cached_mask:
        mask_bytes = await mask.read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.NEAREST)
    else:
        mask_img, mask_area, confidence = run_segmentation(image, target_class)
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_png_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

        if confidence == "low" or mask_area < MIN_MASK_AREA:
            return JSONResponse(
                {
                    "ok": False,
                    "confidence": confidence,
                    "mask_area": mask_area,
                    "message": "No hemos podido identificar la pared con suficiente confianza. Prueba con otra foto, más frontal y con mejor luz.",
                },
                status_code=200,
            )

    result_img = recolor(image, mask_img, (target_l, target_a, target_b))

    result_buf = io.BytesIO()
    result_img.save(result_buf, format="WEBP", quality=88)

    response = {
        "ok": True,
        "result_webp_base64": base64.b64encode(result_buf.getvalue()).decode("ascii"),
        "mask_area": mask_area,
        "confidence": confidence,
        "used_cached_mask": used_cached_mask,
    }
    if mask_png_base64:
        response["mask_png_base64"] = mask_png_base64

    return JSONResponse(response)

