import json
import os

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
OUT_DIR = os.environ.get("MODEL_DIR", "/model")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.eval()
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

    dummy = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy,
        os.path.join(OUT_DIR, "segmentation.onnx"),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )

    with open(os.path.join(OUT_DIR, "id2label.json"), "w") as f:
        json.dump(model.config.id2label, f)

    with open(os.path.join(OUT_DIR, "preprocessor_config.json"), "w") as f:
        json.dump(
            {
                "image_mean": processor.image_mean,
                "image_std": processor.image_std,
                "size": processor.size,
            },
            f,
        )

    print(f"Modelo exportado en {OUT_DIR}")


if __name__ == "__main__":
    main()

