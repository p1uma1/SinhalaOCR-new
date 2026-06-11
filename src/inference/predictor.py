import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision.datasets import ImageFolder

from src.data.class_labels import CLASS_LABELS_PATH, resolve_character_labels, save_class_labels
from src.models.trocr_model import SinhalaTrOCR
from src.models.vision_encoder import DeiTClassifier
from src.utils.device import configure_gpu


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSOR_NAME = "google/vit-base-patch16-384"
TOKENIZER_NAME = "keshan/SinhalaBERTo"
STAGE1_WEIGHTS = ROOT_DIR / "outputs/stage1/best_classifier.pth"
STAGE2_WEIGHTS = ROOT_DIR / "outputs/stage2/best_trocr.pth"
CLASS_DATA_DIR = ROOT_DIR / "Datasets/Dataset454/train"
CLASS_LABELS_FILE = CLASS_LABELS_PATH


class ModelService:
    def __init__(self):
        self.device, self.use_amp = configure_gpu()
        self.processor = ViTImageProcessor.from_pretrained(PROCESSOR_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self._stage1_model = None
        self._stage2_model = None
        self._class_folders = None
        self._character_labels = None

    @property
    def class_folders(self):
        if self._class_folders is None and CLASS_DATA_DIR.is_dir():
            dataset = ImageFolder(str(CLASS_DATA_DIR))
            self._class_folders = dataset.classes
        return self._class_folders or []

    @property
    def character_labels(self):
        if self._character_labels is None:
            labels = resolve_character_labels(self.class_folders, CLASS_LABELS_FILE)
            if labels:
                self._character_labels = labels
            elif self.class_folders:
                self._character_labels = resolve_character_labels(self.class_folders)
                save_class_labels(self.class_folders, CLASS_LABELS_FILE)
        return self._character_labels or []

    def status(self):
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "stage1_ready": STAGE1_WEIGHTS.is_file(),
            "stage2_ready": STAGE2_WEIGHTS.is_file(),
            "num_classes": len(self.character_labels) or len(self.class_folders) or 454,
            "labels_ready": bool(self.character_labels),
        }

    def _load_stage1(self):
        if self._stage1_model is not None:
            return self._stage1_model

        if not STAGE1_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"Stage 1 weights not found at {STAGE1_WEIGHTS}. Run train_stage1.py first."
            )

        num_classes = len(self.class_folders) or len(self.character_labels) or 454
        model = DeiTClassifier(num_classes=num_classes).to(self.device)
        model.load_state_dict(
            torch.load(STAGE1_WEIGHTS, map_location=self.device, weights_only=True)
        )
        model.eval()
        self._stage1_model = model
        return model

    def _load_stage2(self):
        if self._stage2_model is not None:
            return self._stage2_model

        encoder_path = ROOT_DIR / "outputs/stage1/pretrained_encoder.pth"
        model = SinhalaTrOCR(
            encoder_pretrained_path=str(encoder_path) if encoder_path.is_file() else None
        ).to(self.device)

        if STAGE2_WEIGHTS.is_file():
            model.load_state_dict(
                torch.load(STAGE2_WEIGHTS, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(
                f"Stage 2 weights not found at {STAGE2_WEIGHTS}. Run train_stage2.py first."
            )

        model.eval()
        self._stage2_model = model
        return model

    def _image_from_bytes(self, image_bytes):
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _pixel_values(self, image):
        return self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

    def predict_character(self, image_bytes, top_k=5):
        image = self._image_from_bytes(image_bytes)
        model = self._load_stage1()
        pixel_values = self._pixel_values(image)

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast(self.device.type):
                    logits = model(pixel_values)
            else:
                logits = model(pixel_values)

            probabilities = torch.softmax(logits, dim=-1)[0]
            num_labels = len(self.character_labels) or len(self.class_folders) or probabilities.numel()
            top_scores, top_indices = torch.topk(probabilities, min(top_k, num_labels))

        predictions = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            character = (
                self.character_labels[idx]
                if idx < len(self.character_labels)
                else str(idx)
            )
            class_id = (
                self.class_folders[idx]
                if idx < len(self.class_folders)
                else str(idx)
            )
            predictions.append(
                {
                    "label": character,
                    "character": character,
                    "class_id": class_id,
                    "confidence": round(score * 100, 2),
                }
            )

        return {
            "top_prediction": predictions[0],
            "predictions": predictions,
        }

    def _append_context(self, context, text, separator=" "):
        if not text:
            return context
        if not context:
            return text
        return f"{context}{separator}{text}"

    def _build_decoder_prefix(self, context):
        if not context or not context.strip():
            return None, 0

        prefix_ids = self.tokenizer(
            context.strip(),
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=96,
        ).input_ids.to(self.device)

        cls_id = torch.tensor([[self.tokenizer.cls_token_id]], device=self.device)
        decoder_input_ids = torch.cat([cls_id, prefix_ids], dim=1)
        return decoder_input_ids, decoder_input_ids.shape[1]

    def predict_segment(self, image_bytes, context=""):
        """OCR one line/word crop, optionally continuing from prior decoded text."""
        image = self._image_from_bytes(image_bytes)
        model = self._load_stage2()
        pixel_values = self._pixel_values(image)
        decoder_input_ids, prefix_len = self._build_decoder_prefix(context)

        gen_kwargs = {}
        if decoder_input_ids is not None:
            gen_kwargs["decoder_input_ids"] = decoder_input_ids

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast(self.device.type):
                    generated_ids = model.generate(pixel_values, **gen_kwargs)
            else:
                generated_ids = model.generate(pixel_values, **gen_kwargs)

        if prefix_len > 0:
            new_ids = generated_ids[:, prefix_len:]
        else:
            new_ids = generated_ids

        text = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return {"text": text}

    def predict_line(self, image_bytes, context=""):
        return self.predict_segment(image_bytes, context=context)

    def predict_document(self, image_bytes):
        from src.preprocessing.document import extract_lines_from_bytes
        import base64

        extraction = extract_lines_from_bytes(
            image_bytes, return_debug=True, segment_words=True
        )
        line_results = []
        document_context = ""
        lines = extraction["lines"]

        for line_idx, line in enumerate(lines):
            word_results = []
            words = line.get("words") or []

            if words:
                for word in words:
                    image_data = base64.b64decode(word["image_base64"])
                    segment = self.predict_segment(
                        image_data, context=document_context
                    )
                    word_text = segment["text"]
                    word_results.append(
                        {
                            "index": word["index"],
                            "bbox": word.get("bbox"),
                            "image_base64": word["image_base64"],
                            "text": word_text,
                        }
                    )
                    if word_text:
                        document_context = self._append_context(
                            document_context, word_text, separator=" "
                        )

                line_text = " ".join(
                    item["text"] for item in word_results if item["text"]
                )
            else:
                image_data = base64.b64decode(line["image_base64"])
                segment = self.predict_segment(image_data, context=document_context)
                line_text = segment["text"]
                if line_text:
                    document_context = self._append_context(
                        document_context, line_text, separator=" "
                    )

            line_results.append(
                {
                    "index": line["index"],
                    "image_base64": line["image_base64"],
                    "text": line_text,
                    "words": word_results,
                    "word_count": len(word_results),
                }
            )

            if line_text and line_idx < len(lines) - 1:
                document_context = f"{document_context}\n"

        full_text = "\n".join(item["text"] for item in line_results if item["text"])

        return {
            "line_count": extraction["line_count"],
            "word_count": extraction.get("word_count", 0),
            "segmentation": extraction.get("segmentation", "line+word"),
            "lines": line_results,
            "full_text": full_text,
            "debug": extraction.get("debug", {}),
        }


@lru_cache(maxsize=1)
def get_model_service():
    return ModelService()
