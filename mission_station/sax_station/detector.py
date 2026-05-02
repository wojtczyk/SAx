from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


class YoloDetector:
    def __init__(self) -> None:
        self._model_name: str | None = None
        self._model: Any | None = None

    def load(self, model_name: str) -> None:
        if self._model is not None and self._model_name == model_name:
            return

        config_dir = Path(__file__).resolve().parents[1] / "data" / "ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._model_name = model_name

    def _device(self) -> str:
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def detect(self, frame, confidence_threshold: float) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("YOLO model has not been loaded.")

        results = self._model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False,
            device=self._device(),
        )
        detections: list[Detection] = []
        names = results[0].names

        for box in results[0].boxes:
            label_index = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=str(names[label_index]),
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        return detections

    def draw(self, frame, detections: list[Detection]):
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            label = f"{detection.label} {detection.confidence:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 204, 113), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (46, 204, 113),
                2,
                cv2.LINE_AA,
            )
        return annotated
