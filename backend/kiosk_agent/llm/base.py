"""Base LLM client interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from PIL import Image

from ..config import ModelConfig


@dataclass
class ModelAction:
    """Normalized representation of what the VLM responded with."""

    raw_text: str
    payload: Dict[str, Any]


class BaseModelClient(ABC):
    """Common interface for talking to either a local or hosted VLM."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, instruction: str, image: Image.Image) -> str:
        """Return the raw model text output."""

    def propose_action(self, instruction: str, image: Image.Image) -> ModelAction:
        raw = self.generate(instruction, image)
        raw_text = self._coerce_raw_text(raw)
        payload = self._safe_parse(raw_text)
        return ModelAction(raw_text=raw_text, payload=payload)

    @staticmethod
    def _safe_parse(raw_text: str) -> Dict[str, Any]:
        """Best effort JSON parsing with fallbacks for code fenced output."""
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if "\n" in candidate:
                candidate = candidate.split("\n", 1)[1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            raise ValueError(f"Model response is not valid JSON:\n{raw_text}")

    @staticmethod
    def _coerce_raw_text(raw: Any) -> str:
        if isinstance(raw, ModelAction):
            return raw.raw_text
        if isinstance(raw, tuple):
            return raw[0]
        return str(raw)
