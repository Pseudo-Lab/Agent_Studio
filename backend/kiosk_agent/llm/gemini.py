"""Google Gemini client for vision-language tasks."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from PIL import Image

from ..config import ModelConfig
from ..utils import get_logger
from .base import BaseModelClient
from ..prompts.schema import GUI_OUTPUT

logger = get_logger(__name__)


class GeminiClient(BaseModelClient):
    """Calls the Gemini SDK vision models."""

    def __init__(self, config: ModelConfig):
        if not config.gemini_api_key:
            raise ValueError("Set ModelConfig.gemini_api_key when provider='gemini'.")
        super().__init__(config)
        
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._types = types

    def generate(self, instruction: str, image: Image.Image) -> Tuple[str, int, int]:
        """
        Generate action from Gemini model.
        
        Returns:
            Tuple of (response_text, image_width, image_height)
        """
        types = self._types
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            system_instruction=self.config.system_prompt,
            response_mime_type="application/json",
            response_json_schema=GUI_OUTPUT.model_json_schema(),
        )
        
        contents = self._build_parts(instruction, image)
        response = self._client.models.generate_content(
            model=self.config.gemini_model,
            contents=contents,
            config=config,
        )
        
        output_text = (response.text or "").strip()
        if not output_text:
            raise RuntimeError("Gemini returned an empty response.")
        
        width, height = image.size
        logger.debug(f"Raw response: {response.text}")
        dict_response = json.loads(response.text)

        converted_bounding_boxes = []
        abs_y1 = int(dict_response["box_2d"][0]/1000 * height)
        abs_x1 = int(dict_response["box_2d"][1]/1000 * width)
        abs_y2 = int(dict_response["box_2d"][2]/1000 * height)
        abs_x2 = int(dict_response["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_y1, abs_x1, abs_y2, abs_x2])

        logger.debug(f"Image: {width}x{height}, box: {converted_bounding_boxes}")

        mid_y = (abs_y1 + abs_y2)/2
        mid_x = (abs_x1 + abs_x2)/2
        logger.debug(f"Mid point: [{mid_x}, {mid_y}], thought: {dict_response['thought'][:50]}...")
        
        return response.text, width, height

    def encode_image(self, image_path):
        """Encode image file to bytes."""
        with open(image_path, "rb") as image_file:
            return image_file.read()

    def _build_parts(self, instruction: str, image: Optional[Image.Image]) -> List:
        """Build content parts for Gemini API."""
        parts = []
        if image is not None:
            parts.append(image)
        parts.append(instruction)
        return parts

    @staticmethod
    def _build_image_part(image: Optional[Image.Image]):
        """Build image part for Gemini API."""
        if image is None:
            return None
        from google.genai import types
        return types.Part.from_bytes(
            data=image,
            mime_type="image/png",
        )
