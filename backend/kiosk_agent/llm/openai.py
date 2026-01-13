"""OpenAI ChatGPT client for vision-language tasks."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

from ..config import ModelConfig
from ..utils import get_logger
from .base import BaseModelClient

logger = get_logger(__name__)


class GUI_OUTPUT(BaseModel):
    """Schema for ChatGPT GUI action output."""
    thought: str = Field(description="Agent reasoning (not executed)")
    action: Literal['CLICK', 'LONG_CLICK', 'SWIPE', 'INPUT', 'BACK', 'HOME'] = Field(
        description="Type of GUI action to perform."
    )
    value: Optional[str] = Field(
        default=None, 
        description="Text to input when action == INPUT (non-empty). Otherwise null/omit."
    )
    position: List[float] = Field(
        description="Normalized (0~1) screen coordinate [x,y] for action.",
        min_length=2,
        max_length=2,
    )


class ChatGPTClient(BaseModelClient):
    """Calls OpenAI's ChatGPT vision models."""

    def __init__(self, config: ModelConfig):
        if not config.openai_api_key:
            raise ValueError("Set ModelConfig.openai_api_key when provider='chatgpt'.")
        super().__init__(config)
        
        from openai import OpenAI
        self._client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_api_base)

    def generate(self, instruction: str, image: Image.Image) -> str:
        """Generate action from ChatGPT model."""
        encoded_image = self._encode_image(image)
        
        response = self._client.responses.parse(
            model=self.config.openai_model,
            input=[
                {
                    "role": "system", 
                    "content": self.config.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": instruction
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded_image}",
                        },
                    ],
                },
            ],
            text_format=GUI_OUTPUT
        )
        logger.debug(f"Raw response: {response.output_text}")
        return response.output_text

    def encode_image(self, image_path):
        """Encode image file to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_completion(self, content: Any) -> str:
        """Parse completion content to string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content)
