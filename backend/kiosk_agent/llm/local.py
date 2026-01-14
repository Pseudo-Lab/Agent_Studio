"""Local vLLM client for self-hosted models."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict

import requests
from PIL import Image

from ..config import ModelConfig
from .base import BaseModelClient


class LocalVLLMClient(BaseModelClient):
    """Talks to a locally hosted vLLM server that mimics the OpenAI chat-completions API."""

    def __init__(self, config: ModelConfig):
        if not config.vllm_model_name:
            raise ValueError("Set ModelConfig.vllm_model_name when provider='local_vllm'.")
        super().__init__(config)

    def generate(self, instruction: str, image: Image.Image) -> str:
        """Generate action from local vLLM model."""
        payload: Dict[str, Any] = {
            "model": self.config.vllm_model_name,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._encode_image(image)}"},
                        },
                    ],
                },
            ],
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"
        
        response = requests.post(
            f"{self.config.vllm_base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected vLLM response: {body}") from exc
        
        return content if isinstance(content, str) else str(content)

    def generate_text(self, prompt: str) -> str:
        """Text-only generation for Planning mode (no image)."""
        payload: Dict[str, Any] = {
            "model": self.config.vllm_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 유용한 도우미입니다. 요구된 형식(JSON)을 정확히 지켜서 응답하세요.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self.config.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vllm_api_key}"

        response = requests.post(
            f"{self.config.vllm_base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected vLLM response: {body}") from exc

        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
