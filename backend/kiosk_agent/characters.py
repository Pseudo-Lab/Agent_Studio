"""
Character system for Kiosk Agent TTS.

Provides random character selection with unique voice and personality.
Character information is loaded from config/characters.yaml.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .utils import get_logger

logger = get_logger(__name__)

# Session-to-character mapping for consistency
_session_character_map: Dict[str, "Character"] = {}

# Config file path (can be overridden via env var)
CONFIG_PATH = os.getenv(
    "CHARACTERS_CONFIG_PATH",
    str(Path(__file__).resolve().parents[1] / "config" / "characters.yaml")
)


@dataclass
class Character:
    """Character with voice and personality."""
    
    id: str
    name: str
    nickname: str
    ref_audio_filename: str
    ref_text: str
    image_path: str
    completion_messages: List[str] = field(default_factory=list)
    quit_messages: List[str] = field(default_factory=list)
    
    def get_completion_message(self, base_thought: str) -> str:
        """Style the completion message with character personality."""
        if self.completion_messages:
            prefix = random.choice(self.completion_messages)
            return f"{prefix} {base_thought}"
        return base_thought
    
    def get_quit_message(self) -> str:
        """Get styled quit message."""
        if self.quit_messages:
            return random.choice(self.quit_messages)
        return "작업이 취소되었습니다."


def _load_characters_from_yaml(config_path: str) -> List[Character]:
    """Load characters from YAML config file."""
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Characters config not found: {config_path}")
        return _get_default_characters()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        characters = []
        for char_data in data.get("characters", []):
            characters.append(Character(
                id=char_data["id"],
                name=char_data["name"],
                nickname=char_data["nickname"],
                ref_audio_filename=char_data["ref_audio"],
                ref_text=char_data["ref_text"],
                image_path=char_data["image_path"],
                completion_messages=char_data.get("completion_messages", []),
                quit_messages=char_data.get("quit_messages", []),
            ))
        
        logger.info(f"Loaded {len(characters)} characters from {config_path}")
        return characters
        
    except Exception as e:
        logger.error(f"Failed to load characters config: {e}")
        return _get_default_characters()


def _get_default_characters() -> List[Character]:
    """Fallback default characters if config loading fails."""
    return [
        Character(
            id="character1",
            name="캐릭터1",
            nickname="캐릭터1",
            ref_audio_filename="",
            ref_text="",
            image_path="/images/default.jpg",
            completion_messages=["완료되었습니다."],
            quit_messages=["작업이 취소되었습니다."],
        )
    ]


# Load characters on module import
ALL_CHARACTERS = _load_characters_from_yaml(CONFIG_PATH)


def get_character_for_session(session_id: str) -> Character:
    """
    Get or assign a character for a session.
    
    Same session always gets the same character for consistency.
    """
    if session_id in _session_character_map:
        return _session_character_map[session_id]
    
    # Random selection for new session
    character = random.choice(ALL_CHARACTERS)
    _session_character_map[session_id] = character
    logger.info(f"Assigned {character.nickname} to session {session_id[:8]}")
    return character


def clear_session_character(session_id: str) -> None:
    """Clear character assignment for a session."""
    if session_id in _session_character_map:
        del _session_character_map[session_id]


def get_character_image_path(character: Character) -> str:
    """Get character image URL path for frontend."""
    return character.image_path


def get_character_ref_audio_path(character: Character, agent_root: Optional[Path] = None) -> Optional[str]:
    """Get reference audio path for character."""
    if not character.ref_audio_filename:
        return None
        
    if agent_root is None:
        agent_root = Path(__file__).resolve().parents[2]
    
    # Check multiple locations
    candidates = [
        agent_root / character.ref_audio_filename,
        agent_root / "tts" / character.ref_audio_filename,
        Path(__file__).resolve().parent / "voice" / character.ref_audio_filename,
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    logger.warning(f"Reference audio not found for {character.nickname}: {character.ref_audio_filename}")
    return None


def reload_characters() -> None:
    """Reload characters from config file."""
    global ALL_CHARACTERS
    ALL_CHARACTERS = _load_characters_from_yaml(CONFIG_PATH)
