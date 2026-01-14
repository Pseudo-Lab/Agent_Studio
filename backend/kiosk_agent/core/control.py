"""ADB (Android Debug Bridge) control module."""

from __future__ import annotations

import subprocess
import time
from typing import List, Sequence

from ..config import ADBConfig
from ..utils import get_logger

logger = get_logger(__name__)


class ADBController:
    """Light wrapper around adb shell gestures."""

    # Dubeolsik (2-set) Hangul keyboard mapping.
    # Used as a fallback when `adb shell input text` cannot inject Hangul directly
    # (some devices crash with InputShellCommand NPE for non-ASCII chars).
    _CHO_KEYS = [
        "r",  # ㄱ
        "R",  # ㄲ
        "s",  # ㄴ
        "e",  # ㄷ
        "E",  # ㄸ
        "f",  # ㄹ
        "a",  # ㅁ
        "q",  # ㅂ
        "Q",  # ㅃ
        "t",  # ㅅ
        "T",  # ㅆ
        "d",  # ㅇ
        "w",  # ㅈ
        "W",  # ㅉ
        "c",  # ㅊ
        "z",  # ㅋ
        "x",  # ㅌ
        "v",  # ㅍ
        "g",  # ㅎ
    ]
    _JUNG_KEYS = [
        "k",   # ㅏ
        "o",   # ㅐ
        "i",   # ㅑ
        "O",   # ㅒ
        "j",   # ㅓ
        "p",   # ㅔ
        "u",   # ㅕ
        "P",   # ㅖ
        "h",   # ㅗ
        "hk",  # ㅘ
        "ho",  # ㅙ
        "hl",  # ㅚ
        "y",   # ㅛ
        "n",   # ㅜ
        "nj",  # ㅝ
        "np",  # ㅞ
        "nl",  # ㅟ
        "b",   # ㅠ
        "m",   # ㅡ
        "ml",  # ㅢ
        "l",   # ㅣ
    ]
    _JONG_KEYS = [
        "",    # (none)
        "r",   # ㄱ
        "R",   # ㄲ
        "rt",  # ㄳ
        "s",   # ㄴ
        "sw",  # ㄵ
        "sg",  # ㄶ
        "e",   # ㄷ
        "f",   # ㄹ
        "fr",  # ㄺ
        "fa",  # ㄻ
        "fq",  # ㄼ
        "ft",  # ㄽ
        "fx",  # ㄾ
        "fv",  # ㄿ
        "fg",  # ㅀ
        "a",   # ㅁ
        "q",   # ㅂ
        "qt",  # ㅄ
        "t",   # ㅅ
        "T",   # ㅆ
        "d",   # ㅇ
        "w",   # ㅈ
        "c",   # ㅊ
        "z",   # ㅋ
        "x",   # ㅌ
        "v",   # ㅍ
        "g",   # ㅎ
    ]
    _HANGUL_BASE = 0xAC00
    _HANGUL_LAST = 0xD7A3
    _CHO_DIV = 588
    _JUNG_DIV = 28

    def __init__(self, config: ADBConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def _base_cmd(self) -> List[str]:
        cmd = [self.config.adb_path]
        if self.config.device_id:
            cmd += ["-s", self.config.device_id]
        return cmd

    def tap(self, x: int, y: int) -> Sequence[str]:
        # tap(=click) (y,x) position.
        return self._run(["shell", "input", "touchscreen", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms=100) -> Sequence[str]:
        cmd = ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        return self._run(cmd)

    def long_press(self, x: int, y: int, duration_ms: int = 500) -> Sequence[str]:
        """Long press at (x, y) coordinates."""
        return self._run([
            "shell", "input", "touchscreen", "swipe",
            str(x), str(y), str(x), str(y), str(duration_ms)
        ])

    def press(self, key: str) -> Sequence[str]:
        return self._run(["shell", "input", "keyevent", key])

    def type_text(self, text: str) -> Sequence[str]:
        safe_text = text.replace(" ", "%s")
        try:
            return self._run(["shell", "input", "text", safe_text])
        except subprocess.CalledProcessError:
            # Some Android builds can't inject Hangul directly via `input text` (NPE in InputShellCommand).
            # Fall back to typing dubeolsik keystrokes, which are ASCII and therefore injectable.
            if any(self._HANGUL_BASE <= ord(ch) <= self._HANGUL_LAST for ch in text):
                fallback = self._hangul_to_dubeolsik(text)
                safe_fallback = fallback.replace(" ", "%s")
                logger.warning("adb input text failed for Hangul; falling back to dubeolsik keystrokes")
                return self._run(["shell", "input", "text", safe_fallback])
            raise

    def back(self) -> Sequence[str]:
        """Press back button."""
        return self.press("KEYCODE_BACK")

    def home(self) -> Sequence[str]:
        """Press home button."""
        return self.press("KEYCODE_HOME")

    def run_raw(self, args: Sequence[str]) -> Sequence[str]:
        return self._run(list(args))
    
    def screenshot(self):
        abs_path = self.config.screenshot_abs_path
        new_path = abs_path + "/" + "mac_test_" + str(self.config.steps) + ".png"
        self.config.steps += 1

        return self._run(["exec-out", "screencap", "-p", ">", f"{new_path}"])

    def _run(self, args: Sequence[str]) -> Sequence[str]:
        full_cmd = self._base_cmd() + list(args)
        
        if self.dry_run:
            logger.info(f"[DRY RUN] {' '.join(full_cmd)}")
            return full_cmd

        final_cmd = " ".join(full_cmd)
        subprocess.run(final_cmd, shell=True, check=True)
        time.sleep(1.5)  # 동작완료 대기
        return full_cmd

    @classmethod
    def _hangul_to_dubeolsik(cls, text: str) -> str:
        """
        Convert Hangul syllables to dubeolsik (2-set) keystrokes.

        Example:
            "대상혁 버거" -> "eotkdgur qjrj"

        This relies on the device keyboard/IME being in Korean mode for the keystrokes
        to compose into Hangul. If the IME is in English mode, it will type the keystrokes
        literally, but it will not crash.
        """
        out: List[str] = []
        for ch in text:
            code = ord(ch)
            if cls._HANGUL_BASE <= code <= cls._HANGUL_LAST:
                sindex = code - cls._HANGUL_BASE
                cho = sindex // cls._CHO_DIV
                jung = (sindex % cls._CHO_DIV) // cls._JUNG_DIV
                jong = sindex % cls._JUNG_DIV
                out.append(cls._CHO_KEYS[cho] + cls._JUNG_KEYS[jung] + cls._JONG_KEYS[jong])
            else:
                out.append(ch)
        return "".join(out)
