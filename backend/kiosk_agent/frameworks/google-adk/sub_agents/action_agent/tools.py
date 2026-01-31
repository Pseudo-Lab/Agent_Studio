"""ADB action tools for the ADK action agent.

Reuses the existing ADBController from kiosk_agent.core so that coordinate
translation, Hangul input, etc. remain consistent with the LangGraph version.

Coordinate system
-----------------
All ``box_2d`` parameters use Gemini's normalised 0-1000 format:
    [y_min, x_min, y_max, x_max]
The helper ``_box_to_pixel_center`` converts them to device pixel coords.
"""

from ...shared_libraries import Status

# Default screen resolution used when ADB capture is unavailable.
_FALLBACK_SCREEN_SIZE = (1080, 2400)  # (width, height)


def _get_controller():
    """Lazily create an ADBController from the existing kiosk_agent core.

    Lazy import avoids circular dependencies at module load time.
    """
    from .....core.control import ADBController
    from .....config import ADBConfig
    return ADBController(ADBConfig())


def _get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the connected device screen.

    Falls back to ``_FALLBACK_SCREEN_SIZE`` if ADB is unreachable.
    """
    try:
        from .....core.perception import AndroidScreenshotter
        from .....config import ScreenshotConfig
        screenshotter = AndroidScreenshotter(ScreenshotConfig())
        screenshot = screenshotter.capture(save=False)
        return screenshot.image.size
    except Exception:
        return _FALLBACK_SCREEN_SIZE


def _box_to_pixel_center(
    y_min: float, x_min: float, y_max: float, x_max: float
) -> tuple[int, int]:
    """Convert normalised box_2d (0-1000) to pixel centre point.

    Args:
        y_min, x_min, y_max, x_max: Normalised bounding-box coordinates.

    Returns:
        (cx, cy) pixel coordinates at the centre of the bounding box.
    """
    width, height = _get_screen_size()
    top = int(max(0, min(1000, y_min)) / 1000 * height)
    left = int(max(0, min(1000, x_min)) / 1000 * width)
    bottom = int(max(0, min(1000, y_max)) / 1000 * height)
    right = int(max(0, min(1000, x_max)) / 1000 * width)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    return cx, cy


def adb_click(y_min: float, x_min: float, y_max: float, x_max: float) -> dict:
    """Tap at the center of a bounding box. Coordinates in box_2d format [ymin, xmin, ymax, xmax] normalized 0-1000."""
    print(f"TOOL CALLED: adb_click(box=[{y_min}, {x_min}, {y_max}, {x_max}])")

    try:
        cx, cy = _box_to_pixel_center(y_min, x_min, y_max, x_max)
        controller = _get_controller()
        controller.tap(cx, cy)
        return {
            "status": Status.SUCCESS.value,
            "message": f"Tapped at pixel ({cx}, {cy})"
        }
    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }


def adb_swipe() -> dict:
    """Scroll down on the screen (swipe-up gesture from bottom to centre)."""
    print("TOOL CALLED: adb_swipe()")

    try:
        from .....config import ADBConfig
        config = ADBConfig()
        controller = _get_controller()

        # Swipe from bottom-centre to mid-centre (scroll down)
        x_center = 500
        y_start = 1000
        y_end = 500
        controller.swipe(
            x_center, y_start, x_center, y_end,
            duration_ms=config.default_swipe_duration_ms,
        )
        return {
            "status": Status.SUCCESS.value,
            "message": "Swiped (scrolled down)"
        }
    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }


def adb_type(text: str) -> dict:
    """Type text on the device."""
    print(f"TOOL CALLED: adb_type(text={text})")

    try:
        controller = _get_controller()
        controller.type_text(text)
        return {
            "status": Status.SUCCESS.value,
            "message": f"Typed: {text}"
        }
    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }


def adb_press(key: str) -> dict:
    """Press a key on the device (e.g., 'back', 'home', 'enter')."""
    print(f"TOOL CALLED: adb_press(key={key})")

    # Map friendly names to Android KEYCODE constants
    _KEY_MAP = {
        'back': 'KEYCODE_BACK',
        'home': 'KEYCODE_HOME',
        'enter': 'KEYCODE_ENTER',
        'power': 'KEYCODE_POWER',
        'menu': 'KEYCODE_MENU',
        'delete': 'KEYCODE_DEL',
    }
    keycode = _KEY_MAP.get(key.lower(), f'KEYCODE_{key.upper()}')

    try:
        controller = _get_controller()
        controller.press(keycode)
        return {
            "status": Status.SUCCESS.value,
            "message": f"Pressed: {key}"
        }
    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }


def adb_long_click(y_min: float, x_min: float, y_max: float, x_max: float) -> dict:
    """Long press at the center of a bounding box. Coordinates in box_2d format normalized 0-1000."""
    print(f"TOOL CALLED: adb_long_click(box=[{y_min}, {x_min}, {y_max}, {x_max}])")

    try:
        cx, cy = _box_to_pixel_center(y_min, x_min, y_max, x_max)
        controller = _get_controller()
        controller.long_press(cx, cy, duration_ms=500)
        return {
            "status": Status.SUCCESS.value,
            "message": f"Long pressed at pixel ({cx}, {cy})"
        }
    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }
