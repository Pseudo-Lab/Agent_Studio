"""Schema definitions for VLM output parsing."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InterruptInfo(BaseModel):
    """Information for human-in-the-loop interrupts."""
    reason: Literal["AMBIGUOUS_CHOICE", "MISSING_INFO", "CONFIRMATION_REQUIRED"]
    question: str
    # Optional multiple-choice list (objective selections). Either key may be used.
    options: Optional[List[str]] = None
    question_list: Optional[List[str]] = None
    # 텍스트 입력이 필요한 경우 (검색어, 전화번호, 주소 등)
    requires_text_input: Optional[bool] = None


class GUI_OUTPUT(BaseModel):
    """Schema for GUI action output from VLM."""
    thought: str
    action: Literal["CLICK", "LONG_CLICK", "SWIPE", "INPUT", "BACK", "HOME", "INTERRUPT", "FINISH", "ABORT"]
    # NOTE: Always include box_2d (use [0,0,0,0] when not applicable).
    box_2d: List[float] = Field(
        min_length=4, 
        max_length=4, 
        description="[ymin, xmin, ymax, xmax], normalized 0-1000"
    )
    # Value to type for INPUT. For non-INPUT actions, use null.
    value: Optional[str] = Field(
        description="Text to input when action == INPUT (non-empty string). Otherwise null."
    )
    interrupt: Optional[InterruptInfo] = None


class GUI_OUTPUT_PLANNING(BaseModel):
    """Schema for GUI action output from VLM (planning mode)."""
    thought: str
    action: Literal["CLICK", "LONG_CLICK", "SWIPE", "INPUT", "BACK", "HOME", "INTERRUPT", "FINISH", "ABORT"]
    box_2d: List[float] = Field(
        min_length=4,
        max_length=4,
        description="[ymin, xmin, ymax, xmax], normalized 0-1000",
    )
    value: Optional[str] = Field(
        description="Text to input when action == INPUT (non-empty string). Otherwise null."
    )
    interrupt: Optional[InterruptInfo] = None
    step_decision: Literal["repeat", "advance", "abort"] = Field(
        description="Planning step decision for the current to-do.",
    )
