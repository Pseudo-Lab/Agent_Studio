"""Base agent interface for all framework implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from ..config import AgentConfig
from ..types import AgentState, AgentStepResult, AgentStreamEvent


class BaseAgent(ABC):
    """
    Abstract base class for Kiosk Agent implementations.
    
    Provides a common interface across different agent frameworks
    (LangGraph, CrewAI, Microsoft Agent Framework, Google ADK).
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    def run(self, instruction: str) -> AgentStepResult:
        """
        Execute agent synchronously.
        
        Args:
            instruction: User instruction for the task
            
        Returns:
            Final result of agent execution
        """

    @abstractmethod
    def stream(
        self,
        instruction: str,
        *,
        thread_id: Optional[str] = None,
    ) -> Iterator[AgentStreamEvent]:
        """
        Execute agent with streaming output.
        
        Args:
            instruction: User instruction for the task
            thread_id: Optional thread ID for session tracking
            
        Yields:
            Stream events during execution
        """

    @abstractmethod
    def init_state(self, instruction: str) -> AgentState:
        """
        Initialize agent state.
        
        Args:
            instruction: User instruction for the task
            
        Returns:
            Initial agent state
        """

    @abstractmethod
    def prepare_workflow(
        self,
        instruction: str,
        previous_state: Optional[AgentState] = None,
    ) -> tuple[Any, AgentState]:
        """
        Prepare workflow graph and initial state.
        
        Args:
            instruction: User instruction
            previous_state: Optional state to resume from
            
        Returns:
            Tuple of (compiled graph, initial state)
        """
