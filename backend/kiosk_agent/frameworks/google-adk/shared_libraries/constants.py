"""Status constants shared across all ADK sub-agents and tools."""

from enum import Enum


class Status(Enum):
    """Standardised status codes returned by every tool function.

    Used by the orchestrator and status_agent to decide control flow
    (e.g. exit_loop on FINISH, retry on FAIL).
    """

    FINISH = "finish"          # Goal achieved -> exit loop
    CONTINUE = "continue"      # Goal not yet achieved -> next iteration
    SATISFIED = "satisfied"    # Goal satisfied (alias)
    IMPOSSIBLE = "impossible"  # Goal cannot be achieved
    ERROR = "error"            # Unexpected error
    FAIL = "fail"              # Tool execution failed (retryable)
    SUCCESS = "success"        # Tool execution succeeded
