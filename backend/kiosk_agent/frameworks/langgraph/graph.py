"""Graph building and routing logic for LangGraph Kiosk Agent."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...types import AgentState
from ...utils import get_logger

logger = get_logger(__name__)


class GraphMixin:
    """
    Mixin class providing graph building and routing logic.
    
    Builds the LangGraph workflow with nodes:
    - vlm: VLM reasoning
    - execute: ADB execution
    - router: Route decision
    - human: Human-in-the-loop
    - analyze: Situation analysis
    """

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build LangGraph workflow.
        
        Graph structure:
        
        vlm -> execute -> router -+-> loop -> vlm
                                  +-> analyze -> loop/end
                                  +-> human -> resume/abort
                                  +-> end
        """
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("vlm", self._vlm_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("router", self._router_node)
        builder.add_node("human", self._human_node)
        builder.add_node("analyze", self._analyze_node)
        
        # Set entry point and edges
        builder.set_entry_point("vlm")
        builder.add_edge("vlm", "execute")
        builder.add_edge("execute", "router")
        
        # Router conditional edges
        builder.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "loop": "vlm",
                "analyze": "analyze",
                "human": "human",
                "end": END,
            },
        )
        
        # Analyze conditional edges
        builder.add_conditional_edges(
            "analyze",
            self._route_decision,
            {
                "loop": "vlm",
                "end": END,
            },
        )
        
        # Human conditional edges
        builder.add_conditional_edges(
            "human",
            self._human_route,
            {
                "resume": "vlm",
                "abort": END,
            },
        )
        
        return builder.compile()

    def _route_decision(self, state: AgentState) -> Literal["loop", "analyze", "human", "end"]:
        """Get route from state."""
        route = state.get("route", "end")
        if route in ("loop", "analyze", "human", "end"):
            return route
        return "end"

    def _human_route(self, state: AgentState) -> Literal["resume", "abort"]:
        """Get human review route."""
        if state.get("route") == "abort" or state.get("status") == "aborted":
            return "abort"
        return "resume"

    def _determine_route(self, state: AgentState) -> Literal["loop", "analyze", "human", "end"]:
        """
        Priority-based routing logic.
        
        Priority order:
        1. End conditions (task_complete, aborted)
        2. Human escalation (needs_human)
        3. Max iterations check
        4. Default: keep looping
        """
        status = state.get("status", "")
        iteration = state.get("iteration", 0)

        # End conditions
        if status in ["task_complete", "aborted"]:
            return "end"

        # Human escalation
        if status == "needs_human":
            return "human"

        # Max iterations
        if iteration >= self.max_iterations:
            return "end"

        # Default: keep looping
        return "loop"
