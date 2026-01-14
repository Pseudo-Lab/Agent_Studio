"""Planning Mode implementation for LangGraph Kiosk Agent.

Provides unified planning node with:
1. Detect unknown entities in user instruction
2. Web search for context enrichment (Tavily tool)
3. Generate step-by-step plan

All in a single node to simplify graph structure.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ...types import AgentState
from ...utils import get_logger
from .prompts import DETECT_UNKNOWN_PROMPT, PLAN_GENERATION_PROMPT, REPLAN_PROMPT

logger = get_logger(__name__)


class PlanningMixin:
    """
    Mixin class providing Planning Mode functionality.
    
    Single unified node:
    - planning: Detect unknown → Search (tool) → Generate plan
    """

    _tavily_tool: Optional[Any] = None
    _planning_initialized: bool = False

    def _init_planning_tools(self) -> None:
        """Initialize Tavily search tool."""
        if self._planning_initialized:
            return
            
        tavily_api_key = self.config.planning.tavily_api_key
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        
        if tavily_api_key:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
                
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                search_wrapper = TavilySearchAPIWrapper()
                self._tavily_tool = TavilySearchResults(
                    api_wrapper=search_wrapper,
                    max_results=self.config.planning.max_search_results,
                )
                logger.info("Tavily search tool initialized")
            except ImportError:
                logger.warning("langchain-community not installed, web search disabled")
                self._tavily_tool = None
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily: {e}")
                self._tavily_tool = None
        else:
            logger.warning("TAVILY_API_KEY not set, web search disabled")
            self._tavily_tool = None
        
        self._planning_initialized = True

    def _planning_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Unified planning node: detect → search (tool) → plan.
        
        Combines all planning steps into a single node for simpler graph.
        Uses Tavily as a tool internally rather than a separate node.
        """
        existing_plan = state.get("plan") or []
        if state.get("planning_complete") and existing_plan:
            logger.info("[Planning] Skipping (existing plan)")
            return {
                "plan": existing_plan,
                "plan_step_index": int(state.get("plan_step_index") or 0),
                "planning_complete": True,
                "status": "planning_skipped",
            }

        instruction = state.get("instruction", "")
        logger.info(f"[Planning] Starting for: {instruction[:50]}...")
        
        # Step 1: Detect unknown entities
        unknown_entities = self._detect_unknown_entities(instruction)
        
        # Step 2: Search for context using Tavily tool (if needed)
        search_context = ""
        search_status = "web_search_skipped"
        
        if unknown_entities and self._tavily_tool:
            search_context = self._search_with_tavily(unknown_entities)
            search_status = "web_search_complete" if search_context else "web_search_failed"
        
        # Step 3: Generate plan
        plan, enriched_instruction = self._generate_plan(
            instruction, 
            search_context,
            self.config.planning.max_plan_steps,
        )
        
        logger.info(f"[Planning] Complete: {len(plan)} steps, search={search_status}")
        
        return {
            "plan": plan,
            "plan_step_index": 0,
            "instruction": enriched_instruction,
            "original_instruction": instruction,
            "unknown_entities": unknown_entities,
            "search_context": search_context[:500] if search_context else "",
            "planning_complete": True,
            "status": "planning_complete",
        }

    def _detect_unknown_entities(self, instruction: str) -> List[str]:
        """Detect unknown entities using LLM."""
        prompt = DETECT_UNKNOWN_PROMPT.format(instruction=instruction)
        
        try:
            response = self.model_client.generate_text(prompt)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                entities = result.get("unknown_entities", [])
                logger.info(f"[Planning] Unknown entities: {entities}")
                return entities
        except Exception as e:
            logger.warning(f"[Planning] Detection failed: {e}")
        
        return []

    def _search_with_tavily(self, entities: List[str]) -> str:
        """Use Tavily tool to search for entity context."""
        if not self._tavily_tool:
            return ""
        
        logger.info(f"[Planning] Searching with Tavily for: {entities}")
        search_results: List[str] = []
        
        for entity in entities:
            try:
                query = f"{entity} 메뉴 가격 정보"
                results = self._tavily_tool.invoke(query)
                
                if isinstance(results, list):
                    for r in results[:2]:
                        if isinstance(r, dict):
                            content = r.get("content", "")
                            if content:
                                search_results.append(f"[{entity}]: {content[:500]}")
                        elif isinstance(r, str):
                            search_results.append(f"[{entity}]: {r[:500]}")
            except Exception as e:
                logger.warning(f"[Planning] Tavily search failed for '{entity}': {e}")
        
        context = "\n".join(search_results)
        logger.info(f"[Planning] Search context: {len(context)} chars")
        return context

    def _generate_plan(
        self, 
        instruction: str, 
        search_context: str,
        max_steps: int,
    ) -> tuple[List[str], str]:
        """Generate step-by-step plan using LLM."""
        context_section = ""
        if search_context:
            context_section = f"\n추가 정보 (웹 검색 결과):\n{search_context}\n"
        
        prompt = PLAN_GENERATION_PROMPT.format(
            instruction=instruction,
            context_section=context_section,
            max_steps=max_steps,
        )
        
        try:
            response = self.model_client.generate_text(prompt)
            json_match = re.search(r'\{[^{}]*"plan"[^{}]*\}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                plan = result.get("plan", [])
                logger.info(f"[Planning] Generated {len(plan)} steps")
                
                enriched = instruction
                if search_context:
                    enriched = f"{instruction}\n\n[참고 정보]:\n{search_context[:1000]}"
                
                return plan, enriched
        except Exception as e:
            logger.warning(f"[Planning] Plan generation failed: {e}")
        
        return [f"작업 수행: {instruction}"], instruction

    def _replan_node(self, state: AgentState) -> Dict[str, Any]:
        """Re-plan during execution if situation changes."""
        instruction = state.get("original_instruction") or state.get("instruction", "")
        plan = state.get("plan", [])
        plan_step_index = state.get("plan_step_index", 0)
        
        completed_steps = plan[:plan_step_index] if plan_step_index > 0 else []
        current_status = state.get("thought", "")
        
        logger.info(f"[Planning] Re-planning at step {plan_step_index}")
        
        prompt = REPLAN_PROMPT.format(
            instruction=instruction,
            plan=plan,
            completed_steps=completed_steps,
            current_status=current_status,
        )
        
        try:
            response = self.model_client.generate_text(prompt)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                action = result.get("action", "continue")
                
                if action == "replan":
                    new_plan = result.get("new_plan", [])
                    logger.info(f"[Planning] Re-planned with {len(new_plan)} new steps")
                    return {
                        "plan": new_plan,
                        "plan_step_index": 0,
                        "status": "replanned",
                    }
        except Exception as e:
            logger.warning(f"[Planning] Re-plan failed: {e}")
        
        return {"status": "plan_continues"}
