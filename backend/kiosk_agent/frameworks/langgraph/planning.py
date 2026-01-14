"""Planning Mode implementation for LangGraph Kiosk Agent.

Provides task decomposition and web search capabilities:
1. Detect unknown entities in user instruction
2. Web search for context enrichment (Tavily)
3. Generate step-by-step plan
4. Re-plan during execution if needed
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from ...types import AgentState
from ...utils import get_logger

logger = get_logger(__name__)

# Prompts for planning
DETECT_UNKNOWN_PROMPT = """다음 명령에서 일반적으로 알려지지 않은 고유명사, 브랜드명, 또는 특수한 메뉴/상품명이 있는지 확인하세요.

명령: {instruction}

규칙:
- "아메리카노", "카페라떼" 같은 일반적인 메뉴는 알려진 것으로 판단
- "대상혁버거", "민정이네라떼" 같은 특수한 이름은 알려지지 않은 것으로 판단
- 브랜드명이나 매장명이 특이한 경우도 알려지지 않은 것으로 판단

응답 형식 (JSON):
{{"unknown_entities": ["엔티티1", "엔티티2"], "reasoning": "판단 이유"}}

알 수 없는 개념이 없다면:
{{"unknown_entities": [], "reasoning": "모든 개념이 일반적으로 알려진 것입니다."}}
"""

PLAN_GENERATION_PROMPT = """키오스크에서 다음 작업을 완료하기 위한 단계별 계획을 세우세요.

목표: {instruction}
{context_section}

규칙:
1. 각 단계는 구체적이고 실행 가능해야 합니다
2. 키오스크 UI 조작에 필요한 단계만 포함하세요
3. 주관적 선택이 필요한 부분은 "사용자에게 묻기"로 표시
4. 최대 {max_steps}단계 이내로 작성

응답 형식 (JSON):
{{"plan": ["1단계: 메뉴 탭 클릭", "2단계: 카테고리 선택", ...], "reasoning": "계획 수립 이유"}}
"""

REPLAN_PROMPT = """현재 진행 상황을 바탕으로 계획을 수정해야 하는지 확인하세요.

원래 목표: {instruction}
원래 계획: {plan}
완료된 단계: {completed_steps}
현재 상태: {current_status}

계획대로 진행 가능하면:
{{"action": "continue", "reasoning": "계획대로 진행 가능"}}

계획 수정이 필요하면:
{{"action": "replan", "new_plan": ["새로운 단계1", ...], "reasoning": "수정 이유"}}
"""


class PlanningMixin:
    """
    Mixin class providing Planning Mode functionality.
    
    Nodes:
    - detect_unknown: Detect unknown entities in instruction
    - web_search: Search for unknown entities via Tavily
    - plan: Generate step-by-step plan
    - replan: Update plan during execution
    """

    _tavily_tool: Optional[Any] = None
    _planning_initialized: bool = False

    def _init_planning_tools(self) -> None:
        """Initialize planning tools (Tavily for web search)."""
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

    def _detect_unknown_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Detect unknown entities in user instruction.
        
        Uses LLM to identify proper nouns, brand names, or special 
        menu items that may require web search for context.
        """
        instruction = state.get("instruction", "")
        logger.info(f"[Planning] Detecting unknown entities in: {instruction[:50]}...")
        
        prompt = DETECT_UNKNOWN_PROMPT.format(instruction=instruction)
        
        try:
            # Use the existing model client for detection
            response = self.model_client.generate_text(prompt)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                unknown_entities = result.get("unknown_entities", [])
                reasoning = result.get("reasoning", "")
                
                logger.info(f"[Planning] Unknown entities: {unknown_entities}, reason: {reasoning}")
                
                return {
                    "unknown_entities": unknown_entities,
                    "original_instruction": instruction,
                    "status": "detecting_unknown",
                }
            else:
                logger.warning("[Planning] Could not parse detection response")
                return {
                    "unknown_entities": [],
                    "original_instruction": instruction,
                    "status": "detecting_unknown",
                }
                
        except Exception as e:
            logger.error(f"[Planning] Detection failed: {e}")
            return {
                "unknown_entities": [],
                "original_instruction": instruction,
                "status": "detecting_unknown",
            }

    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Search for unknown entities using Tavily.
        
        Enriches context with web search results for better planning.
        """
        unknown_entities = state.get("unknown_entities", [])
        
        if not unknown_entities:
            logger.info("[Planning] No unknown entities, skipping web search")
            return {"search_context": "", "status": "web_search_skipped"}
        
        if not self._tavily_tool:
            logger.warning("[Planning] Tavily not available, skipping web search")
            return {"search_context": "", "status": "web_search_unavailable"}
        
        logger.info(f"[Planning] Searching for: {unknown_entities}")
        
        search_results: List[str] = []
        
        for entity in unknown_entities:
            try:
                query = f"{entity} 메뉴 가격 정보"
                results = self._tavily_tool.invoke(query)
                
                if isinstance(results, list):
                    for r in results[:2]:  # Top 2 results per entity
                        if isinstance(r, dict):
                            content = r.get("content", "")
                            if content:
                                search_results.append(f"[{entity}]: {content[:500]}")
                        elif isinstance(r, str):
                            search_results.append(f"[{entity}]: {r[:500]}")
                            
            except Exception as e:
                logger.warning(f"[Planning] Search failed for '{entity}': {e}")
        
        context = "\n".join(search_results) if search_results else ""
        logger.info(f"[Planning] Search context length: {len(context)} chars")
        
        return {
            "search_context": context,
            "status": "web_search_complete",
        }

    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate step-by-step plan for the task.
        
        Uses instruction and search context to create actionable plan.
        """
        instruction = state.get("original_instruction") or state.get("instruction", "")
        search_context = state.get("search_context", "")
        max_steps = self.config.planning.max_plan_steps
        
        logger.info(f"[Planning] Generating plan for: {instruction[:50]}...")
        
        # Build context section
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
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{[^{}]*"plan"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                plan = result.get("plan", [])
                reasoning = result.get("reasoning", "")
                
                logger.info(f"[Planning] Generated plan with {len(plan)} steps: {reasoning}")
                
                # Enrich instruction with context if available
                enriched_instruction = instruction
                if search_context:
                    enriched_instruction = f"{instruction}\n\n[참고 정보]:\n{search_context[:1000]}"
                
                return {
                    "plan": plan,
                    "plan_step_index": 0,
                    "instruction": enriched_instruction,
                    "planning_complete": True,
                    "status": "planning_complete",
                }
            else:
                logger.warning("[Planning] Could not parse plan response, using default")
                return {
                    "plan": [f"작업 수행: {instruction}"],
                    "plan_step_index": 0,
                    "planning_complete": True,
                    "status": "planning_complete",
                }
                
        except Exception as e:
            logger.error(f"[Planning] Plan generation failed: {e}")
            return {
                "plan": [f"작업 수행: {instruction}"],
                "plan_step_index": 0,
                "planning_complete": True,
                "status": "planning_complete",
            }

    def _should_search(self, state: AgentState) -> Literal["search", "plan"]:
        """
        Routing decision: should we search or skip to planning?
        """
        unknown_entities = state.get("unknown_entities", [])
        
        if unknown_entities and self._tavily_tool:
            return "search"
        return "plan"

    def _replan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Re-plan during execution if situation changes.
        
        Called when VLA loop detects unexpected situation.
        """
        instruction = state.get("original_instruction") or state.get("instruction", "")
        plan = state.get("plan", [])
        plan_step_index = state.get("plan_step_index", 0)
        history = state.get("history", [])
        
        # Build completed steps
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
            
            import json
            import re
            
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
                else:
                    logger.info("[Planning] Continuing with existing plan")
                    return {"status": "plan_continues"}
                    
        except Exception as e:
            logger.warning(f"[Planning] Re-plan failed: {e}")
        
        return {"status": "plan_continues"}
