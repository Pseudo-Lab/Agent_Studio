#!/usr/bin/env python3
"""
Kiosk Agent CLI runner.

Usage:
    python run_agent.py "빅맥 세트 주문해줘"
    python run_agent.py --model gemini "행운버거 주문해줘"
"""

import argparse
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_PATH = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))


def main():
    parser = argparse.ArgumentParser(description="Run Kiosk Agent")
    parser.add_argument("instruction", help="Instruction to execute")
    parser.add_argument(
        "--model", 
        default="gemini",
        choices=["gemini", "openai", "local"],
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ADB commands without executing"
    )
    args = parser.parse_args()
    
    # Import after path setup
    from kiosk_agent import Config
    from kiosk_agent.frameworks import LangGraphAgent
    
    # Configure
    config = Config.from_env()
    config.model.provider = args.model
    
    # Create and run agent
    agent = LangGraphAgent(config, dry_run=args.dry_run)
    
    print(f"[Agent] Starting with instruction: {args.instruction}")
    print(f"[Agent] Model: {config.model.provider}")
    print("-" * 50)
    
    # Stream execution
    for event in agent.stream(args.instruction):
        if event.thought:
            print(f"[Step {event.state.get('iteration', 0)}] {event.thought}")
        if event.requires_human_input:
            print("\n[HITL] Agent needs input:")
            interrupt = event.state.get("payload", {}).get("interrupt", {})
            question = interrupt.get("question", "What should I do?")
            print(f"  Question: {question}")
            
            options = interrupt.get("options", [])
            if options:
                print("  Options:")
                for i, opt in enumerate(options, 1):
                    print(f"    {i}. {opt}")
            
            response = input("\n  Your response: ").strip()
            if response.lower() in {"quit", "exit", "종료"}:
                print("[Agent] Aborted by user")
                break
            
            # Resume with response
            for resume_event in agent.resume(event.state, response):
                if resume_event.thought:
                    print(f"[Step {resume_event.state.get('iteration', 0)}] {resume_event.thought}")
    
    print("-" * 50)
    print("[Agent] Execution complete")


if __name__ == "__main__":
    main()
