"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Terminal } from "lucide-react";

import TargetChip from "@/components/TargetChip";
import ShinyText from "@/components/ShinyText";

import type { ChatMessage, PlanningState } from "../types";
import { PlanningPanel } from "./PlanningPanel";

type Props = {
  msg: ChatMessage;
  isExpanded: boolean;
  onToggle: () => void;
  actionColors: Record<string, string>;
  isValidBox2d: (box?: number[]) => boolean;
  enablePlanning: boolean;
  showPlanningPanel: boolean;
  planningState: PlanningState | null;
  isCurrentThinking: boolean;
  onClosePlanning: () => void;
};

export function ThinkingBlock({
  msg,
  isExpanded,
  onToggle,
  actionColors,
  isValidBox2d,
  enablePlanning,
  showPlanningPanel,
  planningState,
  isCurrentThinking,
  onClosePlanning,
}: Props) {
  if (msg.type !== "thinking") return null;

  const showPlan = Boolean(
    enablePlanning && showPlanningPanel && planningState && isCurrentThinking
  );

  return (
    <div className="mb-10 pl-1">
      {/* Thinking Header (subtle) */}
      <div className="opacity-50 hover:opacity-80 transition-opacity duration-300">
        <button
          onClick={onToggle}
          className="flex items-center gap-3 text-gray-400 hover:text-gray-200 transition-all group w-full text-left"
        >
          <div className="flex items-center justify-center w-5 h-5 rounded-full bg-white/[0.03] border border-white/[0.05] group-hover:bg-white/[0.08] transition-colors">
            <ChevronDown
              className={`w-3 h-3 transition-transform duration-300 ${
                isExpanded ? "" : "-rotate-90"
              }`}
            />
          </div>
          {msg.isStreaming ? (
            <ShinyText
              text="Agent is thinking..."
              className="text-[13px] font-medium tracking-tight"
              color="#6b7280"
              shineColor="#ffffff"
              speed={1.5}
              spread={100}
            />
          ) : (
            <span className="text-[13px] font-medium tracking-tight opacity-80">
              Thought process ({msg.steps?.length || 0} steps)
            </span>
          )}
          {msg.isStreaming && (
            <div className="flex gap-1 ml-1">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-1 h-1 rounded-full bg-emerald-500/40"
                  animate={{ opacity: [0.2, 1, 0.2] }}
                  transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                />
              ))}
            </div>
          )}
        </button>
      </div>

      {/* Thinking Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-5 ml-2.5 pl-6 border-l border-white/[0.1] space-y-10 py-1">
              {/* Planning panel lives in the step area */}
              {showPlan && (
                <div className="opacity-100">
                  <PlanningPanel
                    planningState={planningState}
                    isVisible={true}
                    onClose={onClosePlanning}
                  />
                </div>
              )}

              {msg.steps?.map((step, idx) => (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="relative group"
                >
                  {/* Minimal Timeline Dot */}
                  <div className="absolute -left-7 top-1.5 w-2 h-2 rounded-full bg-[#111114] border border-white/[0.2] group-hover:border-white/[0.4] transition-colors" />

                  <div className="space-y-3.5">
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] font-bold text-gray-600 uppercase tracking-[0.15em]">
                        Step {step.iteration || idx + 1}
                      </span>
                      {step.action && (
                        <span
                          className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-sm bg-white/[0.03] ${
                            actionColors[step.action] || "text-gray-500"
                          }`}
                        >
                          {step.action}
                        </span>
                      )}
                    </div>

                    <p className="text-[14px] text-gray-300 leading-[1.7] font-normal max-w-[95%]">
                      {step.thought}
                    </p>

                    {/* Sub-data - very subtle */}
                    <div className="flex flex-wrap gap-3 pt-1">
                      {typeof step.progress === "boolean" && (
                        <div
                          className={`font-mono text-[11px] px-2.5 py-1 rounded inline-flex items-center gap-2 ${
                            step.progress
                              ? "text-emerald-400/70 bg-emerald-500/5"
                              : "text-gray-600 bg-white/[0.02]"
                          }`}
                        >
                          <span className="opacity-60">result</span>
                          <span className="font-semibold">
                            {step.progress ? "progress" : "no-progress"}
                          </span>
                        </div>
                      )}

                      {typeof step.difference === "number" && (
                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                          <span className="opacity-60">Δ</span>
                          <span className="font-semibold">
                            {step.difference.toFixed(4)}
                          </span>
                        </div>
                      )}

                      {step.status && (
                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                          <span className="opacity-60">status</span>
                          <span className="font-semibold">
                            {String(step.status)}
                          </span>
                        </div>
                      )}

                      {step.adb_commands && step.adb_commands.length > 0 && (
                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                          <Terminal className="w-2.5 h-2.5 opacity-40" />
                          <span className="opacity-20">$</span>{" "}
                          {String(step.adb_commands[0] || "").substring(0, 45)}
                          ...
                        </div>
                      )}

                      {(() => {
                        const box = step.box_2d;
                        if (!isValidBox2d(box)) return null;
                        return <TargetChip box={box} />;
                      })()}
                    </div>
                  </div>
                </motion.div>
              ))}

              {/* Streaming Indicator inside expanded view */}
              {msg.isStreaming && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="relative"
                >
                  <div className="absolute -left-7 top-1.5 w-2 h-2 rounded-full bg-emerald-500/30 animate-pulse" />
                  <span className="text-[13px] text-gray-600 font-medium animate-pulse pl-1">
                    Thinking...
                  </span>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

