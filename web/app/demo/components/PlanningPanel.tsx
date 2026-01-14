"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Search, 
  ListChecks, 
  Loader2, 
  Globe, 
  SlidersHorizontal, 
  ChevronDown, 
  ChevronRight,
  Check,
  Circle,
  ArrowRight,
} from "lucide-react";
import type { PlanningState } from "../types";

type Props = {
  planningState: PlanningState | null;
  isVisible: boolean;
  onClose?: () => void;
};

const statusConfig = {
  detecting_unknown: {
    label: "Analyzing",
    generating: true,
  },
  web_search_complete: {
    label: "Searching",
    generating: false,
  },
  web_search_skipped: {
    label: "Planning",
    generating: false,
  },
  planning_complete: {
    label: "Executing",
    generating: false,
  },
};

export function PlanningPanel({ planningState, isVisible }: Props) {
  const [isExpanded, setIsExpanded] = useState(true);
  
  if (!isVisible || !planningState) return null;

  type StatusKey = keyof typeof statusConfig;
  const statusKey: StatusKey =
    planningState.status && planningState.status in statusConfig
      ? (planningState.status as StatusKey)
      : "detecting_unknown";
  const config = statusConfig[statusKey];
  
  const totalSteps = planningState.plan.length;
  const completedSteps = planningState.planStepIndex || 0;
  const hasSteps = totalSteps > 0;
  const isGenerating = config.generating || (planningState.status !== "planning_complete");

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
      className="w-full"
    >
      {/* Cursor-style To-dos Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full text-left group"
      >
        {/* Expand/Collapse Icon */}
        <div className="text-gray-500 group-hover:text-gray-400 transition-colors">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </div>
        
        {/* List Icon */}
        <SlidersHorizontal className="w-4 h-4 text-gray-500" />
        
        {/* Title with count */}
        <span className="text-sm text-gray-400">
          To-dos
        </span>
        {hasSteps && (
          <span className="text-sm text-gray-500 ml-1">
            {totalSteps}
          </span>
        )}
      </button>

      {/* To-do Items */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 ml-6 space-y-1">
              {/* Steps */}
              {planningState.plan.map((step, i) => {
                const isCompleted = i < completedSteps;
                const isCurrent = i === completedSteps && planningState.status === "planning_complete";
                const isPending = i > completedSteps || (i === completedSteps && planningState.status !== "planning_complete");

                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -5 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-start gap-2 py-0.5"
                  >
                    {/* Step indicator */}
                    <div className="flex-shrink-0 mt-0.5">
                      {isCompleted ? (
                        <div className="w-4 h-4 rounded-full bg-transparent flex items-center justify-center">
                          <Check className="w-3.5 h-3.5 text-gray-500" strokeWidth={2.5} />
                        </div>
                      ) : isCurrent ? (
                        <motion.div
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                          className="w-4 h-4 flex items-center justify-center"
                        >
                          <ArrowRight className="w-3.5 h-3.5 text-amber-400" />
                        </motion.div>
                      ) : (
                        <div className="w-4 h-4 flex items-center justify-center">
                          <Circle className="w-3 h-3 text-gray-600" />
                        </div>
                      )}
                    </div>
                    
                    {/* Step text */}
                    <span
                      className={`text-sm leading-relaxed ${
                        isCompleted
                          ? "text-gray-500 line-through decoration-gray-600"
                          : isCurrent
                          ? "text-gray-300"
                          : "text-gray-400"
                      }`}
                    >
                      {step}
                    </span>
                  </motion.div>
                );
              })}

              {/* Generating indicator (like Cursor) */}
              {isGenerating && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2 py-0.5 mt-1"
                >
                  <div className="w-4 h-4 flex items-center justify-center">
                    <Loader2 className="w-3.5 h-3.5 text-gray-500 animate-spin" />
                  </div>
                  <span className="text-sm text-gray-500">
                    {planningState.status === "detecting_unknown" && "Analyzing request..."}
                    {planningState.status === "web_search_complete" && "Searching web..."}
                    {planningState.status === "web_search_skipped" && "Creating plan..."}
                    {!planningState.status && "Generating."}
                  </span>
                </motion.div>
              )}

              {/* Search targets (prettier UI) */}
              {planningState.unknownEntities.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-2 py-1.5 mt-1"
                >
                  <div className="flex items-center gap-1.5 text-[11px] text-gray-500 font-medium">
                    <Search className="w-3 h-3" />
                    <span>Targets</span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {planningState.unknownEntities.map((entity, i) => (
                      <motion.span
                        key={i}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="inline-flex items-center gap-1 text-xs font-semibold text-amber-200 bg-gradient-to-r from-amber-500/20 to-orange-500/20 px-2.5 py-1 rounded-full border border-amber-400/30 shadow-[0_0_8px_rgba(251,191,36,0.15)]"
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                        {entity}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Search context result (show what was found) */}
              {planningState.searchContext && planningState.searchContext.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="mt-2 py-2"
                >
                  <div className="flex items-center gap-1.5 text-[11px] text-gray-500 font-medium mb-1.5">
                    <Globe className="w-3 h-3" />
                    <span>Search Result</span>
                    <span className="text-emerald-400/70">✓</span>
                  </div>
                  <div className="p-2.5 bg-black/30 rounded-lg border border-white/[0.06] text-[11px] text-gray-400 leading-relaxed max-h-24 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10">
                    {planningState.searchContext.slice(0, 300)}
                    {planningState.searchContext.length > 300 && (
                      <span className="text-gray-600">...</span>
                    )}
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
