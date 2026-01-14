"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ListChecks, Loader2, Globe, Sparkles, SlidersHorizontal } from "lucide-react";
import type { PlanningState } from "../types";

type Props = {
  planningState: PlanningState | null;
  isVisible: boolean;
};

const statusConfig = {
  detecting_unknown: {
    icon: SlidersHorizontal,
    label: "분석 중",
    description: "요청에서 모르는 개념을 찾고 있어요",
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/25",
  },
  web_search_complete: {
    icon: Globe,
    label: "검색 완료",
    description: "웹에서 정보를 찾았어요",
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30",
  },
  web_search_skipped: {
    icon: Search,
    label: "검색 건너뜀",
    description: "모든 개념이 알려진 것이에요",
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    borderColor: "border-gray-500/30",
  },
  planning_complete: {
    icon: ListChecks,
    label: "계획 완료",
    description: "실행 계획을 세웠어요",
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30",
  },
};

export function PlanningPanel({ planningState, isVisible }: Props) {
  if (!isVisible || !planningState) return null;

  const config = statusConfig[planningState.status] || statusConfig.detecting_unknown;
  const IconComponent = config.icon;
  const isLoading = planningState.status === "detecting_unknown";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.95 }}
        transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
        className={`mx-4 mb-4 p-4 rounded-2xl border ${config.bgColor} ${config.borderColor} backdrop-blur-sm`}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-xl ${config.bgColor}`}>
            {isLoading ? (
              <Loader2 className={`w-5 h-5 ${config.color} animate-spin`} />
            ) : (
              <IconComponent className={`w-5 h-5 ${config.color}`} />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-bold ${config.color}`}>{config.label}</span>
              <Sparkles className={`w-3.5 h-3.5 ${config.color} opacity-60`} />
            </div>
            <span className="text-xs text-gray-400">{config.description}</span>
          </div>
        </div>

        {/* Unknown Entities */}
        {planningState.unknownEntities.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mb-3"
          >
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5 font-semibold">
              발견된 미지의 개념
            </div>
            <div className="flex flex-wrap gap-1.5">
              {planningState.unknownEntities.map((entity, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="px-2.5 py-1 text-xs font-medium bg-amber-500/15 text-amber-300 rounded-lg border border-amber-500/25"
                >
                  {entity}
                </motion.span>
              ))}
            </div>
          </motion.div>
        )}

        {/* Search Context Preview */}
        {planningState.searchContext && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mb-3"
          >
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5 font-semibold">
              웹 검색 결과
            </div>
            <div className="p-2.5 bg-black/30 rounded-xl border border-white/5 text-xs text-gray-400 line-clamp-3">
              {planningState.searchContext.slice(0, 200)}
              {planningState.searchContext.length > 200 && "..."}
            </div>
          </motion.div>
        )}

        {/* Plan Steps */}
        {planningState.plan.length > 0 && planningState.status === "planning_complete" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-semibold">
              실행 계획
            </div>
            <div className="space-y-1.5">
              {planningState.plan.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="flex items-start gap-2.5"
                >
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center mt-0.5">
                    <span className="text-[10px] font-bold text-emerald-400">{i + 1}</span>
                  </div>
                  <span className="text-xs text-gray-300 leading-relaxed">{step}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Progress indicator for detecting */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-3"
          >
            <div className="h-1 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-amber-500 to-amber-400"
                animate={{ x: ["-100%", "100%"] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                style={{ width: "50%" }}
              />
            </div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
