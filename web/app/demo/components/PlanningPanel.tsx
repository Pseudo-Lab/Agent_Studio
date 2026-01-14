"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ListChecks, Loader2, Globe, SlidersHorizontal, X } from "lucide-react";
import type { PlanningState } from "../types";

type Props = {
  planningState: PlanningState | null;
  isVisible: boolean;
  onClose?: () => void;
};

const statusConfig = {
  detecting_unknown: {
    icon: SlidersHorizontal,
    label: "분석 중",
    description: "요청에서 모르는 개념을 확인하고 있어요",
  },
  web_search_complete: {
    icon: Globe,
    label: "검색 완료",
    description: "웹에서 참고 정보를 가져왔어요",
  },
  web_search_skipped: {
    icon: Search,
    label: "검색 생략",
    description: "추가 검색 없이 바로 계획을 만들어요",
  },
  planning_complete: {
    icon: ListChecks,
    label: "계획 완료",
    description: "이제 실행을 시작할게요",
  },
};

export function PlanningPanel({ planningState, isVisible, onClose }: Props) {
  if (!isVisible || !planningState) return null;

  type StatusKey = keyof typeof statusConfig;
  const statusKey: StatusKey =
    planningState.status && planningState.status in statusConfig
      ? (planningState.status as StatusKey)
      : "detecting_unknown";
  const config = statusConfig[statusKey];
  const IconComponent = config.icon;
  const isLoading = statusKey === "detecting_unknown";
  const isMuted = statusKey === "web_search_skipped";

  const accent = {
    icon: isMuted ? "text-gray-400" : "text-amber-300",
    title: isMuted ? "text-gray-300" : "text-amber-300",
    pill: isMuted
      ? "bg-white/[0.04] border-white/[0.10] text-gray-300"
      : "bg-amber-400/10 border-amber-400/20 text-amber-300",
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -20, scale: 0.95 }}
        transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
        className="w-full p-3 rounded-[24px] border border-white/[0.08] bg-[#141416]/80 backdrop-blur-md shadow-[0_10px_30px_rgba(0,0,0,0.35)] relative overflow-hidden"
      >
        {/* top accent */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-400/35 to-transparent" />

        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-start gap-3 min-w-0">
            <div className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center flex-shrink-0">
              {isLoading ? (
                <Loader2 className={`w-4.5 h-4.5 ${accent.icon} animate-spin`} />
              ) : (
                <IconComponent className={`w-4.5 h-4.5 ${accent.icon}`} />
              )}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className={`text-xs font-bold ${accent.title}`}>{config.label}</span>
                <span
                  className={[
                    "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-widest border",
                    accent.pill,
                  ].join(" ")}
                >
                  PLAN
                </span>
              </div>
              <div className="text-[11px] text-gray-500 leading-relaxed truncate">{config.description}</div>
            </div>
          </div>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="w-8 h-8 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.06] hover:border-white/[0.10] transition-colors flex items-center justify-center text-gray-500 hover:text-gray-300 flex-shrink-0"
              title="닫기"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Unknown Entities */}
        {planningState.unknownEntities.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mb-3"
          >
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1.5 font-semibold">
              Unknown
            </div>
            <div className="flex flex-wrap gap-1.5">
              {planningState.unknownEntities.map((entity, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className="px-2.5 py-1 text-xs font-semibold bg-amber-400/10 text-amber-200 rounded-lg border border-amber-400/20"
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
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1.5 font-semibold">
              Context
            </div>
            <div className="p-3 bg-black/25 rounded-xl border border-white/[0.08] text-xs text-gray-400 line-clamp-3 leading-relaxed">
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
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2 font-semibold">
              Steps
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
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-amber-400/10 border border-amber-400/20 flex items-center justify-center mt-0.5">
                    <span className="text-[10px] font-bold text-amber-300">{i + 1}</span>
                  </div>
                  <span className="text-xs text-gray-300 leading-relaxed line-clamp-2">{step}</span>
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
            className="mt-2"
          >
            <div className="h-[2px] bg-white/[0.06] rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-amber-400/80 to-orange-400/80"
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
