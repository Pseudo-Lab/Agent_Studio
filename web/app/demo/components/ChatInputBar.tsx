"use client";

import React, { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, CheckCircle2, ChevronUp, Loader2, Mic, SlidersHorizontal, Square, X } from "lucide-react";

type Props = {
  inputValue: string;
  setInputValue: (v: string) => void;
  requiresHumanInput: boolean;
  isRunning: boolean;

  // Voice input
  isRecording: boolean;
  isTranscribing: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;

  // Model select
  selectedModel: string;
  setSelectedModel: (m: string) => void;
  showModelSelector: boolean;
  setShowModelSelector: (v: boolean) => void;

  // Planning mode
  enablePlanning: boolean;
  setEnablePlanning: (v: boolean) => void;

  // Submit / stop
  onSubmit: (e: React.FormEvent) => void;
  onStop: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disableSubmit: boolean;
};

// 음파 애니메이션 컴포넌트
function VoiceWaveAnimation() {
  return (
    <div className="flex items-center justify-center gap-1">
      {[...Array(5)].map((_, i) => (
        <motion.div
          key={i}
          className="w-1 bg-red-400 rounded-full"
          animate={{
            height: [12, 28, 12],
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            delay: i * 0.1,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// 녹음 시간 표시 컴포넌트
function RecordingTimer({ isRecording }: { isRecording: boolean }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!isRecording) {
      setSeconds(0);
      return;
    }
    const interval = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [isRecording]);

  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return (
    <span className="font-mono text-2xl text-white/90 tabular-nums">
      {String(mins).padStart(2, "0")}:{String(secs).padStart(2, "0")}
    </span>
  );
}

export function ChatInputBar({
  inputValue,
  setInputValue,
  requiresHumanInput,
  isRunning,
  isRecording,
  isTranscribing,
  onStartRecording,
  onStopRecording,
  selectedModel,
  setSelectedModel,
  showModelSelector,
  setShowModelSelector,
  enablePlanning,
  setEnablePlanning,
  onSubmit,
  onStop,
  disableSubmit,
}: Props) {
  return (
    <>
      {/* 녹음 오버레이 모달 */}
      <AnimatePresence>
        {(isRecording || isTranscribing) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
              className="relative flex flex-col items-center gap-8 p-10 rounded-3xl bg-gradient-to-b from-[#1c1c22] to-[#141418] border border-white/10 shadow-2xl"
            >
              {/* 닫기 버튼 */}
              {!isTranscribing && (
                <button
                  onClick={onStopRecording}
                  className="absolute top-4 right-4 p-2 rounded-full text-white/40 hover:text-white/70 hover:bg-white/10 transition-colors"
                  title="취소"
                >
                  <X className="w-5 h-5" />
                </button>
              )}

              {/* 상태 텍스트 */}
              <div className="text-center">
                <h3 className="text-lg font-semibold text-white mb-1">
                  {isTranscribing ? "변환 중..." : "듣고 있어요"}
                </h3>
                <p className="text-sm text-white/50">
                  {isTranscribing ? "음성을 텍스트로 변환하고 있습니다" : "말씀해 주세요"}
                </p>
              </div>

              {/* 음파 애니메이션 또는 로딩 */}
              <div className="w-32 h-16 flex items-center justify-center">
                {isTranscribing ? (
                  <Loader2 className="w-10 h-10 text-emerald-400 animate-spin" />
                ) : (
                  <VoiceWaveAnimation />
                )}
              </div>

              {/* 녹음 시간 */}
              {!isTranscribing && <RecordingTimer isRecording={isRecording} />}

              {/* 중지 버튼 */}
              {!isTranscribing && (
                <button
                  onClick={onStopRecording}
                  className="flex items-center gap-3 px-8 py-4 rounded-2xl bg-red-500 hover:bg-red-600 text-white font-semibold transition-all shadow-lg shadow-red-500/30 hover:shadow-red-500/40"
                >
                  <Square className="w-5 h-5" />
                  <span>녹음 완료</span>
                </button>
              )}

              {/* 펄스 링 */}
              {!isTranscribing && (
                <div className="absolute inset-0 -z-10 flex items-center justify-center pointer-events-none">
                  <motion.div
                    className="w-48 h-48 rounded-full border-2 border-red-500/30"
                    animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                  />
                  <motion.div
                    className="absolute w-48 h-48 rounded-full border-2 border-red-500/30"
                    animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut", delay: 0.5 }}
                  />
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-6 bg-transparent relative">
        <form onSubmit={onSubmit} className="relative flex items-center gap-3">
          <div className="flex-1 relative flex items-center">
            {/* 녹음 버튼 - 진하게 */}
            <button
              type="button"
              onClick={onStartRecording}
              disabled={isTranscribing || isRecording}
              className="absolute left-2 p-2.5 rounded-xl transition-all duration-300 z-10 bg-[#1a1a1f] text-gray-400 hover:text-white hover:bg-[#252528] border border-white/[0.08]"
              title="음성 입력"
            >
              <Mic className="w-4.5 h-4.5" />
            </button>

            {/* 텍스트 입력 - 진하게 */}
            <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={requiresHumanInput ? "응답을 입력하세요..." : "무엇을 도와드릴까요?"}
            className={[
              "w-full pl-14 pr-32 py-4 rounded-[20px] bg-[#1a1a1f] border border-white/[0.10] text-white placeholder:text-gray-500 placeholder:font-medium focus:outline-none focus:border-white/[0.20] focus:bg-[#1e1e23] transition-all duration-500 font-medium",
              requiresHumanInput
                ? "bg-[#1e1e23] border-white/[0.18] placeholder:text-gray-400 focus:border-white/[0.28] focus:bg-[#222228]"
                : "",
            ].join(" ")}
            disabled={isRunning && !requiresHumanInput}
          />

          {/* Planning Toggle & Model Selector - Inside Input */}
          <div className="absolute right-3 flex items-center gap-1">
            {/* Planning Mode Toggle with Tooltip */}
            <div className="relative group/plan">
              <button
                type="button"
                onClick={() => setEnablePlanning(!enablePlanning)}
                disabled={isRunning}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-[11px] font-bold transition-all uppercase tracking-wider ${
                  enablePlanning
                    ? "bg-amber-400/90 text-amber-950 border border-amber-400/50"
                    : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.05]"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <SlidersHorizontal className={`w-3.5 h-3.5 ${enablePlanning ? "text-amber-950" : ""}`} />
                <span className="hidden sm:inline">Plan</span>
              </button>
              
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-[#1c1c22] border border-white/10 rounded-xl opacity-0 invisible group-hover/plan:opacity-100 group-hover/plan:visible transition-all duration-200 pointer-events-none z-50 w-56">
                <div className="text-xs font-semibold text-white mb-1">
                  {enablePlanning ? "Planning Mode ON" : "Planning Mode"}
                </div>
                <div className="text-[10px] text-gray-400 leading-relaxed">
                  {enablePlanning 
                    ? "모르는 개념을 웹에서 검색하고, 단계별 계획을 세워 실행합니다."
                    : "켜면 복잡한 요청을 자동으로 분석하고 계획을 세웁니다."}
                </div>
                {/* Arrow */}
                <div className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-[#1c1c22] border-r border-b border-white/10 rotate-45 -mt-1" />
              </div>
            </div>

            {/* Model Selector */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelSelector(!showModelSelector)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] font-bold text-gray-500 hover:text-gray-300 hover:bg-white/[0.05] transition-all uppercase tracking-wider"
              >
                <span>{selectedModel === "gemini-3-preview" ? "Pro" : "Flash"}</span>
                <ChevronUp
                  className={`w-3 h-3 transition-transform duration-300 ${
                    showModelSelector ? "rotate-180" : ""
                  }`}
                />
              </button>

              <AnimatePresence>
                {showModelSelector && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.96 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.96 }}
                    transition={{ duration: 0.2 }}
                    className="absolute bottom-full right-0 mb-3 w-52 p-1.5 bg-[#252528] border border-white/[0.12] rounded-2xl shadow-2xl z-50 backdrop-blur-xl"
                  >
                    <div className="space-y-1">
                      <button
                        onClick={() => {
                          setSelectedModel("gemini-3-preview");
                          setShowModelSelector(false);
                        }}
                        type="button"
                        className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs flex items-center justify-between group transition-colors ${
                          selectedModel === "gemini-3-preview"
                            ? "bg-white/[0.05] text-white"
                            : "text-gray-500 hover:bg-white/[0.03] hover:text-gray-300"
                        }`}
                      >
                        <div className="flex flex-col gap-0.5">
                          <span className="font-bold tracking-tight text-[11px] uppercase">
                            Gemini 3 Pro
                          </span>
                          <span className="text-[10px] opacity-40 font-medium">
                            Deep Reasoning
                          </span>
                        </div>
                        {selectedModel === "gemini-3-preview" && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/60" />
                        )}
                      </button>

                      <button
                        onClick={() => {
                          setSelectedModel("gemini-flash");
                          setShowModelSelector(false);
                        }}
                        type="button"
                        className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs flex items-center justify-between group transition-colors ${
                          selectedModel === "gemini-flash"
                            ? "bg-white/[0.05] text-white"
                            : "text-gray-500 hover:bg-white/[0.03] hover:text-gray-300"
                        }`}
                      >
                        <div className="flex flex-col gap-0.5">
                          <span className="font-bold tracking-tight text-[11px] uppercase">
                            Gemini 3 Flash
                          </span>
                          <span className="text-[10px] opacity-40 font-medium">
                            Fast & Efficient
                          </span>
                        </div>
                        {selectedModel === "gemini-flash" && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/60" />
                        )}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <button
          type={isRunning && !requiresHumanInput ? "button" : "submit"}
          onClick={isRunning && !requiresHumanInput ? onStop : undefined}
          disabled={disableSubmit}
          className={`p-3.5 rounded-[18px] transition-all duration-300 shadow-xl ${
            isRunning && !requiresHumanInput
              ? enablePlanning
                ? "bg-amber-400 text-amber-950 hover:bg-red-500 hover:text-white cursor-pointer shadow-amber-400/20 hover:shadow-red-500/20"
                : "bg-emerald-500 text-white hover:bg-red-500 cursor-pointer shadow-emerald-500/10 hover:shadow-red-500/20"
              : enablePlanning
                ? "bg-amber-400 text-amber-950 hover:bg-amber-300 disabled:bg-amber-400/20 disabled:text-white/40 shadow-amber-400/20"
                : "bg-emerald-500 text-white hover:bg-emerald-400 disabled:bg-emerald-500/20 disabled:text-white/40 shadow-emerald-500/10"
          }`}
          title={isRunning && !requiresHumanInput ? "클릭하여 정지" : "전송"}
        >
          {isRunning && !requiresHumanInput ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <ArrowUp className="w-5 h-5" />
          )}
        </button>
      </form>

        <p className="mt-4 text-[10px] text-white/30 text-center font-medium tracking-tight">
          키오스크 에이전트는 실수를 할 수 있습니다. 중요한 작업은 확인하세요.
        </p>
      </div>
    </>
  );
}

