"use client";

import { motion } from "framer-motion";

type Props = {
  onClose: () => void;
};

export function AdbErrorCard({ onClose }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="flex justify-center my-8"
    >
      <div className="relative w-[420px]">
        {/* Animated glow effect */}
        <motion.div
          animate={{ opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 3, repeat: Infinity }}
          className="absolute -inset-1 bg-gradient-to-r from-emerald-500/15 to-teal-500/15 rounded-2xl blur-lg"
        />

        <div className="relative bg-gradient-to-b from-[#1a1a1f] to-[#131316] border border-white/[0.06] rounded-2xl px-6 py-5 shadow-2xl">
          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            className="absolute top-3 right-3 p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-white/5 transition-colors"
          >
            <svg
              className="w-4 h-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>

          {/* Header row: Robot + Text side by side */}
          <div className="flex items-center gap-5 mb-4 pr-6">
            {/* Android Robot Icon - Compact */}
            <motion.div
              animate={{ y: [0, -3, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
              className="relative flex-shrink-0"
            >
              <div className="absolute inset-0 bg-emerald-400/15 rounded-full blur-lg scale-125" />
              <svg className="w-14 h-14 relative" viewBox="0 0 96 96" fill="none">
                <line
                  x1="32"
                  y1="28"
                  x2="26"
                  y2="18"
                  stroke="#3DDC84"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                />
                <line
                  x1="64"
                  y1="28"
                  x2="70"
                  y2="18"
                  stroke="#3DDC84"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                />
                <path
                  d="M22 44C22 35 30 28 48 28C66 28 74 35 74 44V48C74 51 71 54 68 54H28C25 54 22 51 22 48V44Z"
                  fill="#3DDC84"
                />
                <circle cx="36" cy="42" r="3.5" fill="#1a1a1f" />
                <circle cx="60" cy="42" r="3.5" fill="#1a1a1f" />
                <rect x="26" y="56" width="44" height="24" rx="3" fill="#3DDC84" />
                <rect x="14" y="56" width="7" height="16" rx="3.5" fill="#3DDC84" />
                <rect x="75" y="56" width="7" height="16" rx="3.5" fill="#3DDC84" />
              </svg>
              {/* X indicator */}
              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-gray-800 rounded-full flex items-center justify-center border border-gray-700">
                <svg
                  className="w-3 h-3 text-red-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                >
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </div>
            </motion.div>

            {/* Title & Hint */}
            <div className="flex-1 min-w-0">
              <h3 className="text-base font-bold text-white leading-tight">
                디바이스 연결 필요
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                USB 디버깅이 활성화된 Android 기기를 연결하세요
              </p>
            </div>
          </div>

          {/* Steps - Compact horizontal layout */}
          <div className="flex gap-2 mb-3">
            {[
              { icon: "🔌", text: "USB 연결" },
              { icon: "⚙️", text: "디버깅 ON" },
              { icon: "✓", text: "허용 승인" },
            ].map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.08 }}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg bg-white/[0.03] border border-white/[0.05]"
              >
                <span className="text-sm">{step.icon}</span>
                <span className="text-[11px] text-gray-400">{step.text}</span>
              </motion.div>
            ))}
          </div>

          {/* Terminal hint - Inline */}
          <div className="flex items-center justify-between p-2 bg-black/40 rounded-lg border border-white/[0.04]">
            <div className="flex items-center gap-2">
              <div className="flex gap-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500/50" />
                <div className="w-1.5 h-1.5 rounded-full bg-yellow-500/50" />
                <div className="w-1.5 h-1.5 rounded-full bg-green-500/50" />
              </div>
              <code className="text-[11px] text-emerald-400 font-mono">
                $ adb devices
              </code>
            </div>
            <span className="text-[10px] text-gray-500">연결 확인</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
