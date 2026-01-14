'use client';

import { motion } from 'framer-motion';

interface QuickAction {
  text: string;
  emoji: string;
}

interface Props {
  quickActions: QuickAction[];
  greeting: { emoji: string; text: string };
  onQuickAction: (text: string) => void;
}

export function WelcomeScreen({ quickActions, greeting, onQuickAction }: Props) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-4 animate-in fade-in duration-700">
      <div className="w-20 h-20 mb-6">
        <img src="/images/gemini-logo.png" alt="Gemini" className="w-full h-full object-contain" />
      </div>
      <h2 className="text-3xl font-bold text-white mb-3 tracking-tight">키오스크 에이전트</h2>
      <p className="text-gray-400 mb-10">에이전트가 키오스크를 제어하여 주문을 도와드립니다.</p>
      <div className="flex flex-wrap gap-2.5 justify-center">
        {quickActions.map((a) => (
          <button
            key={a.text}
            onClick={() => onQuickAction(a.text)}
            className="px-5 py-2.5 rounded-2xl bg-white/[0.05] hover:bg-white/[0.12] border border-white/[0.15] text-sm text-gray-300 hover:text-white transition-all duration-300"
          >
            {a.emoji} {a.text}
          </button>
        ))}
      </div>
      {/* Time-based Greeting */}
      <p className="mt-12 text-[13px] text-gray-500">
        {greeting.emoji} {greeting.text}. 무엇을 주문해 드릴까요?
      </p>
    </div>
  );
}
