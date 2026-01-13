'use client';

import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Lightbulb, Sparkles } from 'lucide-react';

interface ThoughtEntry {
  id: number;
  iteration: number;
  thought: string;
  stage: string;
}

interface ThoughtPanelProps {
  thoughts: ThoughtEntry[];
  isRunning: boolean;
}

const stageLabels: Record<string, { label: string; color: string }> = {
  vlm: { label: 'VLM 추론', color: 'text-purple-400 bg-purple-400/10' },
  execute: { label: '실행', color: 'text-blue-400 bg-blue-400/10' },
  state_router: { label: '상태 판단', color: 'text-green-400 bg-green-400/10' },
  analyze: { label: '분석', color: 'text-orange-400 bg-orange-400/10' },
  backtrack: { label: '백트래킹', color: 'text-red-400 bg-red-400/10' },
  loop: { label: 'LOOP', color: 'text-cyan-400 bg-cyan-400/10' },
};

export default function ThoughtPanel({ thoughts, isRunning }: ThoughtPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest thought
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thoughts.length]);

  return (
    <div className="bento-card h-[600px] flex flex-col overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="px-8 py-6 border-b border-[#3c4043] bg-[#202124]/50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
            <Brain className="w-5 h-5 text-[#c6a7fb]" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Reasoning</h3>
            <p className="text-xs text-[#9aa0a6] font-medium uppercase tracking-wider">Agent Thought Process</p>
          </div>
        </div>
      </div>

      {/* Thoughts List - Auto-scrolls to latest */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-8 scroll-smooth">
        <AnimatePresence mode="popLayout">
          {thoughts.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-[#5f6368]">
              <div className="w-16 h-16 rounded-full bg-[#303134] flex items-center justify-center mb-4">
                <Brain className="w-8 h-8" />
              </div>
              <p className="text-sm font-medium">대기 중...</p>
            </div>
          ) : (
            thoughts.map((entry, index) => {
              const stageInfo = stageLabels[entry.stage] || { 
                label: entry.stage, 
                color: 'text-gray-400 bg-gray-400/10' 
              };

              return (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="relative pl-8 border-l-2 border-[#3c4043]"
                >
                  <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-[#3c4043] border-4 border-[#1a1b1e]" />
                  
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-[10px] font-bold text-[#c6a7fb] bg-purple-500/10 px-2 py-0.5 rounded-full uppercase tracking-tighter">
                      {stageInfo.label}
                    </span>
                    <span className="text-[10px] font-bold text-[#5f6368]">ITERATION {entry.iteration}</span>
                  </div>

                  <div className="p-5 rounded-2xl bg-[#202124] border border-[#3c4043] shadow-sm">
                    <div className="flex items-start gap-3">
                      <Lightbulb className="w-5 h-5 text-[#fdd663] flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-[#e8eaed] leading-relaxed font-medium">
                        {entry.thought}
                      </p>
                    </div>
                  </div>
                </motion.div>
              );
            })
          )}
        </AnimatePresence>

        {isRunning && (
          <div className="flex items-center gap-4 pl-8">
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-[#8ab4f8]"
                  animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                />
              ))}
            </div>
            <span className="text-xs font-bold text-[#8ab4f8] uppercase tracking-widest">Processing</span>
          </div>
        )}
      </div>
    </div>
  );
}

