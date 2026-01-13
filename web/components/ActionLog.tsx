'use client';

import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, MousePointer2, Move, Type, ArrowLeft, Home, AlertCircle } from 'lucide-react';

interface ActionLogEntry {
  id: number;
  iteration: number;
  action: string;
  commands: string[];
  status: string;
}

interface ActionLogProps {
  logs: ActionLogEntry[];
  isRunning: boolean;
}

const actionIcons: Record<string, React.ElementType> = {
  CLICK: MousePointer2,
  LONG_CLICK: MousePointer2,
  SWIPE: Move,
  INPUT: Type,
  BACK: ArrowLeft,
  HOME: Home,
  INTERRUPT: AlertCircle,
};

const actionColors: Record<string, string> = {
  CLICK: 'text-blue-400 bg-blue-400/10',
  LONG_CLICK: 'text-blue-500 bg-blue-500/10',
  SWIPE: 'text-purple-400 bg-purple-400/10',
  INPUT: 'text-green-400 bg-green-400/10',
  BACK: 'text-orange-400 bg-orange-400/10',
  HOME: 'text-yellow-400 bg-yellow-400/10',
  INTERRUPT: 'text-red-400 bg-red-400/10',
};

export default function ActionLog({ logs, isRunning }: ActionLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length]);

  return (
    <div className="bento-card h-[600px] flex flex-col overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="px-8 py-6 border-b border-[#3c4043] bg-[#202124]/50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center">
            <Terminal className="w-5 h-5 text-[#8ab4f8]" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Action Log</h3>
            <p className="text-xs text-[#9aa0a6] font-medium uppercase tracking-wider">Device Command Stream</p>
          </div>
        </div>
        {isRunning && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20">
            <span className="w-2 h-2 rounded-full bg-[#8ab4f8] animate-pulse" />
            <span className="text-[10px] font-bold text-[#8ab4f8] uppercase">Live</span>
          </div>
        )}
      </div>

      {/* Log List - Auto-scrolls to latest */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4 scroll-smooth">
        <AnimatePresence mode="popLayout">
          {logs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-[#5f6368]">
              <div className="w-16 h-16 rounded-full bg-[#303134] flex items-center justify-center mb-4">
                <Terminal className="w-8 h-8" />
              </div>
              <p className="text-sm font-medium">대기 중...</p>
            </div>
          ) : (
            logs.map((log, index) => {
              const Icon = actionIcons[log.action] || Terminal;
              const colorClass = actionColors[log.action] || 'text-gray-400 bg-gray-400/10';

              return (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-2xl bg-[#202124] border border-[#3c4043] hover:border-[#5f6368] transition-all"
                >
                  <div className="flex items-start gap-4">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${colorClass}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-bold text-[#5f6368] bg-[#303134] px-1.5 py-0.5 rounded">STEP {log.iteration}</span>
                          <span className="text-sm font-bold text-white">{log.action}</span>
                        </div>
                        <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded ${
                          log.status === 'completed' ? 'text-[#81c995] bg-green-500/10' :
                          'text-[#fdd663] bg-yellow-500/10'
                        }`}>
                          {log.status}
                        </span>
                      </div>
                      {log.commands.map((cmd, idx) => (
                        <div key={idx} className="font-mono text-xs text-[#9aa0a6] bg-[#0f0f12] p-2 rounded-lg mt-1 flex gap-2">
                          <span className="text-[#8ab4f8]">$</span>
                          <span className="truncate">{cmd}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              );
            })
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

