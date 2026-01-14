'use client';

import { RotateCcw, ThumbsUp, ThumbsDown, Copy, CheckCircle2 } from 'lucide-react';
import { ResultAudioButton } from './ResultAudioButton';
import type { ChatMessage } from '../types';

interface Props {
  msg: ChatMessage;
  copiedMessageId: string | null;
  isRunning: boolean;
  onRetry: () => void;
  onFeedback: (feedback: 'up' | 'down' | null) => void;
  onCopy: () => void;
}

export function ResultMessage({ msg, copiedMessageId, isRunning, onRetry, onFeedback, onCopy }: Props) {
  return (
    <div className="flex items-start gap-5 my-10 pl-1">
      <div className="flex-1">
        <div className="relative">
          {/* Glow */}
          <div className="absolute -inset-1.5 rounded-[28px] bg-gradient-to-r from-emerald-500/18 via-emerald-500/8 to-emerald-500/10 blur-xl opacity-60" />

          {/* Card */}
          <div className="relative overflow-hidden rounded-[28px] border border-emerald-500/14 bg-[#0f0f12]/55 backdrop-blur-xl shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_20px_60px_rgba(0,0,0,0.55)]">
            {/* Subtle gradient wash */}
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-transparent to-transparent" />
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
            
            <div className="relative p-5">
              <div className="flex items-center gap-4">
                {/* Character Avatar + Nickname */}
                {msg.character?.imagePath && (
                  <div className="flex flex-col items-center flex-shrink-0 w-16">
                    <div className="w-14 h-14 mb-1.5">
                      <img 
                        src={msg.character.imagePath} 
                        alt={msg.character.nickname} 
                        className="w-full h-full object-cover rounded-full border-2 border-emerald-500/50 shadow-[0_0_16px_rgba(52,211,153,0.25)]" 
                      />
                    </div>
                    <span className="text-[10px] font-bold text-emerald-400 text-center">
                      {msg.character.nickname}
                    </span>
                  </div>
                )}

                {/* Content + TTS Button */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3">
                    <p className="flex-1 text-[15px] leading-relaxed font-semibold text-emerald-50/90 break-words">
                      {msg.content}
                    </p>
                    {/* TTS 버튼 */}
                    {msg.audioPath && (
                      <ResultAudioButton audioPath={msg.audioPath} showLabel />
                    )}
                  </div>
                </div>
              </div>
              
              {/* Action bar */}
              <div className="flex items-center gap-2 mt-4 pt-3 border-t border-white/5">
                <button
                  type="button"
                  onClick={onRetry}
                  disabled={isRunning}
                  className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all text-xs text-gray-400 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
                  title="재시도"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>재시도</span>
                </button>

                <button
                  type="button"
                  onClick={() => onFeedback(msg.feedback === 'up' ? null : 'up')}
                  className={[
                    "p-2 rounded-xl transition-colors",
                    msg.feedback === 'up'
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-white/5 hover:bg-white/10 text-gray-500 hover:text-gray-300",
                  ].join(" ")}
                  title="따봉"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>

                <button
                  type="button"
                  onClick={() => onFeedback(msg.feedback === 'down' ? null : 'down')}
                  className={[
                    "p-2 rounded-xl transition-colors",
                    msg.feedback === 'down'
                      ? "bg-rose-500/20 text-rose-300"
                      : "bg-white/5 hover:bg-white/10 text-gray-500 hover:text-gray-300",
                  ].join(" ")}
                  title="싫어요"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </button>

                <button
                  type="button"
                  onClick={onCopy}
                  className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-500 hover:text-gray-300 transition-colors"
                  title="복사"
                >
                  {copiedMessageId === msg.id ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-300" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
