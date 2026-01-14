"use client";

import { useMemo, useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Check, Volume2 } from "lucide-react";

// Character info (TTS voice character)
interface CharacterInfo {
  id: string;
  name: string;
  nickname: string;
  imagePath: string;
}

type Props = {
  question: string;
  options?: string[];
  reason?: string;
  disabled?: boolean;
  onSelect: (option: string) => Promise<void> | void;
  audioPath?: string | null;  // TTS audio path for the question
  character?: CharacterInfo | null;  // Character info
};

export default function InterruptChoiceCard({ question, options, reason, disabled = false, onSelect, audioPath, character }: Props) {
  // TTS audio state
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";
  const audioUrl = audioPath ? `${BACKEND_URL}/tts/audio/${encodeURIComponent(audioPath.split(/[\\/]/).pop() || audioPath)}` : null;

  // Debug log
  useEffect(() => {
    console.log('[InterruptChoiceCard] audioPath:', audioPath, 'audioUrl:', audioUrl);
  }, [audioPath, audioUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    console.log('[InterruptChoiceCard] Setting up audio:', audioUrl);

    const handlePlay = () => { console.log('[Audio] Playing'); setIsPlaying(true); };
    const handlePause = () => { console.log('[Audio] Paused'); setIsPlaying(false); };
    const handleEnded = () => { console.log('[Audio] Ended'); setIsPlaying(false); };
    const handleError = (e: Event) => { console.error('[Audio] Error:', e); };

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("error", handleError);

    // Auto-play
    audio.play().catch((e) => console.error('[Audio] Autoplay failed:', e));

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
    };
  }, [audioUrl]);

  const toggleAudio = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) audio.pause();
    else audio.play().catch(() => { });
  };

  const normalizedOptions = useMemo(() => {
    const arr = Array.isArray(options) ? options : [];
    const seen = new Set<string>();
    const out: string[] = [];
    for (const raw of arr) {
      const v = String(raw || "").trim();
      if (!v) continue;
      if (seen.has(v)) continue;
      seen.add(v);
      out.push(v);
      if (out.length >= 10) break;
    }
    return out;
  }, [options]);

  const [selected, setSelected] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const canInteract = !disabled && !isSubmitting;
  const cols = normalizedOptions.length <= 4 ? 2 : 3;
  const optionVariants = {
    idle: { opacity: 1, scale: 1 },
    dim: { opacity: 0.55, scale: 0.985 },
    selected: { opacity: 1, scale: 1.02 },
  } as const;

  const handleClick = async (opt: string) => {
    if (!canInteract) return;
    setSelected(opt);
    setIsSubmitting(true);
    try {
      await onSelect(opt);
    } finally {
      // keep selected state; UI is usually disabled after respond, but this makes it robust
      setIsSubmitting(false);
    }
  };

  return (
    <div className="relative">
      {/* Glow */}
      <div className="absolute -inset-1.5 rounded-[28px] bg-gradient-to-r from-emerald-500/18 via-emerald-500/8 to-emerald-500/10 blur-xl opacity-60" />

      <div className="relative overflow-hidden rounded-[28px] border border-emerald-500/14 bg-[#0f0f12]/90 backdrop-blur-xl shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_20px_60px_rgba(0,0,0,0.55)]">
        {/* Subtle gradient wash */}
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-transparent to-transparent" />
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />

        <div className="relative p-5">
          {/* Hidden audio element */}
          {audioUrl && <audio ref={audioRef} src={audioUrl} preload="auto" />}

          {/* Layout: Character (left) + Content (right) */}
          <div className="flex items-center gap-4">
            {/* Character Avatar + Nickname - 캐릭터 정보가 있을 때만 표시 */}
            {character?.imagePath && (
              <div className="flex flex-col items-center flex-shrink-0 w-16">
                <div className="w-14 h-14 mb-1.5">
                  <img
                    src={character.imagePath}
                    alt={character.nickname}
                    className="w-full h-full object-cover rounded-full border-2 border-emerald-500/50 shadow-[0_0_16px_rgba(52,211,153,0.25)]"
                  />
                </div>
                <span className="text-[10px] font-bold text-emerald-400 text-center">
                  {character.nickname}
                </span>
              </div>
            )}

            {/* Question + TTS Button */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3">
                <p className="flex-1 text-[15px] leading-relaxed font-semibold text-emerald-50/90 break-words">
                  {question}
                </p>
                {/* TTS 버튼 */}
                {audioUrl && (
                  <button
                    type="button"
                    onClick={toggleAudio}
                    className={`flex-shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${isPlaying
                        ? 'bg-emerald-500/25 text-emerald-300 shadow-[0_0_12px_rgba(52,211,153,0.2)]'
                        : 'bg-white/5 text-gray-400 hover:text-emerald-300 hover:bg-emerald-500/10'
                      }`}
                    title={isPlaying ? "일시정지" : "음성으로 듣기"}
                  >
                    {isPlaying ? (
                      <>
                        <span className="flex items-center gap-0.5">
                          <span className="w-0.5 h-3 bg-emerald-400 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                          <span className="w-0.5 h-4 bg-emerald-400 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                          <span className="w-0.5 h-2 bg-emerald-400 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
                        </span>
                        <span>재생중</span>
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-3 h-3" />
                        <span>듣기</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Options */}
          {normalizedOptions.length > 0 && (
            <div className="mt-4">
              <div
                className={`grid gap-2.5 ${cols === 3 ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1 sm:grid-cols-2"
                  }`}
              >
                {normalizedOptions.map((opt, idx) => {
                  const isSelected = selected === opt;
                  const hasAnySelected = selected !== null;
                  const alpha = String.fromCharCode(65 + idx);
                  return (
                    <motion.button
                      key={opt}
                      type="button"
                      onClick={() => handleClick(opt)}
                      disabled={!canInteract}
                      whileHover={canInteract ? { y: -1 } : undefined}
                      whileTap={canInteract ? { scale: 0.985 } : undefined}
                      variants={optionVariants}
                      animate={isSelected ? "selected" : hasAnySelected ? "dim" : "idle"}
                      transition={{ type: "spring", stiffness: 520, damping: 34, mass: 0.65 }}
                      className={[
                        "group relative text-left rounded-2xl px-4 py-3 border transition-all",
                        "bg-white/[0.03] border-white/[0.07] hover:border-emerald-500/25 hover:bg-white/[0.05]",
                        "disabled:cursor-not-allowed",
                        isSelected
                          ? "border-emerald-400/45 bg-emerald-500/10 shadow-[0_0_0_1px_rgba(52,211,153,0.10),0_0_30px_rgba(52,211,153,0.10)]"
                          : "",
                      ].join(" ")}
                    >
                      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-r from-emerald-500/10 via-transparent to-transparent" />
                      {isSelected && (
                        <motion.div
                          className="absolute -inset-0.5 rounded-[18px] pointer-events-none"
                          initial={{ opacity: 0, scale: 0.98 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.18, ease: "easeOut" }}
                          style={{
                            background:
                              "radial-gradient(120px 80px at 30% 30%, rgba(52,211,153,0.18), transparent 60%), radial-gradient(140px 90px at 80% 60%, rgba(16,185,129,0.10), transparent 65%)",
                          }}
                        />
                      )}
                      <div className="relative flex items-center gap-3">
                        <span
                          className={[
                            "flex-shrink-0 w-6 h-6 rounded-lg border text-[11px] font-black flex items-center justify-center transition-colors",
                            isSelected
                              ? "bg-emerald-500/20 border-emerald-400/35 text-emerald-50/90"
                              : "bg-white/[0.04] border-white/[0.08] text-emerald-200/70",
                          ].join(" ")}
                        >
                          {alpha}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="text-[14px] font-semibold text-emerald-50/85 leading-snug break-words">
                            {opt}
                          </div>
                        </div>

                        <div className="flex-shrink-0">
                          {isSelected ? (
                            <motion.div
                              initial={{ scale: 0.85, opacity: 0 }}
                              animate={{ scale: 1, opacity: 1 }}
                              transition={{ type: "spring", stiffness: 700, damping: 30 }}
                            >
                              <Check className="w-4 h-4 text-emerald-200/85" />
                            </motion.div>
                          ) : null}
                        </div>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </div>
          )}

          {/* No options: keep UI minimal */}
        </div>
      </div>
    </div>
  );
}
