"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Volume2 } from "lucide-react";

export function ResultAudioButton({
  audioPath,
  showLabel = false,
}: {
  audioPath: string;
  showLabel?: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";
  const filename = audioPath.split(/[\\/]/).pop() || audioPath;
  const audioUrl = `${BACKEND_URL}/tts/audio/${encodeURIComponent(filename)}`;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);

    // Auto-play on mount
    audio.play().catch(() => {});

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [audioUrl]);

  const toggleAudio = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) audio.pause();
    else audio.play().catch(() => {});
  };

  return (
    <>
      <audio ref={audioRef} src={audioUrl} preload="auto" />
      <button
        type="button"
        onClick={toggleAudio}
        className={`flex-shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
          isPlaying
            ? "bg-emerald-500/25 text-emerald-300 shadow-[0_0_12px_rgba(52,211,153,0.2)]"
            : "bg-white/5 text-gray-400 hover:text-emerald-300 hover:bg-emerald-500/10"
        }`}
        aria-label={isPlaying ? "일시정지" : "음성으로 듣기"}
        title={isPlaying ? "일시정지" : "음성으로 듣기"}
      >
        {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
        {showLabel && <span>{isPlaying ? "재생중" : "듣기"}</span>}
      </button>
    </>
  );
}

