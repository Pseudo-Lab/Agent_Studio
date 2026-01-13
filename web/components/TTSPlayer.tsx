"use client";

import { useEffect, useRef, useState } from "react";
import { Volume2, VolumeX, Pause, Play, Loader2 } from "lucide-react";

interface TTSPlayerProps {
  audioPath: string;
  autoPlay?: boolean;
  onPlaybackEnd?: () => void;
}

export function TTSPlayer({ audioPath, autoPlay = true, onPlaybackEnd }: TTSPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Extract filename from full path
  const filename = audioPath.split(/[\\/]/).pop() || audioPath;
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";
  const audioUrl = `${BACKEND_URL}/tts/audio/${encodeURIComponent(filename)}`;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
      setError(null);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handlePlay = () => {
      setIsPlaying(true);
    };

    const handlePause = () => {
      setIsPlaying(false);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      onPlaybackEnd?.();
    };

    const handleError = (e: Event) => {
      console.error("Audio playback error:", e);
      setError("오디오 재생에 실패했습니다.");
      setIsPlaying(false);
      setIsLoading(false);
    };

    const handleLoadStart = () => {
      setIsLoading(true);
    };

    const handleCanPlay = () => {
      setIsLoading(false);
    };

    audio.addEventListener("loadstart", handleLoadStart);
    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("error", handleError);

    // Auto-play if enabled
    if (autoPlay) {
      audio.play().catch((err) => {
        console.warn("Auto-play prevented:", err);
        setError("자동 재생이 차단되었습니다. 재생 버튼을 눌러주세요.");
        setIsLoading(false);
      });
    }

    return () => {
      audio.removeEventListener("loadstart", handleLoadStart);
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
    };
  }, [audioUrl, autoPlay, onPlaybackEnd]);

  // Handle volume changes
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.volume = volume;
    }
  }, [volume]);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch((err) => {
        console.error("Play failed:", err);
        setError("재생에 실패했습니다.");
      });
    }
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.muted = !audio.muted;
    setIsMuted(!isMuted);
  };

  const handleSeek = (value: number[]) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;

    const newTime = (value[0] / 100) * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (value: number[]) => {
    setVolume(value[0] / 100);
    if (value[0] === 0) {
      setIsMuted(true);
      if (audioRef.current) audioRef.current.muted = true;
    } else if (isMuted) {
      setIsMuted(false);
      if (audioRef.current) audioRef.current.muted = false;
    }
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-black/60 border border-emerald-500/40 backdrop-blur-sm shadow-[0_0_15px_rgba(16,185,129,0.2)]">
      {/* Hidden audio element */}
      <audio ref={audioRef} src={audioUrl} preload="auto" />
      
      {/* Play/Pause Button */}
      <button
        onClick={togglePlayPause}
        disabled={!!error || isLoading}
        className="h-8 w-8 flex items-center justify-center rounded-full bg-gradient-to-br from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black shadow-[0_0_10px_rgba(16,185,129,0.5)] transition-all duration-200 disabled:opacity-50"
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isPlaying ? (
          <Pause className="h-4 w-4" />
        ) : (
          <Play className="h-4 w-4 ml-0.5" />
        )}
      </button>

      {/* Status indicator */}
      <div className="flex items-center gap-1.5">
        <div className={`w-1.5 h-1.5 rounded-full ${isPlaying ? 'bg-emerald-400 animate-pulse' : 'bg-emerald-500/50'}`} />
        <span className="text-emerald-300/90 text-xs font-medium">질문</span>
      </div>

      {/* Mini progress bar */}
      <div className="relative w-20 h-1 bg-gray-700/50 rounded-full overflow-hidden">
        <div 
          className="absolute inset-y-0 left-0 bg-emerald-500 rounded-full transition-all duration-150"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      {/* Time */}
      <span className="text-[10px] font-mono text-gray-400 tabular-nums w-8">
        {formatTime(currentTime)}
      </span>

      {/* Volume toggle */}
      <button
        onClick={toggleMute}
        disabled={!!error || isLoading}
        className="h-6 w-6 flex items-center justify-center rounded-full hover:bg-emerald-500/20 transition-colors"
      >
        {isMuted || volume === 0 ? (
          <VolumeX className="h-3 w-3 text-gray-500" />
        ) : (
          <Volume2 className="h-3 w-3 text-emerald-400" />
        )}
      </button>
    </div>
  );
}
