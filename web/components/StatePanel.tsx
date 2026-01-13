'use client';

import { motion } from 'framer-motion';
import { Activity, Gauge, Route, Zap, CheckCircle2, XCircle, Clock, AlertTriangle } from 'lucide-react';

interface StateInfo {
  iteration: number;
  maxIterations: number;
  status: string;
  progress?: boolean;
  route: string;
  stage: string;
}

interface StatePanelProps {
  state: StateInfo;
  isRunning: boolean;
}

const statusConfig: Record<string, { icon: React.ElementType; color: string; label: string }> = {
  ready: { icon: Clock, color: 'text-gray-400', label: '대기 중' },
  running: { icon: Zap, color: 'text-[#8ab4f8]', label: '실행 중' },
  action_proposed: { icon: Activity, color: 'text-[#c6a7fb]', label: '액션 제안됨' },
  continue: { icon: CheckCircle2, color: 'text-[#81c995]', label: '진행 중' },
  no_progress: { icon: AlertTriangle, color: 'text-[#fdd663]', label: '진행 없음' },
  completed: { icon: CheckCircle2, color: 'text-[#81c995]', label: '완료' },
  success: { icon: CheckCircle2, color: 'text-[#81c995]', label: '성공' },
  waiting_human: { icon: Clock, color: 'text-[#ffbb00]', label: '사용자 입력 대기' },
  backtracked: { icon: Route, color: 'text-[#f28b82]', label: '백트래킹' },
  error: { icon: XCircle, color: 'text-[#f28b82]', label: '오류' },
};

const routeConfig: Record<string, { color: string; label: string }> = {
  idle: { color: 'bg-[#3c4043] text-[#9aa0a6]', label: 'Idle' },
  loop: { color: 'bg-blue-500/10 text-[#8ab4f8]', label: 'Loop' },
  analyze: { color: 'bg-purple-500/10 text-[#c6a7fb]', label: 'Analyze' },
  human: { color: 'bg-orange-500/10 text-[#ffbb00]', label: 'Human' },
  backtrack: { color: 'bg-red-500/10 text-[#f28b82]', label: 'Backtrack' },
  end: { color: 'bg-green-500/10 text-[#81c995]', label: 'End' },
};

export default function StatePanel({ state, isRunning }: StatePanelProps) {
  const statusInfo = statusConfig[state.status] || statusConfig.ready;
  const routeInfo = routeConfig[state.route] || routeConfig.idle;
  const StatusIcon = statusInfo.icon;
  const progressPercent = Math.min((state.iteration / state.maxIterations) * 100, 100);

  return (
    <div className="bento-card h-[600px] flex flex-col overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="px-8 py-6 border-b border-[#3c4043] bg-[#202124]/50 flex items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-green-500/10 flex items-center justify-center">
          <Gauge className="w-5 h-5 text-[#81c995]" />
        </div>
        <div>
          <h3 className="text-lg font-bold text-white">System Status</h3>
          <p className="text-xs text-[#9aa0a6] font-medium uppercase tracking-wider">Agent Orchestration</p>
        </div>
      </div>

      {/* State Content */}
      <div className="flex-1 p-8 space-y-10">
        {/* Status Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-black text-[#5f6368] uppercase tracking-widest">Current State</span>
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full bg-[#303134] ${statusInfo.color}`}>
              <StatusIcon className="w-3.5 h-3.5" />
              <span className="text-[11px] font-bold uppercase">{statusInfo.label}</span>
            </div>
          </div>
          
          <div className="relative h-24 rounded-3xl bg-[#202124] border border-[#3c4043] p-6 flex flex-col justify-center overflow-hidden">
            {isRunning && (
              <motion.div 
                className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-transparent"
                animate={{ x: ['-100%', '100%'] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              />
            )}
            <div className="flex items-center justify-between mb-3 relative z-10">
              <span className="text-xs font-bold text-[#9aa0a6]">PROGRESS</span>
              <span className="text-xl font-black text-white font-mono">{Math.round(progressPercent)}%</span>
            </div>
            <div className="h-2 w-full bg-[#303134] rounded-full overflow-hidden relative z-10">
              <motion.div
                className="h-full bg-[#8ab4f8]"
                initial={{ width: 0 }}
                animate={{ width: `${progressPercent}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-5 rounded-3xl bg-[#202124] border border-[#3c4043]">
            <span className="text-[10px] font-bold text-[#5f6368] uppercase block mb-2 tracking-tighter">Active Route</span>
            <span className={`text-[10px] font-black uppercase px-2 py-1 rounded ${routeInfo.color}`}>
              {routeInfo.label}
            </span>
          </div>
          <div className="p-5 rounded-3xl bg-[#202124] border border-[#3c4043]">
            <span className="text-[10px] font-bold text-[#5f6368] uppercase block mb-2 tracking-tighter">Steps</span>
            <span className="text-xl font-black text-white font-mono">
              {state.iteration}<span className="text-[#5f6368] text-sm ml-1">/ {state.maxIterations}</span>
            </span>
          </div>
        </div>

        {/* Pipeline Visualization */}
        <div className="space-y-4">
          <span className="text-[10px] font-black text-[#5f6368] uppercase tracking-widest">Active Pipeline</span>
          <div className="flex items-center justify-between gap-2">
            {['stt', 'vlm', 'execute', 'router'].map((s) => (
              <div key={s} className="flex flex-col items-center gap-2 flex-1">
                <div className={`w-full h-1.5 rounded-full transition-all duration-500 ${
                  state.stage.includes(s) ? 'bg-[#8ab4f8] shadow-[0_0_10px_#8ab4f8]' : 'bg-[#303134]'
                }`} />
                <span className={`text-[9px] font-bold uppercase ${
                  state.stage.includes(s) ? 'text-[#8ab4f8]' : 'text-[#5f6368]'
                }`}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Live Indicator */}
      <div className="px-8 py-4 bg-[#202124]/50 border-t border-[#3c4043] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-[#81c995] animate-pulse' : 'bg-[#5f6368]'}`} />
          <span className="text-[10px] font-bold text-[#9aa0a6] uppercase tracking-widest">
            {isRunning ? 'System Active' : 'System Idle'}
          </span>
        </div>
        <Zap className={`w-4 h-4 ${isRunning ? 'text-[#fdd663]' : 'text-[#5f6368]'}`} />
      </div>
    </div>
  );
}
