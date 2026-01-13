'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, 
  User, 
  AlertCircle, 
  Send, 
  Mic,
  Square,
  Loader2,
  CheckCircle,
  XCircle,
  ArrowRight
} from 'lucide-react';

interface ChatMessage {
  id: string;
  type: 'agent' | 'user' | 'system' | 'interrupt';
  content: string;
  timestamp: number;
  metadata?: {
    stage?: string;
    action?: string;
    iteration?: number;
    question?: string;
    reason?: string;
  };
}

interface ChatPanelProps {
  agentState: any;
  onUserResponse: (response: string) => void;
  isRunning: boolean;
}

export default function ChatPanel({ agentState, onUserResponse, isRunning }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isWaitingForUser, setIsWaitingForUser] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastProcessedRef = useRef<string>('');

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Process agent state changes into chat messages
  useEffect(() => {
    if (!agentState) return;

    const stateKey = `${agentState.stage}-${agentState.iteration}-${agentState.thought}-${agentState.status}`;
    if (stateKey === lastProcessedRef.current) return;
    lastProcessedRef.current = stateKey;

    // Add agent thought as message first
    if (agentState.thought && agentState.stage) {
      const agentMsg: ChatMessage = {
        id: `agent-${agentState.iteration}-${agentState.stage}-${Date.now()}`,
        type: 'agent',
        content: agentState.thought,
        timestamp: Date.now(),
        metadata: {
          stage: agentState.stage,
          action: agentState.action,
          iteration: agentState.iteration,
        },
      };

      setMessages(prev => {
        // Check if we already have this thought
        const exists = prev.some(m => 
          m.type === 'agent' && 
          m.content === agentState.thought &&
          m.metadata?.iteration === agentState.iteration
        );
        if (exists) return prev;
        return [...prev, agentMsg];
      });
    }

    // Check for human input requirement (HITL) - show question from interrupt
    if (agentState.status === 'waiting_human' || agentState.requires_human_input) {
      setIsWaitingForUser(true);
      
      // Get question from interrupt object
      const question = agentState.interrupt?.question || '계속 진행할까요?';
      const reason = agentState.interrupt?.reason || 'CONFIRMATION_REQUIRED';
      
      const interruptMsg: ChatMessage = {
        id: `interrupt-${Date.now()}`,
        type: 'interrupt',
        content: question, // Show the actual question from the model
        timestamp: Date.now(),
        metadata: {
          question: question,
          reason: reason,
          stage: agentState.stage,
        },
      };
      
      setMessages(prev => {
        // Prevent duplicate interrupt messages
        const hasRecentInterrupt = prev.some(m => 
          m.type === 'interrupt' && 
          Date.now() - m.timestamp < 1000
        );
        if (hasRecentInterrupt) return prev;
        return [...prev, interruptMsg];
      });
      return;
    }

    // Add system status messages
    if (agentState.status === 'done') {
      const systemMsg: ChatMessage = {
        id: `system-done-${Date.now()}`,
        type: 'system',
        content: '✅ 작업이 완료되었습니다.',
        timestamp: Date.now(),
      };
      setMessages(prev => {
        const hasDone = prev.some(m => m.type === 'system' && m.content.includes('완료'));
        if (hasDone) return prev;
        return [...prev, systemMsg];
      });
      setIsWaitingForUser(false);
    }
    
    if (agentState.status === 'aborted') {
      const systemMsg: ChatMessage = {
        id: `system-aborted-${Date.now()}`,
        type: 'system',
        content: '❌ 작업이 취소되었습니다.',
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, systemMsg]);
      setIsWaitingForUser(false);
    }

  }, [agentState]);

  // Handle user response submission
  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputValue.trim()) return;

    // Add user message
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputValue,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);

    // Send to agent
    onUserResponse(inputValue);
    setInputValue('');
    setIsWaitingForUser(false);
  };

  // STT Handlers
  const handleStartRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setIsTranscribing(true);
        
        try {
          const formData = new FormData();
          formData.append('file', blob, 'recording.webm');

          const response = await fetch('/api/stt/transcribe', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) throw new Error('Transcription failed');

          const data = await response.json();
          if (data.text) {
            setInputValue(prev => (prev ? `${prev} ${data.text}` : data.text));
          }
        } catch (error) {
          console.error('STT Error:', error);
        } finally {
          setIsTranscribing(false);
          stream.getTracks().forEach(track => track.stop());
        }
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (error) {
      console.error('Microphone Access Error:', error);
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      setMediaRecorder(null);
    }
  };

  // Quick responses for HITL
  const quickResponses = [
    { text: '네, 계속 진행해요', value: '네, 계속 진행해줘', icon: '✅' },
    { text: '취소할게요', value: '취소', icon: '❌' },
  ];
  
  // Suggested responses based on context
  const getSuggestedResponses = () => {
    const interrupt = messages.find(m => m.type === 'interrupt')?.metadata;
    if (!interrupt) return quickResponses;
    
    // Add context-specific suggestions
    if (interrupt.reason === 'AMBIGUOUS_CHOICE') {
      return [
        { text: '네, 와퍼로 주문해주세요', value: '네, 와퍼로 주문해주세요', icon: '🍔' },
        { text: '다른 메뉴 보여주세요', value: '다른 메뉴를 보여주세요', icon: '📋' },
        ...quickResponses,
      ];
    }
    
    return quickResponses;
  };

  return (
    <div className="flex flex-col h-full bg-[#1a1b1e] rounded-3xl border border-white/10 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/10 bg-[#1a1b1e]/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-white font-semibold">Kiosk Agent</h3>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              <span className="text-xs text-gray-400">
                {isRunning ? '작업 중...' : isWaitingForUser ? '응답 대기 중' : '대기 중'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-full text-center text-gray-500"
            >
              <Bot className="w-12 h-12 mb-4 opacity-30" />
              <p>에이전트와의 대화가 여기에 표시됩니다</p>
              <p className="text-sm mt-1">메뉴를 주문해보세요!</p>
            </motion.div>
          ) : (
            messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}
              >
                {/* Avatar */}
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center shrink-0
                  ${msg.type === 'agent' ? 'bg-blue-500/20 text-blue-400' : ''}
                  ${msg.type === 'user' ? 'bg-green-500/20 text-green-400' : ''}
                  ${msg.type === 'interrupt' ? 'bg-amber-500/20 text-amber-400' : ''}
                  ${msg.type === 'system' ? 'bg-gray-500/20 text-gray-400' : ''}
                `}>
                  {msg.type === 'agent' && <Bot className="w-4 h-4" />}
                  {msg.type === 'user' && <User className="w-4 h-4" />}
                  {msg.type === 'interrupt' && <AlertCircle className="w-4 h-4" />}
                  {msg.type === 'system' && <CheckCircle className="w-4 h-4" />}
                </div>

                {/* Message Content */}
                <div className={`
                  max-w-[80%] rounded-2xl px-4 py-3
                  ${msg.type === 'agent' ? 'bg-[#25262a] text-gray-200' : ''}
                  ${msg.type === 'user' ? 'bg-blue-600 text-white' : ''}
                  ${msg.type === 'interrupt' ? 'bg-amber-500/10 border border-amber-500/30 text-amber-200' : ''}
                  ${msg.type === 'system' ? 'bg-gray-500/10 text-gray-400 text-sm' : ''}
                `}>
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                  
                  {/* Metadata for agent messages */}
                  {msg.type === 'agent' && msg.metadata?.action && (
                    <div className="mt-2 pt-2 border-t border-white/10 flex items-center gap-2">
                      <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400">
                        {msg.metadata.action}
                      </span>
                      <span className="text-xs text-gray-500">
                        Step {msg.metadata.iteration}
                      </span>
                    </div>
                  )}

                  {/* HITL Question - Show suggested responses */}
                  {msg.type === 'interrupt' && isWaitingForUser && (
                    <div className="mt-3 pt-3 border-t border-amber-500/20">
                      <p className="text-xs text-amber-400/70 mb-2">빠른 응답:</p>
                      <div className="flex flex-wrap gap-2">
                        {getSuggestedResponses().map((resp) => (
                          <button
                            key={resp.value}
                            onClick={() => {
                              onUserResponse(resp.value);
                              setIsWaitingForUser(false);
                              const userMsg: ChatMessage = {
                                id: `user-${Date.now()}`,
                                type: 'user',
                                content: resp.text,
                                timestamp: Date.now(),
                              };
                              setMessages(prev => [...prev, userMsg]);
                            }}
                            className={`
                              px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1
                              ${resp.value === 'abort' 
                                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                                : 'bg-white/5 text-gray-300 hover:bg-white/10 border border-white/10'
                              }
                            `}
                          >
                            <span>{resp.icon}</span>
                            {resp.text}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-white/10 bg-[#15161a]">
        {/* HITL Indicator */}
        {isWaitingForUser && (
          <div className="mb-3 flex items-center gap-2 px-3 py-2 bg-amber-500/10 rounded-lg border border-amber-500/20">
            <AlertCircle className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-amber-300">에이전트가 응답을 기다리고 있습니다</span>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="flex items-center gap-3">
          {/* STT Button */}
          <button
            type="button"
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            disabled={isTranscribing || !isWaitingForUser}
            className={`
              p-3 rounded-full transition-all
              ${isRecording 
                ? 'bg-red-500/20 text-red-500 animate-pulse' 
                : isWaitingForUser
                  ? 'bg-[#25262a] text-gray-400 hover:text-white hover:bg-[#2d2e32]'
                  : 'bg-[#1a1b1e] text-gray-600 cursor-not-allowed'
              }
            `}
          >
            {isTranscribing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : isRecording ? (
              <Square className="w-5 h-5 fill-current" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </button>

          {/* Text Input */}
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={isWaitingForUser ? "답변을 입력하세요... (예: 네, 와퍼로 주문해주세요)" : "에이전트 응답 대기 중..."}
            className={`
              flex-1 rounded-xl px-4 py-3 text-sm transition-all
              ${isWaitingForUser 
                ? 'bg-[#25262a] text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50' 
                : 'bg-[#1a1b1e] text-gray-500 placeholder-gray-600 cursor-not-allowed'
              }
            `}
            disabled={!isWaitingForUser}
          />

          {/* Send Button */}
          <button
            type="submit"
            disabled={!inputValue.trim() || !isWaitingForUser}
            className={`
              p-3 rounded-full transition-all
              ${isWaitingForUser && inputValue.trim()
                ? 'bg-amber-500 text-white hover:bg-amber-400 active:scale-95'
                : 'bg-[#3c4043] text-gray-500 cursor-not-allowed'
              }
            `}
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}

