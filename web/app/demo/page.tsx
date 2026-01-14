'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Loader2,
  Mic,
  Square,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Terminal,
  CheckCircle2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  ArrowUp,
} from 'lucide-react';

import Header from '@/components/Header';
import GridScan from '@/components/GridScan';
// TTSPlayer is now integrated into InterruptChoiceCard
import InterruptChoiceCard from '@/components/InterruptChoiceCard';
import TargetChip from '@/components/TargetChip';
import ShinyText from '@/components/ShinyText';

import type { AGUIEvent, ChatMessage, CharacterInfo, ThinkingStep, PlanningState } from './types';
import { ResultAudioButton } from './components/ResultAudioButton';
import { ChatInputBar } from './components/ChatInputBar';
import { PlanningPanel } from './components/PlanningPanel';

const actionColors: Record<string, string> = {
  CLICK: 'text-blue-400',
  SWIPE: 'text-purple-400',
  INPUT: 'text-green-400',
  INTERRUPT: 'text-yellow-400',
};

const isValidBox2d = (box?: number[]): box is [number, number, number, number] => {
  if (!Array.isArray(box) || box.length !== 4) return false;
  // [0,0,0,0] (INTERRUPT 규칙) 등은 "좌표 없음"으로 간주
  const hasNonZero = box.some(v => typeof v === 'number' && v !== 0);
  if (!hasNonZero) return false;
  // 값 범위 기본 검증(0~1000). 범위 밖이면 표시하지 않음
  return box.every(v => typeof v === 'number' && v >= 0 && v <= 1000);
};

export default function Home() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set());
  const [currentStep, setCurrentStep] = useState(0);
  const [requiresHumanInput, setRequiresHumanInput] = useState(false);
  const [activeInterruptId, setActiveInterruptId] = useState<string | null>(null);
  const [currentCharacter, setCurrentCharacter] = useState<CharacterInfo | null>(null);  // 현재 세션의 캐릭터

  // TTS audio is now stored per-message in ChatMessage.audioPath

  // Current thinking message ID for streaming updates
  const currentThinkingIdRef = useRef<string | null>(null);

  // Track ALL seen interrupt questions to avoid duplicates
  const seenInterruptsRef = useRef<Set<string>>(new Set());

  // STT
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("gemini-flash");
  const [showModelSelector, setShowModelSelector] = useState(false);
  
  // Planning Mode
  const [enablePlanning, setEnablePlanning] = useState(false);
  const [planningState, setPlanningState] = useState<PlanningState | null>(null);
  const [showPlanningPanel, setShowPlanningPanel] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const latestTtsAudioPathRef = useRef<string | null>(null);  // Track latest TTS audio for result message

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, expandedThinking]); // expandedThinking 변경 시에도 스크롤

  // Toggle thinking expansion
  const toggleThinking = (id: string) => {
    setExpandedThinking(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      return newSet;
    });
  };

  // Add message
  const addMessage = useCallback((msg: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    const newMsg = { ...msg, id: crypto.randomUUID(), timestamp: new Date() };
    setMessages(prev => [...prev, newMsg]);
    return newMsg.id;
  }, []);

  const setMessageFeedback = useCallback((id: string, feedback: 'up' | 'down' | null) => {
    setMessages(prev => prev.map(m => (m.id === id ? { ...m, feedback } : m)));
  }, []);

  const copyMessage = useCallback(async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(id);
      window.setTimeout(() => {
        setCopiedMessageId(curr => (curr === id ? null : curr));
      }, 1200);
    } catch (e) {
      console.warn('[Copy] Failed', e);
    }
  }, []);

  // Update thinking message with new step
  const addThinkingStep = useCallback((step: ThinkingStep) => {
    const thinkingId = currentThinkingIdRef.current;
    if (!thinkingId) return;
    
    setMessages(prev => prev.map(m => {
      if (m.id === thinkingId && m.type === 'thinking') {
        const steps = m.steps || [];
        return { ...m, steps: [...steps, step], isStreaming: true };
      }
      return m;
    }));
  }, []);

  // Finalize thinking
  const finalizeThinking = useCallback(() => {
    const thinkingId = currentThinkingIdRef.current;
    if (!thinkingId) return;
    
    setMessages(prev => prev.map(m => 
      m.id === thinkingId ? { ...m, isStreaming: false } : m
    ));
  }, []);

  // STT
  const handleStartRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setIsTranscribing(true);
        try {
          const formData = new FormData();
          formData.append('file', blob, 'recording.webm');
          const res = await fetch(`${BACKEND_URL}/stt/transcribe`, { method: 'POST', body: formData });
          if (res.ok) {
            const data = await res.json();
            if (data.text) setInputValue(prev => prev ? `${prev} ${data.text}` : data.text);
          } else {
            console.error('[STT] Failed:', res.status, await res.text());
          }
        } catch (e) { console.error('[STT] Error:', e); }
        finally { setIsTranscribing(false); stream.getTracks().forEach(t => t.stop()); }
      };
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (e) { console.error('[STT] Recording error:', e); }
  };

  const handleStopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      setMediaRecorder(null);
    }
  };

  // Process AG-UI Event
  const processAGUIEvent = useCallback((event: AGUIEvent) => {
    console.log('[AG-UI]', event);

    if (event.type === 'RUN_STARTED') {
      // Prefer AG-UI canonical casing, fallback to legacy snake_case
      const startedThreadId = (event as any).threadId ?? (event as any).thread_id;
      if (startedThreadId && !threadId) {
        setThreadId(startedThreadId);
      }
      // 셰프 정보 설정 (세션 시작 시)
      // Backend sends character info as 'chef' key for compatibility
      const startCharacter: CharacterInfo | undefined = (event as any).chef;
      if (startCharacter) {
        console.log('[RUN_STARTED] Character assigned:', startCharacter.nickname);
        setCurrentCharacter(startCharacter);
      }
      // 새 Thinking 블록 생성
      const id = addMessage({
        type: 'thinking',
        content: '추론 중...',
        isStreaming: true,
        steps: [],
      });
      currentThinkingIdRef.current = id;
      // Auto-expand new thinking block
      setExpandedThinking(prev => {
        const newSet = new Set(Array.from(prev));
        newSet.add(id);
        return newSet;
      });
    }

    if (event.type === 'RUN_FINISHED') {
      finalizeThinking();
      setIsRunning(false);
      setActiveInterruptId(null);
      
      // Add result message based on status
      const result = (event as any).result;
      const status =
        result?.status ??
        (event as any).status;
      if (status !== 'waiting_human') {
        const finalAction =
          result?.finalAction ??
          (event as any).final_action ??
          (event as any).finalAction;
        const finalThought =
          result?.finalThought ??
          (event as any).final_thought ??
          (event as any).finalThought;
        let resultMsg =
          typeof finalThought === 'string' && finalThought.trim()
            ? finalThought.trim()
            : '작업이 완료되었습니다.';
        if (status === 'cancelled') {
          resultMsg =
            typeof finalThought === 'string' && finalThought.trim()
              ? finalThought.trim()
              : '작업이 취소되었습니다.';
        } else if (!finalThought && finalAction) {
          resultMsg = `${finalAction} 작업을 수행했습니다.`;
        }
        // Attach TTS audio path and character info if available
        const ttsPath = latestTtsAudioPathRef.current;
        const characterInfo: CharacterInfo | undefined = (event as any).chef ?? result?.chef;
        // 캐릭터 정보 저장
        if (characterInfo) {
          setCurrentCharacter(characterInfo);
        }
        addMessage({ type: 'result', content: resultMsg, audioPath: ttsPath, character: characterInfo || currentCharacter });
        latestTtsAudioPathRef.current = null;  // Clear after use
      }
    }

    if (event.type === 'RUN_ERROR') {
      finalizeThinking();
      setIsRunning(false);
      setActiveInterruptId(null);
      addMessage({ type: 'system', content: `❌ ${event.message}` });
    }

    if (event.type === 'STATE_SNAPSHOT' && event.snapshot) {
      const s = event.snapshot;
      
      if (s.thought) {
        // 백엔드에서 받은 iteration 사용
        setCurrentStep(s.iteration || 0);
        
        addThinkingStep({
          id: crypto.randomUUID(),
          thought: s.thought,
          action: s.action,
          iteration: s.iteration,  // 백엔드 iteration 사용
          status: s.status,
          adb_commands: s.adb_commands,
          box_2d: s.box_2d,  // 바운딩박스 좌표
          progress: typeof s.progress === 'boolean' ? s.progress : undefined,
          difference: typeof s.difference === 'number' ? s.difference : undefined,
        });
      }

      // Interrupt 질문 - 전체 세션에서 중복 방지
      if (s.interrupt?.question) {
        const rawQuestion = String(s.interrupt.question ?? '');
        const question = rawQuestion.trim().replace(/\s+/g, ' ');
        const reason = typeof s.interrupt?.reason === 'string' ? s.interrupt.reason : undefined;
        const rawOptions =
          s.interrupt?.options ??
          s.interrupt?.question_list ??
          s.interrupt?.questionList;
        const options = Array.isArray(rawOptions)
          ? rawOptions
              .filter((v: any) => typeof v === 'string' && v.trim())
              .map((v: string) => v.trim())
              .slice(0, 10)
          : undefined;
        // 이미 본 질문이면 표시 안함 (Set으로 추적)
        if (!seenInterruptsRef.current.has(question)) {
          seenInterruptsRef.current.add(question);
          finalizeThinking();
          // 저장된 TTS 오디오가 있으면 함께 추가
          const savedAudioPath = latestTtsAudioPathRef.current;
          const interruptMsgId = addMessage({
            type: 'interrupt',
            content: question,
            options,
            reason,
            character: currentCharacter,  // 현재 세션의 캐릭터 정보 추가
            audioPath: savedAudioPath || undefined,  // 저장된 TTS 오디오
          });
          setActiveInterruptId(interruptMsgId);
          console.log('[Interrupt] Created with character:', currentCharacter?.nickname, 'audioPath:', savedAudioPath);
        }
        setRequiresHumanInput(true);
      }

      if (s.status === 'waiting_human') {
        setRequiresHumanInput(true);
      }
    }

    if (event.type === 'CUSTOM' && event.name === 'waiting_human') {
      setRequiresHumanInput(true);
      // waiting_human 이벤트에서 캐릭터 정보 추출 및 저장 (backend sends as 'chef')
      const hitlCharacter: CharacterInfo | undefined = (event as any).value?.chef;
      if (hitlCharacter) {
        console.log('[HITL] Character info received:', hitlCharacter);
        setCurrentCharacter(hitlCharacter);
        // 최근 interrupt 메시지에 캐릭터 정보 추가 (항상 업데이트)
        setMessages(prev => {
          const interruptMsgs = prev.filter(m => m.type === 'interrupt');
          const latestInterrupt = interruptMsgs[interruptMsgs.length - 1];
          if (latestInterrupt) {
            console.log('[HITL] Attaching character to interrupt:', latestInterrupt.id, hitlCharacter.nickname);
            return prev.map(m => 
              m.id === latestInterrupt.id ? { ...m, character: hitlCharacter } : m
            );
          }
          return prev;
        });
      }
    }

    // TTS audio generated - attach to current interrupt message or save for result
    if (event.type === 'CUSTOM' && event.name === 'tts_generated') {
      const audioPath =
        (event as any).value?.audioPath ??
        (event as any).value?.audio_path;
      console.log('[TTS Event] audioPath:', audioPath);
      if (audioPath) {
        // Save for potential use in result message
        latestTtsAudioPathRef.current = audioPath;
        
        // Always try to update the latest interrupt message (항상 업데이트)
        setMessages(prev => {
          // Find the most recent interrupt message
          const interruptMsgs = prev.filter(m => m.type === 'interrupt');
          const latestInterrupt = interruptMsgs[interruptMsgs.length - 1];
          if (latestInterrupt) {
            console.log('[TTS] Attaching audio to interrupt:', latestInterrupt.id, audioPath);
            return prev.map(m => 
              m.id === latestInterrupt.id ? { ...m, audioPath } : m
            );
          }
          console.log('[TTS] No interrupt message found yet, saved for later');
          return prev;
        });
      }
    }

    // Planning Mode update
    if (event.type === 'CUSTOM' && event.name === 'planning_update') {
      const value = (event as any).value;
      if (value) {
        console.log('[Planning] Update received:', value.status);
        setPlanningState({
          phase: "planning",
          status: value.status || "",
          unknownEntities: value.unknown_entities || [],
          searchContext: value.search_context || "",
          plan: value.plan || [],
        });
        setShowPlanningPanel(true);
        
        // Keep planning panel visible (user can close manually)
      }
    }
  }, [addMessage, addThinkingStep, finalizeThinking, activeInterruptId]);

  // Quick submit for HITL multiple-choice options
  const submitInterruptOption = async (choice: string) => {
    const text = String(choice || '').trim();
    if (!text) return;
    if (!threadId) return;
    if (!requiresHumanInput) return;

    addMessage({ type: 'user', content: text });
    setRequiresHumanInput(false);
    setActiveInterruptId(null);
    await startStream('/api/agent/respond', { thread_id: threadId, response: text });
  };

  // Submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('[handleSubmit] inputValue:', inputValue, 'isRunning:', isRunning, 'requiresHumanInput:', requiresHumanInput, 'threadId:', threadId);
    if (!inputValue.trim() || (isRunning && !requiresHumanInput)) {
      console.log('[handleSubmit] Early return - blocked');
      return;
    }

    const userInput = inputValue.trim();
    setInputValue('');
    addMessage({ type: 'user', content: userInput });

    if (requiresHumanInput && threadId) {
      console.log('[handleSubmit] HITL response:', userInput, 'to thread:', threadId);
      setRequiresHumanInput(false);
      setActiveInterruptId(null);
      // HITL 응답 - 새 thinking 블록 생성됨
      await startStream('/api/agent/respond', { thread_id: threadId, response: userInput });
      return;
    }

    // 새 작업 시작 - interrupt 기록 초기화, 셰프 초기화, Planning 초기화
    seenInterruptsRef.current.clear();
    setCurrentCharacter(null);  // 새 세션에서 새 캐릭터 할당을 위해 초기화
    setPlanningState(null);  // Planning 상태 초기화
    setShowPlanningPanel(enablePlanning);  // Planning 모드면 패널 표시
    setIsRunning(true);
    setCurrentStep(0);
    const newThreadId = crypto.randomUUID();
    setThreadId(newThreadId);
    await startStream('/api/agent/start', { 
      instruction: userInput, 
      thread_id: newThreadId,
      model: selectedModel,
      enable_planning: enablePlanning,
    });
  };

  // Stream - 백엔드 직접 호출 (Next.js 프록시 버퍼링 우회)
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';

  const handleStop = useCallback(
    async (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!threadId) return;
      try {
        // Abort the current stream
        abortControllerRef.current?.abort();

        // Call interrupt endpoint
        const res = await fetch(`${BACKEND_URL}/agent/interrupt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            thread_id: threadId,
            response: '무엇을 도와드릴까요?',
          }),
        });

        if (res.ok) {
          setIsRunning(false);
          setRequiresHumanInput(true);
          const interruptId = addMessage({
            type: 'interrupt',
            content: '작업이 중단되었습니다. 무엇을 도와드릴까요?',
            options: [],
            reason: 'USER_STOP',
          });
          setActiveInterruptId(interruptId);
          finalizeThinking();
        }
      } catch (err) {
        console.error('[Interrupt] Failed', err);
      }
    },
    [BACKEND_URL, addMessage, finalizeThinking, threadId]
  );
  
  const startStream = async (url: string, body: object) => {
    console.log('[startStream] Called with url:', url, 'body:', body);
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    setIsRunning(true);

    // Next.js 프록시 대신 백엔드 직접 호출
    const directUrl = url.replace('/api/agent/', `${BACKEND_URL}/agent/`);
    console.log('[Stream] Connecting to:', directUrl, 'body:', JSON.stringify(body));

    try {
      const res = await fetch(directUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortControllerRef.current.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // SSE는 \n\n으로 구분됨
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        
        for (const eventStr of events) {
          const trimmed = eventStr.trim();
          if (trimmed.startsWith('data:')) {
            try {
              const jsonStr = trimmed.slice(5).trim();
              const data = JSON.parse(jsonStr);
              console.log('[SSE Event]', data.type, data);
              processAGUIEvent(data);
            } catch (e) {
              console.warn('[SSE Parse Error]', trimmed, e);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('[Stream Error]', error);
        addMessage({ type: 'system', content: `❌ ${error.message}` });
      }
    } finally {
      setIsRunning(false);
    }
  };

  const quickActions = [
    { text: '빅맥셋트 주문해줘', emoji: '🍔' },
    { text: '행운버거 주문해줘', emoji: '🍀' },
    { text: '빅맥이랑 상하이스파이시버거 두개 주문해줘', emoji: '🍔' },
  ];

  // Time-based greeting
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 9) return { emoji: '🌅', text: '좋은 새벽이에요' };
    if (hour >= 9 && hour < 12) return { emoji: '☀️', text: '좋은 아침이에요' };
    if (hour >= 12 && hour < 17) return { emoji: '🌤️', text: '좋은 오후에요' };
    if (hour >= 17 && hour < 21) return { emoji: '🌆', text: '좋은 저녁이에요' };
    return { emoji: '🌙', text: '좋은 밤이에요' };
  };
  const greeting = getGreeting();

  return (
    <main className="flex min-h-screen flex-col bg-[#111114] text-gray-100 font-sans relative">
      {/* Background - GridScan */}
      <div className="absolute inset-0 z-0 opacity-70 pointer-events-none">
        <GridScan
          sensitivity={0.55}
          lineThickness={1.5}
          linesColor="#4a3d6a"
          gridScale={0.12}
          scanColor="#4ade80"
          scanOpacity={0.5}
          bloomIntensity={0.8}
          noiseIntensity={0.02}
          scanDuration={2.0}
          scanDelay={1.5}
        />
      </div>
      
      <div className="relative z-10 flex flex-col flex-1">
        <Header />
      
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full pt-2">
        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-6 py-10 space-y-10 scrollbar-hide">
          {/* Planning Mode Panel (sticky top) */}
          {enablePlanning && showPlanningPanel && (
            <div className="sticky top-0 z-30 -mx-6 px-6 pt-2 pb-3 bg-gradient-to-b from-[#111114] via-[#111114]/95 to-transparent backdrop-blur-sm">
              <PlanningPanel
                planningState={planningState}
                isVisible={true}
                onClose={() => setShowPlanningPanel(false)}
              />
            </div>
          )}

          {messages.length === 0 && (
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
                    onClick={() => setInputValue(a.text)}
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
          )}

          {/* Message List */}
          <AnimatePresence mode="popLayout">
            {messages.map((msg) => {
              const isThinkingExpanded = expandedThinking.has(msg.id);

              return (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
                  className="w-full"
                >
                  {/* User Message - Claude Style */}
                  {msg.type === 'user' && (
                    <div className="flex justify-end mb-4">
                      <div className="flex flex-col items-end max-w-[85%]">
                        <div className="px-5 py-3.5 rounded-[24px] rounded-br-md bg-[#2f2f32] text-white shadow-sm border border-white/5">
                          <p className="text-[15px] leading-relaxed font-normal">{msg.content}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* System Message */}
                  {msg.type === 'system' && (
                    <div className="flex justify-center my-8">
                      <span className="text-[11px] text-gray-400 bg-white/[0.05] border border-white/[0.08] px-4 py-1.5 rounded-full tracking-tight">
                        {msg.content}
                      </span>
                    </div>
                  )}

                  {/* Thinking Block - Sophisticated Minimalist (hide if 0 steps and not streaming) */}
                  {msg.type === 'thinking' && (msg.isStreaming || (msg.steps && msg.steps.length > 0)) && (
                    <div className="mb-10 pl-1 opacity-50 hover:opacity-80 transition-opacity duration-300">
                      {/* Thinking Header */}
                      <button
                        onClick={() => toggleThinking(msg.id)}
                        className="flex items-center gap-3 text-gray-400 hover:text-gray-200 transition-all group w-full text-left"
                      >
                        <div className="flex items-center justify-center w-5 h-5 rounded-full bg-white/[0.03] border border-white/[0.05] group-hover:bg-white/[0.08] transition-colors">
                          <ChevronDown className={`w-3 h-3 transition-transform duration-300 ${isThinkingExpanded ? '' : '-rotate-90'}`} />
                        </div>
                        {msg.isStreaming ? (
                          <ShinyText 
                            text="Agent is thinking..." 
                            className="text-[13px] font-medium tracking-tight"
                            color="#6b7280"
                            shineColor="#ffffff"
                            speed={1.5}
                            spread={100}
                          />
                        ) : (
                          <span className="text-[13px] font-medium tracking-tight opacity-80">
                            Thought process ({msg.steps?.length || 0} steps)
                          </span>
                        )}
                        {msg.isStreaming && (
                          <div className="flex gap-1 ml-1">
                            {[0, 1, 2].map(i => (
                              <motion.div 
                                key={i}
                                className="w-1 h-1 rounded-full bg-emerald-500/40"
                                animate={{ opacity: [0.2, 1, 0.2] }}
                                transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                              />
                            ))}
                          </div>
                        )}
                      </button>
                      
                      {/* Thinking Content */}
                      <AnimatePresence>
                        {isThinkingExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
                            className="overflow-hidden"
                          >
                            <div className="mt-5 ml-2.5 pl-6 border-l border-white/[0.1] space-y-10 py-1">
                              {msg.steps?.map((step, idx) => (
                                <motion.div
                                  key={step.id}
                                  initial={{ opacity: 0, y: 5 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  className="relative group"
                                >
                                  {/* Minimal Timeline Dot */}
                                  <div className="absolute -left-7 top-1.5 w-2 h-2 rounded-full bg-[#111114] border border-white/[0.2] group-hover:border-white/[0.4] transition-colors" />
                                  
                                  <div className="space-y-3.5">
                                    <div className="flex items-center gap-3">
                                      <span className="text-[10px] font-bold text-gray-600 uppercase tracking-[0.15em]">Step {step.iteration || idx + 1}</span>
                                      {step.action && (
                                        <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-sm bg-white/[0.03] ${actionColors[step.action] || 'text-gray-500'}`}>
                                          {step.action}
                                        </span>
                                      )}
                                    </div>
                                    
                                    <p className="text-[14px] text-gray-300 leading-[1.7] font-normal max-w-[95%]">
                                      {step.thought}
                                    </p>

                                    {/* Sub-data - very subtle */}
                                    <div className="flex flex-wrap gap-3 pt-1">
                                      {typeof step.progress === 'boolean' && (
                                        <div className={`font-mono text-[11px] px-2.5 py-1 rounded inline-flex items-center gap-2 ${
                                          step.progress
                                            ? 'text-emerald-400/70 bg-emerald-500/5'
                                            : 'text-gray-600 bg-white/[0.02]'
                                        }`}>
                                          <span className="opacity-60">result</span>
                                          <span className="font-semibold">{step.progress ? 'progress' : 'no-progress'}</span>
                                        </div>
                                      )}

                                      {typeof step.difference === 'number' && (
                                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                                          <span className="opacity-60">Δ</span>
                                          <span className="font-semibold">{step.difference.toFixed(4)}</span>
                                        </div>
                                      )}

                                      {step.status && (
                                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                                          <span className="opacity-60">status</span>
                                          <span className="font-semibold">{String(step.status)}</span>
                                        </div>
                                      )}
                                      {step.adb_commands && step.adb_commands.length > 0 && (
                                        <div className="font-mono text-[11px] text-gray-600 bg-white/[0.02] px-2.5 py-1 rounded inline-flex items-center gap-2">
                                          <Terminal className="w-2.5 h-2.5 opacity-40" />
                                          <span className="opacity-20">$</span> {String(step.adb_commands[0] || '').substring(0, 45)}...
                                        </div>
                                      )}
                                      
                                      {(() => {
                                        const box = step.box_2d;
                                        if (!isValidBox2d(box)) return null;
                                        return (
                                          <TargetChip box={box} />
                                        );
                                      })()}
                                    </div>
                                  </div>
                                </motion.div>
                              ))}

                              {/* Streaming Indicator inside expanded view */}
                              {msg.isStreaming && (
                                <motion.div 
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  className="relative"
                                >
                                  <div className="absolute -left-7 top-1.5 w-2 h-2 rounded-full bg-emerald-500/30 animate-pulse" />
                                  <span className="text-[13px] text-gray-600 font-medium animate-pulse pl-1">Thinking...</span>
                                </motion.div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  {/* Result Message - HITL Card Style (동일한 스타일) */}
                  {msg.type === 'result' && (
                    <div className="flex items-start gap-5 my-10 pl-1">
                      <div className="flex-1">
                        <div className="relative">
                          {/* Glow */}
                          <div className="absolute -inset-1.5 rounded-[28px] bg-gradient-to-r from-emerald-500/18 via-emerald-500/8 to-emerald-500/10 blur-xl opacity-60" />

                          {/* Card - HITL과 동일한 스타일 */}
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
                                    {/* TTS 버튼 - HITL과 동일한 스타일 */}
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
                                  onClick={() => {
                                    const idx = messages.findIndex(m => m.id === msg.id);
                                    const prompt =
                                      idx >= 0
                                        ? [...messages]
                                            .slice(0, idx)
                                            .reverse()
                                            .find(m => m.type === 'user')?.content
                                        : undefined;
                                    if (!prompt || isRunning) return;

                                    seenInterruptsRef.current.clear();
                                    setIsRunning(true);
                                    setRequiresHumanInput(false);
                                    setActiveInterruptId(null);
                                    setCurrentStep(0);
                                    const newThreadId = crypto.randomUUID();
                                    setThreadId(newThreadId);
                                    startStream('/api/agent/start', {
                                      instruction: prompt,
                                      thread_id: newThreadId,
                                      model: selectedModel,
                                      enable_planning: enablePlanning,
                                    });
                                  }}
                                  disabled={isRunning}
                                  className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all text-xs text-gray-400 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
                                  title="재시도"
                                >
                                  <RotateCcw className="w-3.5 h-3.5" />
                                  <span>재시도</span>
                                </button>

                                <button
                                  type="button"
                                  onClick={() => setMessageFeedback(msg.id, msg.feedback === 'up' ? null : 'up')}
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
                                  onClick={() => setMessageFeedback(msg.id, msg.feedback === 'down' ? null : 'down')}
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
                                  onClick={() => copyMessage(msg.id, msg.content)}
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
                  )}

                  {/* Interrupt - Elegant Notice Style with Character */}
                  {msg.type === 'interrupt' && (
                    <div className="flex items-start gap-5 my-10 pl-1">
                      <div className="flex-1">
                        <InterruptChoiceCard
                          question={msg.content}
                          options={msg.options}
                          reason={msg.reason}
                          audioPath={msg.audioPath}
                          character={msg.character}
                          disabled={
                            isRunning ||
                            !requiresHumanInput ||
                            !threadId ||
                            (activeInterruptId !== null && msg.id !== activeInterruptId)
                          }
                          onSelect={submitInterruptOption}
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </AnimatePresence>

          {/* Active Streaming Indicator (when no thinking block yet) - hide during HITL */}
          {isRunning && !requiresHumanInput && !activeInterruptId && !(enablePlanning && showPlanningPanel) && messages.filter(m => m.type === 'thinking' && m.isStreaming).length === 0 && (
            <div className="flex items-center gap-3 text-gray-500 pl-2 animate-pulse">
              <Loader2 className="w-4 h-4 animate-spin opacity-40" />
              <span className="text-[13px] font-medium tracking-tight">Initializing agent...</span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>


        {/* Status Bar */}
        {(isRunning || currentStep > 0) && (
          <div className="px-6 py-2.5 border-t border-white/[0.03] bg-black/20 flex items-center justify-between text-[10px] font-bold text-gray-600 uppercase tracking-[0.2em]">
            <div className="flex items-center gap-4">
              <span>Step {currentStep}</span>
              {isRunning && <span className="text-emerald-500/60 flex items-center gap-1.5"><span className="w-1 h-1 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.4)]" /> Active</span>}
              {requiresHumanInput && <span className="text-yellow-500/60">Awaiting Response</span>}
            </div>
          </div>
        )}

        <ChatInputBar
          inputValue={inputValue}
          setInputValue={setInputValue}
          requiresHumanInput={requiresHumanInput}
          isRunning={isRunning}
          isRecording={isRecording}
          isTranscribing={isTranscribing}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          showModelSelector={showModelSelector}
          setShowModelSelector={setShowModelSelector}
          enablePlanning={enablePlanning}
          setEnablePlanning={setEnablePlanning}
          onSubmit={handleSubmit}
          onStop={handleStop}
          disableSubmit={!isRunning && !requiresHumanInput && !inputValue.trim()}
        />
      </div>
      </div>
    </main>
  );
}
