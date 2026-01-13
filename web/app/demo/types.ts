export interface ThinkingStep {
  id: string;
  thought: string;
  action?: string;
  iteration?: number;
  status?: string;
  adb_commands?: string[];
  box_2d?: number[]; // 바운딩박스 좌표 [y1, x1, y2, x2]
  progress?: boolean;
  difference?: number;
}

export interface CharacterInfo {
  id: string;
  name: string;
  nickname: string;
  imagePath: string;
}

export interface ChatMessage {
  id: string;
  type: "user" | "thinking" | "result" | "system" | "interrupt";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  steps?: ThinkingStep[];
  options?: string[];
  reason?: string;
  feedback?: "up" | "down" | null;
  audioPath?: string | null; // TTS audio path for interrupt messages
  character?: CharacterInfo | null; // Character for this message
}

export interface AGUIEvent {
  type: string;
  snapshot?: Record<string, any>;
  message?: string;
  name?: string;
  value?: any;
  // AG-UI canonical fields (camelCase)
  threadId?: string;
  runId?: string;
  timestamp?: number;
  result?: any;
  // Backward-compat fields (snake_case)
  thread_id?: string;
  run_id?: string;
  // Character info (backend sends as 'chef' for compatibility)
  chef?: CharacterInfo;
}

