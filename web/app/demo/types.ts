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
  type: "user" | "thinking" | "result" | "system" | "interrupt" | "adb_error";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  steps?: ThinkingStep[];
  options?: string[];
  reason?: string;
  feedback?: "up" | "down" | null;
  audioPath?: string | null; // TTS audio path for interrupt messages
  character?: CharacterInfo | null; // Character for this message
  metadata?: Record<string, any>; // Additional metadata (e.g., hint for adb_error)
}

export interface PlanningState {
  phase: "planning";
  status: "detecting_unknown" | "web_search_complete" | "web_search_skipped" | "planning_complete" | "";
  unknownEntities: string[];
  searchContext: string;
  plan: string[];
  // To-do style step tracking
  planStepIndex: number;  // Current step being executed
  stepCompleted?: boolean;  // True when a step just completed
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
  // Character info
  character?: CharacterInfo;
  // Planning mode enabled flag
  planningEnabled?: boolean;
}
