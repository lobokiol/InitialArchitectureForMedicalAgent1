export interface IntentResult {
  has_symptom: boolean;
  has_process: boolean;
  main_intent: 'symptom' | 'process' | 'mixed' | 'non_medical';
  symptom_query?: string;
  process_query?: string;
  triage_route?: 'disease' | 'symptom' | 'reject';
}

export interface RetrievedDoc {
  id: string;
  source: 'medical' | 'process' | 'tool';
  title?: string;
  content: string;
  score?: number;
}

export interface ClarifyChoice {
  id: string;
  label: string;
  slot?: string;
}

export interface DeptChoice {
  id: string;
  label: string;
  target_departments: string[];
}

export interface OnCallDoctor {
  name: string;
  time: string;
  slots: number;
}

export interface ChatResponse {
  user_id: string;
  thread_id: string;
  reply: string;
  intent_result?: IntentResult;
  used_docs: { medical: RetrievedDoc[]; process: RetrievedDoc[] };
  awaiting_clarify: boolean;
  clarify_phase?: string;
  clarify_choices: ClarifyChoice[];
  awaiting_dept_choice: boolean;
  dept_choices: DeptChoice[];
  multi_select: boolean;
  dept_confidence?: number;
  dept_confidence_passed?: boolean;
  dept_confidence_reason?: string;
  locked_department?: string;
  oncall_appointments?: OnCallDoctor[];
  oncall_fetch_error?: string;
  node_trace: string[];
  app_state?: Record<string, unknown>;
}

export interface ThreadInfo {
  thread_id: string;
  title: string;
  created_at: string;
  last_active_at: string;
  is_deleted: boolean;
}

export interface UserInfo {
  user_id: string;
  name?: string;
}

export interface ReadyResponse {
  status: 'ok' | 'degraded';
  checks?: Record<string, { ok: boolean; detail?: string }>;
}

export interface Turn {
  id: string;
  userMessage: string;
  assistantReply: string;
  chatSnapshot?: ChatResponse;
  choices?: {
    type: 'clarify' | 'dept';
    phase?: string;
    items: ClarifyChoice[] | DeptChoice[];
    multiSelect?: boolean;
  };
}

export type ChatPhase = 'idle' | 'loading' | 'awaiting_clarify' | 'awaiting_dept' | 'settled';

export type SlashCommand =
  | { type: 'help' }
  | { type: 'new' }
  | { type: 'threads' }
  | { type: 'switch'; threadId: string }
  | { type: 'delete' }
  | { type: 'user' }
  | { type: 'exit' }
  | { type: 'message'; text: string };
