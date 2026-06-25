import type { ChatPhase, ChatResponse } from '../types';
import type { Turn } from '../types';

export function extractChoices(data: ChatResponse): Turn['choices'] | undefined {
  if (data.awaiting_clarify && data.clarify_choices.length > 0) {
    return { type: 'clarify', phase: data.clarify_phase, items: data.clarify_choices };
  }
  if (data.awaiting_dept_choice && data.dept_choices.length > 0) {
    return { type: 'dept', items: data.dept_choices, multiSelect: data.multi_select };
  }
  return undefined;
}

export function getChatPhase(data: ChatResponse): ChatPhase {
  if (data.awaiting_clarify && data.clarify_choices.length > 0) return 'awaiting_clarify';
  if (data.awaiting_dept_choice && data.dept_choices.length > 0) return 'awaiting_dept';
  return 'settled';
}

export function formatDeptSelection(zeroBasedIndices: number[]): string {
  return zeroBasedIndices.map((i) => i + 1).join(',');
}

export function applyResponseToTurn(turn: Turn, data: ChatResponse): Turn {
  return {
    ...turn,
    assistantReply: data.reply,
    chatSnapshot: data,
    choices: extractChoices(data),
  };
}
