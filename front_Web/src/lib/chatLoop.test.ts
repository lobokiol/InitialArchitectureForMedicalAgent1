import { describe, it, expect } from 'vitest';
import { extractChoices, getChatPhase, formatDeptSelection } from './chatLoop';
import type { ChatResponse } from '../types';

const base: ChatResponse = {
  user_id: 'u',
  thread_id: 't',
  reply: 'hi',
  used_docs: { medical: [], process: [] },
  awaiting_clarify: false,
  clarify_choices: [],
  awaiting_dept_choice: false,
  dept_choices: [],
  multi_select: false,
  node_trace: [],
};

describe('extractChoices', () => {
  it('returns clarify when awaiting', () => {
    const data = {
      ...base,
      awaiting_clarify: true,
      clarify_phase: 'age',
      clarify_choices: [{ id: '1', label: '18-60' }],
    };
    expect(extractChoices(data)?.type).toBe('clarify');
  });

  it('returns dept when awaiting dept choice', () => {
    const data = {
      ...base,
      awaiting_dept_choice: true,
      dept_choices: [{ id: '1', label: '麻木', target_departments: ['骨科'] }],
      multi_select: true,
    };
    expect(extractChoices(data)?.type).toBe('dept');
    expect(extractChoices(data)?.multiSelect).toBe(true);
  });
});

describe('formatDeptSelection', () => {
  it('joins 1-based indices', () => {
    expect(formatDeptSelection([0, 2])).toBe('1,3');
  });
});

describe('getChatPhase', () => {
  it('returns awaiting_clarify', () => {
    const data = {
      ...base,
      awaiting_clarify: true,
      clarify_choices: [{ id: '1', label: 'x' }],
    };
    expect(getChatPhase(data)).toBe('awaiting_clarify');
  });

  it('returns settled when no choices', () => {
    expect(getChatPhase(base)).toBe('settled');
  });
});
