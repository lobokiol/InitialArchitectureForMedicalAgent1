import { useCallback, useReducer, useState } from 'react';
import { getErrorDetail, postChat } from '../lib/api';
import { applyResponseToTurn, getChatPhase } from '../lib/chatLoop';
import { parseSlashCommand } from '../lib/commands';
import type { ChatPhase, ChatResponse, SlashCommand, Turn } from '../types';

interface ChatState {
  turns: Turn[];
  viewIndex: number;
  phase: ChatPhase;
  lastResponse: ChatResponse | null;
  error: string | null;
}

type Action =
  | { type: 'CLEAR' }
  | { type: 'SET_VIEW'; index: number }
  | { type: 'START_LOADING' }
  | { type: 'ADD_TURN'; turn: Turn }
  | { type: 'UPDATE_TURN'; id: string; data: ChatResponse }
  | { type: 'SET_PHASE'; phase: ChatPhase }
  | { type: 'SET_ERROR'; error: string | null }
  | { type: 'SET_LAST'; data: ChatResponse };

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'CLEAR':
      return { turns: [], viewIndex: 0, phase: 'idle', lastResponse: null, error: null };
    case 'SET_VIEW':
      return {
        ...state,
        viewIndex: Math.max(0, Math.min(action.index, Math.max(0, state.turns.length - 1))),
      };
    case 'START_LOADING':
      return { ...state, phase: 'loading', error: null };
    case 'ADD_TURN':
      return {
        ...state,
        turns: [...state.turns, action.turn],
        viewIndex: state.turns.length,
        phase: 'loading',
        error: null,
      };
    case 'UPDATE_TURN': {
      const turns = state.turns.map((t) =>
        t.id === action.id ? applyResponseToTurn(t, action.data) : t,
      );
      const phase = getChatPhase(action.data);
      return {
        ...state,
        turns,
        phase: phase === 'settled' ? 'idle' : phase,
        lastResponse: action.data,
      };
    }
    case 'SET_PHASE':
      return { ...state, phase: action.phase };
    case 'SET_ERROR':
      return { ...state, phase: 'idle', error: action.error };
    case 'SET_LAST':
      return { ...state, lastResponse: action.data };
    default:
      return state;
  }
}

const initialState: ChatState = {
  turns: [],
  viewIndex: 0,
  phase: 'idle',
  lastResponse: null,
  error: null,
};

function newTurnId() {
  return `turn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export function useChat(userId: string, threadId: string) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [pendingCommand, setPendingCommand] = useState<SlashCommand | null>(null);

  const clearTurns = useCallback(() => {
    dispatch({ type: 'CLEAR' });
  }, []);

  const goPrev = useCallback(() => {
    dispatch({ type: 'SET_VIEW', index: state.viewIndex - 1 });
  }, [state.viewIndex]);

  const goNext = useCallback(() => {
    dispatch({ type: 'SET_VIEW', index: state.viewIndex + 1 });
  }, [state.viewIndex]);

  const postAndUpdate = useCallback(
    async (turnId: string, message: string, tid: string) => {
      dispatch({ type: 'START_LOADING' });
      try {
        const data = await postChat({
          user_id: userId,
          ...(tid ? { thread_id: tid } : {}),
          message,
        });
        dispatch({ type: 'UPDATE_TURN', id: turnId, data });
        return data;
      } catch (err) {
        dispatch({ type: 'SET_ERROR', error: getErrorDetail(err) });
        return null;
      }
    },
    [userId],
  );

  const pickChoice = useCallback(
    async (message: string) => {
      if (state.turns.length === 0) return null;
      const current = state.turns[state.viewIndex];
      if (!current) return null;
      return postAndUpdate(current.id, message, threadId);
    },
    [threadId, state.turns, state.viewIndex, postAndUpdate],
  );

  const sendMessage = useCallback(
    async (text: string): Promise<SlashCommand | null> => {
      const cmd = parseSlashCommand(text);
      if (cmd.type !== 'message') {
        setPendingCommand(cmd);
        return cmd;
      }
      if (!userId || !cmd.text.trim()) return null;
      // threadId 可省略，后端使用当前会话

      const turn: Turn = {
        id: newTurnId(),
        userMessage: cmd.text.trim(),
        assistantReply: '',
      };
      dispatch({ type: 'ADD_TURN', turn });
      await postAndUpdate(turn.id, cmd.text.trim(), threadId);
      return null;
    },
    [userId, threadId, postAndUpdate],
  );

  const consumeCommand = useCallback(() => {
    const c = pendingCommand;
    setPendingCommand(null);
    return c;
  }, [pendingCommand]);

  const currentTurn = state.turns[state.viewIndex] ?? null;

  return {
    turns: state.turns,
    viewIndex: state.viewIndex,
    currentTurn,
    phase: state.phase,
    lastResponse: state.lastResponse,
    error: state.error,
    sendMessage,
    pickChoice,
    clearTurns,
    goPrev,
    goNext,
    consumeCommand,
    pendingCommand,
    clearError: () => dispatch({ type: 'SET_ERROR', error: null }),
  };
}
