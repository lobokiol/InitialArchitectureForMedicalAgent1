import { useState, useCallback, useEffect, useRef } from 'react';
import { chat as chatApi } from '../lib/api';
import type { Message, ChatResponse } from '../types';

export function useChat(userId: string | null, threadId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(threadId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentThreadId(threadId);
  }, [threadId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || !userId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: content.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response: ChatResponse = await chatApi({
        user_id: userId,
        thread_id: currentThreadId || undefined,
        message: content.trim(),
      });

      if (response.thread_id && response.thread_id !== currentThreadId) {
        setCurrentThreadId(response.thread_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.reply,
        timestamp: Date.now(),
        intent_result: response.intent_result,
        used_docs: response.used_docs,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '请求失败，请检查后端服务';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [userId, currentThreadId]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    currentThreadId,
    sendMessage,
    clearMessages,
    messagesEndRef,
  };
}
