import { useState, useCallback, useEffect } from 'react';
import { getThreads, createThread, deleteThread, getCurrentThread, switchThread } from '../lib/api';
import type { ThreadInfo, ThreadState } from '../types';

const THREAD_STORAGE_KEY = 'hospital_guide_current_thread';

export function useThreads(userId: string | null) {
  const [threads, setThreads] = useState<ThreadInfo[]>([]);
  const [currentThread, setCurrentThread] = useState<ThreadState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showThreadList, setShowThreadList] = useState(false);

  const loadCurrentThread = useCallback(async () => {
    if (!userId) return;
    try {
      const thread = await getCurrentThread(userId);
      setCurrentThread({ thread_id: thread.thread_id, title: thread.title });
      localStorage.setItem(THREAD_STORAGE_KEY, JSON.stringify({ thread_id: thread.thread_id, title: thread.title }));
    } catch (err) {
      setError('获取当前会话失败');
    }
  }, [userId]);

  const loadThreads = useCallback(async () => {
    if (!userId) return;
    setIsLoading(true);
    try {
      const list = await getThreads(userId);
      setThreads(list.filter(t => !t.is_deleted));
    } catch (err) {
      setError('获取会话列表失败');
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    if (userId) {
      loadCurrentThread();
      loadThreads();
    }
  }, [userId, loadCurrentThread, loadThreads]);

  const createNewThread = useCallback(async (title?: string) => {
    if (!userId) return null;
    setIsLoading(true);
    try {
      const thread = await createThread(userId, title);
      const newThreadState = { thread_id: thread.thread_id, title: thread.title };
      setCurrentThread(newThreadState);
      localStorage.setItem(THREAD_STORAGE_KEY, JSON.stringify(newThreadState));
      await loadThreads();
      return thread;
    } catch (err) {
      setError('创建会话失败');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [userId, loadThreads]);

  const switchToThread = useCallback(async (threadId: string) => {
    if (!userId) return;
    setIsLoading(true);
    try {
      const thread = await switchThread(userId, threadId);
      const newThreadState = { thread_id: thread.thread_id, title: thread.title };
      setCurrentThread(newThreadState);
      localStorage.setItem(THREAD_STORAGE_KEY, JSON.stringify(newThreadState));
      setShowThreadList(false);
    } catch (err) {
      setError('切换会话失败');
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  const removeThread = useCallback(async (threadId: string) => {
    if (!userId) return;
    setIsLoading(true);
    try {
      const result = await deleteThread(threadId, userId);
      if (result.new_current_thread_id) {
        await switchToThread(result.new_current_thread_id);
      } else {
        await loadCurrentThread();
      }
      await loadThreads();
    } catch (err) {
      setError('删除会话失败');
    } finally {
      setIsLoading(false);
    }
  }, [userId, loadCurrentThread, loadThreads, switchToThread]);

  const openThreadList = useCallback(() => setShowThreadList(true), []);
  const closeThreadList = useCallback(() => setShowThreadList(false), []);

  return {
    threads,
    currentThread,
    isLoading,
    error,
    showThreadList,
    createNewThread,
    switchToThread,
    removeThread,
    openThreadList,
    closeThreadList,
    refreshThreads: loadThreads,
  };
}
