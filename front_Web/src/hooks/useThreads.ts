import { useCallback, useEffect, useState } from 'react';
import {
  createThread,
  deleteThread,
  getCurrentThread,
  getErrorDetail,
  getThreads,
  switchThread,
} from '../lib/api';
import type { ThreadInfo } from '../types';

const LS_THREAD = 'triage_demo_thread_id';

export function useThreads(userId: string) {
  const [threads, setThreads] = useState<ThreadInfo[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState('');
  const [currentTitle, setCurrentTitle] = useState('默认对话');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyCurrent = useCallback((threadId: string, title: string, list?: ThreadInfo[]) => {
    setCurrentThreadId(threadId);
    setCurrentTitle(title);
    localStorage.setItem(LS_THREAD, threadId);
    if (list) setThreads(list.filter((t) => !t.is_deleted));
  }, []);

  const refresh = useCallback(
    async (overrideUserId?: string) => {
      const uid = overrideUserId ?? userId;
      if (!uid) return;
      setLoading(true);
      setError(null);
      try {
        try {
          const [list, current] = await Promise.all([
            getThreads(uid),
            getCurrentThread(uid),
          ]);
          applyCurrent(current.thread_id, current.title, list);
          return;
        } catch {
          // getCurrentThread 偶发 500 时降级
        }

        try {
          const list = (await getThreads(uid)).filter((t) => !t.is_deleted);
          setThreads(list);
          const cached = localStorage.getItem(LS_THREAD);
          const pick = list.find((t) => t.thread_id === cached) ?? list[0];
          if (pick) {
            try {
              await switchThread(uid, pick.thread_id);
            } catch {
              // switch 失败仍用该 thread 发消息
            }
            applyCurrent(pick.thread_id, pick.title);
            return;
          }
        } catch (err) {
          setError(getErrorDetail(err));
        }

        try {
          const created = await createThread(uid);
          applyCurrent(created.thread_id, created.title);
          const list = await getThreads(uid);
          setThreads(list.filter((t) => !t.is_deleted));
          return;
        } catch (err) {
          const detail = getErrorDetail(err);
          const cached = localStorage.getItem(LS_THREAD);
          if (cached) {
            applyCurrent(cached, '缓存会话');
            setError(`会话服务异常，使用缓存 thread（${detail}）`);
            return;
          }
          setError(detail);
          setCurrentThreadId('');
          setCurrentTitle('默认对话');
        }
      } finally {
        setLoading(false);
      }
    },
    [userId, applyCurrent],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createNew = useCallback(async () => {
    const t = await createThread(userId);
    await switchThread(userId, t.thread_id);
    await refresh();
    return t.thread_id;
  }, [userId, refresh]);

  const switchTo = useCallback(
    async (threadId: string) => {
      await switchThread(userId, threadId);
      const found = threads.find((t) => t.thread_id === threadId);
      applyCurrent(threadId, found?.title ?? '对话');
    },
    [userId, threads, applyCurrent],
  );

  const remove = useCallback(
    async (threadId: string) => {
      const res = await deleteThread(userId, threadId);
      if (res.new_current_thread_id) {
        applyCurrent(res.new_current_thread_id, '默认对话');
      }
      await refresh();
    },
    [userId, refresh, applyCurrent],
  );

  return {
    threads,
    currentThreadId,
    currentTitle,
    refresh,
    createNew,
    switchTo,
    remove,
    loading,
    error,
  };
}
