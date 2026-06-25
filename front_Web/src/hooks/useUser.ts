import { useCallback, useState } from 'react';
import { getErrorDetail, getUser, upsertUser } from '../lib/api';

const LS_USER = 'triage_demo_user_id';
const LS_NAME = 'triage_demo_user_name';

export function useUser() {
  const [userId, setUserId] = useState(() => localStorage.getItem(LS_USER) ?? '');
  const [userName, setUserName] = useState(() => localStorage.getItem(LS_NAME) ?? '');
  const [loading, setLoading] = useState(false);

  const initUser = useCallback(async (id: string, name?: string) => {
    setLoading(true);
    try {
      const u = await upsertUser(id, name);
      setUserId(u.user_id);
      const displayName = u.name ?? name ?? '';
      setUserName(displayName);
      localStorage.setItem(LS_USER, u.user_id);
      localStorage.setItem(LS_NAME, displayName);
      return { userId: u.user_id, degraded: false as const };
    } catch (err) {
      // /users 后端异常时仍允许本地 Demo 继续（会话 API 不依赖 user 记录）
      setUserId(id);
      const displayName = name ?? '';
      setUserName(displayName);
      localStorage.setItem(LS_USER, id);
      localStorage.setItem(LS_NAME, displayName);
      return {
        userId: id,
        degraded: true as const,
        error: getErrorDetail(err),
      };
    } finally {
      setLoading(false);
    }
  }, []);

  const loadUser = useCallback(async (id: string) => {
    try {
      const u = await getUser(id);
      setUserName(u.name ?? '');
      return u;
    } catch {
      return null;
    }
  }, []);

  const needsOnboarding = !userId;

  return { userId, userName, initUser, loadUser, loading, needsOnboarding };
}
