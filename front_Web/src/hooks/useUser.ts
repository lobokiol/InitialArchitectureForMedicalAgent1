import { useCallback, useEffect, useState } from 'react';
import { clearTokens, getAccessToken } from '../lib/auth';
import { fetchMe, getErrorDetail } from '../lib/api';

export function useUser() {
  const [userId, setUserId] = useState('');
  const [userName, setUserName] = useState('');
  const [loading, setLoading] = useState(() => !!getAccessToken());
  const [needsOnboarding, setNeedsOnboarding] = useState(() => !getAccessToken());

  const refreshFromMe = useCallback(async () => {
    if (!getAccessToken()) {
      setUserId('');
      setUserName('');
      setNeedsOnboarding(true);
      return { userId: '', degraded: true as const, error: '未登录' };
    }

    setLoading(true);
    try {
      const me = await fetchMe();
      setUserId(me.phone);
      setUserName(me.display_name ?? '');
      setNeedsOnboarding(false);
      return { userId: me.phone, degraded: false as const };
    } catch (err) {
      clearTokens();
      setUserId('');
      setUserName('');
      setNeedsOnboarding(true);
      return { userId: '', degraded: true as const, error: getErrorDetail(err) };
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (getAccessToken()) {
      void refreshFromMe();
    }
  }, [refreshFromMe]);

  const logout = useCallback(() => {
    clearTokens();
    setUserId('');
    setUserName('');
    setNeedsOnboarding(true);
  }, []);

  return { userId, userName, loading, needsOnboarding, refreshFromMe, logout };
}
