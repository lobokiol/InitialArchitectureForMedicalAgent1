import { useEffect, useState } from 'react';
import { getReady } from '../lib/api';
import type { ReadyResponse } from '../types';

export function useReadiness(intervalMs = 30_000) {
  const [ready, setReady] = useState<ReadyResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await getReady();
        if (!cancelled) {
          setReady(r);
          setError(false);
        }
      } catch {
        if (!cancelled) {
          setReady(null);
          setError(true);
        }
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  const dotColor = error ? 'bg-red-500' : ready?.status === 'ok' ? 'bg-brand-500' : 'bg-amber-400';
  const label = error ? 'API 未连接' : ready?.status === 'ok' ? '就绪' : '降级';

  return { ready, error, dotColor, label };
}
