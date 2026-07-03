import axios, { type InternalAxiosRequestConfig } from 'axios';
import { clearTokens, getAccessToken, getRefreshToken, setTokens } from './auth';
import type { ChatResponse, ReadyResponse, ThreadInfo } from '../types';

// 开发默认 /api 走 Vite 代理；生产可设 VITE_API_BASE=http://host:8000
const baseURL = import.meta.env.VITE_API_BASE || '/api';

export const api = axios.create({ baseURL, timeout: 60_000 });

interface RetryConfig extends InternalAxiosRequestConfig {
  _retry?: boolean;
}

api.interceptors.request.use((cfg) => {
  const t = getAccessToken();
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});

let refreshing: Promise<void> | null = null;

api.interceptors.response.use(
  (r) => r,
  async (err) => {
    const cfg = err.config as RetryConfig | undefined;
    if (!cfg || err.response?.status !== 401 || cfg._retry) throw err;
    cfg._retry = true;
    const refresh = getRefreshToken();
    if (!refresh) {
      clearTokens();
      throw err;
    }
    if (!refreshing) {
      refreshing = api
        .post<TokenResponse>('/auth/refresh', { refresh_token: refresh })
        .then(({ data }) => setTokens(data.access_token, data.refresh_token ?? refresh))
        .catch((refreshErr) => {
          clearTokens();
          throw refreshErr;
        })
        .finally(() => {
          refreshing = null;
        });
    }
    await refreshing;
    return api(cfg);
  },
);

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type?: string;
}

export interface MeResponse {
  phone: string;
  display_name: string | null;
  created_at: string;
}

export function getErrorDetail(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const d = err.response?.data?.detail;
    if (typeof d === 'string') return d;
    if (d && typeof d === 'object' && 'detail' in d && typeof d.detail === 'string') {
      return d.detail;
    }
    if (Array.isArray(d)) return d.map((x) => x.msg ?? String(x)).join('; ');
    return err.message;
  }
  return err instanceof Error ? err.message : '未知错误';
}

export async function login(phone: string, password: string): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>('/auth/login', { phone, password });
  return data;
}

export async function register(
  phone: string,
  password: string,
  displayName?: string,
): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>('/auth/register', {
    phone,
    password,
    display_name: displayName,
  });
  return data;
}

export async function fetchMe(): Promise<MeResponse> {
  const { data } = await api.get<MeResponse>('/auth/me');
  return data;
}

export async function postChat(body: {
  thread_id?: string;
  message: string;
}): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', body);
  return data;
}

export async function getThreads(): Promise<ThreadInfo[]> {
  const { data } = await api.get<ThreadInfo[]>('/threads');
  return data;
}

export async function createThread(title?: string) {
  const { data } = await api.post<{ thread_id: string; title: string }>('/threads', { title });
  return data;
}

export async function deleteThread(threadId: string) {
  const { data } = await api.delete<{ deleted: boolean; new_current_thread_id?: string }>(
    `/threads/${threadId}`,
  );
  return data;
}

export async function getCurrentThread(): Promise<ThreadInfo> {
  const { data } = await api.get<ThreadInfo>('/threads/current');
  return data;
}

export async function switchThread(threadId: string) {
  await api.post('/threads/switch', { thread_id: threadId });
}

export async function getReady(): Promise<ReadyResponse> {
  const { data } = await api.get<ReadyResponse>('/ready');
  return data;
}
