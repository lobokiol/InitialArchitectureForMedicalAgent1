import axios from 'axios';
import type { ChatResponse, ReadyResponse, ThreadInfo, UserInfo } from '../types';

// 开发默认 /api 走 Vite 代理；生产可设 VITE_API_BASE=http://host:8000
const baseURL = import.meta.env.VITE_API_BASE || '/api';

export const api = axios.create({ baseURL, timeout: 60_000 });

export function getErrorDetail(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const d = err.response?.data?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x) => x.msg ?? String(x)).join('; ');
    return err.message;
  }
  return err instanceof Error ? err.message : '未知错误';
}

export async function postChat(body: {
  user_id: string;
  thread_id?: string;
  message: string;
}): Promise<ChatResponse> {
  const payload = body.thread_id
    ? body
    : { user_id: body.user_id, message: body.message };
  const { data } = await api.post<ChatResponse>('/chat', payload);
  return data;
}

export async function getThreads(userId: string): Promise<ThreadInfo[]> {
  const { data } = await api.get<ThreadInfo[]>('/threads', { params: { user_id: userId } });
  return data;
}

export async function createThread(userId: string, title?: string) {
  const { data } = await api.post<{ thread_id: string; title: string }>('/threads', {
    user_id: userId,
    title,
  });
  return data;
}

export async function deleteThread(userId: string, threadId: string) {
  const { data } = await api.delete<{ deleted: boolean; new_current_thread_id?: string }>(
    `/threads/${threadId}`,
    { params: { user_id: userId } },
  );
  return data;
}

export async function getCurrentThread(userId: string): Promise<ThreadInfo> {
  const { data } = await api.get<ThreadInfo>('/threads/current', {
    params: { user_id: userId },
  });
  return data;
}

export async function switchThread(userId: string, threadId: string) {
  await api.post('/threads/switch', { user_id: userId, thread_id: threadId });
}

export async function upsertUser(userId: string, name?: string): Promise<UserInfo> {
  const { data } = await api.post<UserInfo>('/users', { user_id: userId, name });
  return data;
}

export async function getUser(userId: string): Promise<UserInfo> {
  const { data } = await api.get<UserInfo>(`/users/${userId}`);
  return data;
}

export async function getReady(): Promise<ReadyResponse> {
  const { data } = await api.get<ReadyResponse>('/ready');
  return data;
}
