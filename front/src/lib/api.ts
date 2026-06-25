import axios from 'axios';
import type { ChatRequest, ChatResponse, ThreadInfo, UserInfo } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chat = async (data: ChatRequest): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/chat', data);
  return response.data;
};

export const healthCheck = async (): Promise<{ status: string }> => {
  const response = await api.get<{ status: string }>('/healthz');
  return response.data;
};

export const getThreads = async (userId: string): Promise<ThreadInfo[]> => {
  const response = await api.get<ThreadInfo[]>('/threads', { params: { user_id: userId } });
  return response.data;
};

export const createThread = async (userId: string, title?: string): Promise<ThreadInfo> => {
  const response = await api.post<ThreadInfo>('/threads', { user_id: userId, title });
  return response.data;
};

export const deleteThread = async (threadId: string, userId: string): Promise<{ deleted: boolean; new_current_thread_id?: string }> => {
  const response = await api.delete<{ deleted: boolean; new_current_thread_id?: string }>(`/threads/${threadId}`, { params: { user_id: userId } });
  return response.data;
};

export const getCurrentThread = async (userId: string): Promise<ThreadInfo> => {
  const response = await api.get<ThreadInfo>('/threads/current', { params: { user_id: userId } });
  return response.data;
};

export const switchThread = async (userId: string, threadId: string): Promise<ThreadInfo> => {
  const response = await api.post<ThreadInfo>('/threads/switch', { user_id: userId, thread_id: threadId });
  return response.data;
};

export const createUser = async (userId: string, name?: string): Promise<UserInfo> => {
  const response = await api.post<UserInfo>('/users', { user_id: userId, name });
  return response.data;
};

export const getUser = async (userId: string): Promise<UserInfo> => {
  const response = await api.get<UserInfo>(`/users/${userId}`);
  return response.data;
};

export default api;
