import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import type { ReadyResponse } from '../types';

interface SettingsPanelProps {
  open: boolean;
  ready: ReadyResponse | null;
  apiError: boolean;
  userId: string;
  userName: string;
  onClose: () => void;
  onEditUser: () => void;
}

export function SettingsPanel({
  open,
  ready,
  apiError,
  userId,
  userName,
  onClose,
  onEditUser,
}: SettingsPanelProps) {
  const apiBase = import.meta.env.VITE_API_BASE || '/api';
  const apiHint =
    apiBase === '/api' ? '/api → 127.0.0.1:8000（Vite 代理）' : apiBase;

  if (!open) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={onClose}>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md bg-white rounded-2xl shadow-xl p-5"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between items-center mb-4">
            <h2 className="font-semibold text-brand-700">设置</h2>
            <button type="button" onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
              <X size={18} />
            </button>
          </div>
          <dl className="space-y-3 text-sm">
            <div>
              <dt className="text-gray-500">API 地址</dt>
              <dd className="font-mono text-xs mt-0.5">{apiHint}</dd>
            </div>
            <div>
              <dt className="text-gray-500">用户</dt>
              <dd className="mt-0.5">
                {userId} {userName && `(${userName})`}
                <button type="button" onClick={onEditUser} className="ml-2 text-brand-600 text-xs hover:underline">
                  编辑
                </button>
              </dd>
            </div>
            <div>
              <dt className="text-gray-500">就绪状态</dt>
              <dd className="mt-0.5">
                {apiError && <span className="text-red-600">无法连接 API，请确认 start-api.ps1 已启动</span>}
                {!apiError && ready && (
                  <span className={ready.status === 'ok' ? 'text-brand-600' : 'text-amber-600'}>
                    {ready.status === 'ok' ? '全部就绪' : '降级运行'}
                  </span>
                )}
              </dd>
            </div>
            {ready?.checks && (
              <ul className="space-y-1 pl-0">
                {Object.entries(ready.checks).map(([name, check]) => (
                  <li key={name} className="flex justify-between text-xs">
                    <span>{name}</span>
                    <span className={check.ok ? 'text-brand-600' : 'text-red-500'}>
                      {check.ok ? '✓' : check.detail ?? '未就绪'}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </dl>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
