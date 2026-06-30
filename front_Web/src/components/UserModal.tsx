import { useState } from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';

interface UserModalProps {
  open: boolean;
  forced?: boolean;
  loading?: boolean;
  onSubmit: (userId: string, name: string) => void;
  onClose?: () => void;
}

export function UserModal({ open, forced, loading, onSubmit, onClose }: UserModalProps) {
  const [userId, setUserId] = useState('demo-user');
  const [name, setName] = useState('');

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-white rounded-2xl shadow-xl p-6"
      >
        <div className="flex justify-between items-start mb-1">
          <h2 className="text-lg font-semibold text-brand-700">欢迎使用智能导诊助手</h2>
          {!forced && onClose && (
            <button type="button" onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
              <X size={18} />
            </button>
          )}
        </div>
        <p className="text-sm text-gray-500 mb-4">设置用户 ID 后即可开始多轮导诊对话</p>
        <label className="block text-sm text-gray-600 mb-1">用户 ID</label>
        <input
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="w-full mb-3 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-brand-500/40 outline-none"
        />
        <label className="block text-sm text-gray-600 mb-1">昵称（可选）</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full mb-4 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-brand-500/40 outline-none"
        />
        <button
          type="button"
          disabled={!userId.trim() || loading}
          onClick={() => onSubmit(userId.trim(), name.trim())}
          className="w-full py-2.5 rounded-lg bg-brand-500 text-white font-medium hover:bg-brand-600 disabled:opacity-50"
        >
          {loading ? '保存中…' : forced ? '开始' : '保存'}
        </button>
      </motion.div>
    </div>
  );
}
