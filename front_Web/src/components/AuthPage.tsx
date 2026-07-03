import { useState } from 'react';
import { motion } from 'framer-motion';
import { getErrorDetail, login, register } from '../lib/api';
import { setTokens } from '../lib/auth';

type Tab = 'login' | 'register';

interface AuthPageProps {
  onAuthed: () => void | Promise<void>;
}

export function AuthPage({ onAuthed }: AuthPageProps) {
  const [tab, setTab] = useState<Tab>('login');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    const trimmedPhone = phone.trim();
    const trimmedPassword = password.trim();
    if (!trimmedPhone || !trimmedPassword) return;
    if (tab === 'register' && trimmedPassword.length < 8) {
      setError('密码至少 8 位');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const tokens =
        tab === 'login'
          ? await login(trimmedPhone, trimmedPassword)
          : await register(trimmedPhone, trimmedPassword, displayName.trim() || undefined);
      setTokens(tokens.access_token, tokens.refresh_token);
      await onAuthed();
    } catch (err) {
      setError(getErrorDetail(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-brand-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-white rounded-2xl shadow-xl p-6"
      >
        <h1 className="text-lg font-semibold text-brand-700 mb-1">智能导诊助手</h1>
        <p className="text-sm text-gray-500 mb-5">使用手机号登录或注册后开始对话</p>

        <div className="flex gap-1 mb-5 p-1 bg-gray-100 rounded-lg">
          <button
            type="button"
            onClick={() => {
              setTab('login');
              setError(null);
            }}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
              tab === 'login' ? 'bg-white text-brand-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            登录
          </button>
          <button
            type="button"
            onClick={() => {
              setTab('register');
              setError(null);
            }}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
              tab === 'register' ? 'bg-white text-brand-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            注册
          </button>
        </div>

        <label className="block text-sm text-gray-600 mb-1">手机号</label>
        <input
          type="tel"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          placeholder="13800138000"
          className="w-full mb-3 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-brand-500/40 outline-none"
        />

        <label className="block text-sm text-gray-600 mb-1">密码</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={tab === 'register' ? '至少 8 位' : '请输入密码'}
          className="w-full mb-3 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-brand-500/40 outline-none"
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
        />

        {tab === 'register' && (
          <>
            <label className="block text-sm text-gray-600 mb-1">昵称（可选）</label>
            <input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full mb-3 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-brand-500/40 outline-none"
            />
          </>
        )}

        {error && <p className="text-sm text-red-600 mb-3">{error}</p>}

        <button
          type="button"
          disabled={!phone.trim() || !password.trim() || loading}
          onClick={handleSubmit}
          className="w-full py-2.5 rounded-lg bg-brand-500 text-white font-medium hover:bg-brand-600 disabled:opacity-50"
        >
          {loading ? '处理中…' : tab === 'login' ? '登录' : '注册'}
        </button>
      </motion.div>
    </div>
  );
}
