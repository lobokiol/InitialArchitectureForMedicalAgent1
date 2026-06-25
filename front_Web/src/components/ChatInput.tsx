import { Send } from 'lucide-react';
import { useState } from 'react';

interface ChatInputProps {
  disabled?: boolean;
  onSend: (text: string) => void;
}

export function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [text, setText] = useState('');

  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText('');
  };

  return (
    <div className="shrink-0 h-14 border-t border-brand-500/20 bg-white px-3 md:px-6 flex items-center gap-2">
      <input
        type="text"
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && submit()}
        placeholder="输入消息或 /help"
        className="flex-1 h-10 px-4 rounded-xl border border-brand-500/20 bg-brand-50/50 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40 disabled:opacity-50"
      />
      <button
        type="button"
        onClick={submit}
        disabled={disabled || !text.trim()}
        className="h-10 w-10 flex items-center justify-center rounded-xl bg-brand-500 text-white hover:bg-brand-600 disabled:opacity-50 transition-colors"
        aria-label="发送"
      >
        <Send size={18} />
      </button>
    </div>
  );
}
