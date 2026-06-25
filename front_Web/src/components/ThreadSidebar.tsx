import { Plus, Trash2 } from 'lucide-react';
import type { ThreadInfo } from '../types';

interface ThreadSidebarProps {
  threads: ThreadInfo[];
  currentThreadId: string;
  loading: boolean;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  onDelete: (threadId: string) => void;
}

export function ThreadSidebar({
  threads,
  currentThreadId,
  loading,
  onSelect,
  onNew,
  onDelete,
}: ThreadSidebarProps) {
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="p-3 border-b border-brand-500/10">
        <button
          type="button"
          onClick={onNew}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:bg-brand-600 transition-colors"
        >
          <Plus size={16} />
          新建会话
        </button>
      </div>
      <div className="flex-1 min-h-0 p-2 space-y-1">
        {loading && threads.length === 0 && (
          <p className="text-xs text-gray-400 text-center py-4">加载中…</p>
        )}
        {threads.map((t) => {
          const active = t.thread_id === currentThreadId;
          return (
            <div
              key={t.thread_id}
              className={`group flex items-center gap-1 rounded-lg ${
                active ? 'bg-brand-50 border border-brand-500/30' : 'hover:bg-gray-50'
              }`}
            >
              <button
                type="button"
                onClick={() => onSelect(t.thread_id)}
                className="flex-1 text-left px-3 py-2 min-w-0"
              >
                <p className={`text-sm truncate ${active ? 'text-brand-700 font-medium' : 'text-gray-700'}`}>
                  {t.title}
                </p>
                <p className="text-[10px] text-gray-400 truncate">{t.thread_id.slice(0, 8)}…</p>
              </button>
              <button
                type="button"
                onClick={() => onDelete(t.thread_id)}
                className="p-2 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                aria-label="删除会话"
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
