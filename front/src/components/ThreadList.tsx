import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageSquare, Plus, Trash2, Loader2 } from 'lucide-react';
import type { ThreadInfo } from '../types';

interface ThreadListProps {
  isOpen: boolean;
  onClose: () => void;
  threads: ThreadInfo[];
  currentThreadId?: string;
  onSwitch: (threadId: string) => void;
  onDelete: (threadId: string) => void;
  onCreate: (title?: string) => void;
  isLoading: boolean;
}

export function ThreadList({
  isOpen,
  onClose,
  threads,
  currentThreadId,
  onSwitch,
  onDelete,
  onCreate,
  isLoading,
}: ThreadListProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '刚刚';
    if (diffMins < 60) return `${diffMins}分钟前`;
    if (diffHours < 24) return `${diffHours}小时前`;
    if (diffDays < 7) return `${diffDays}天前`;
    return date.toLocaleDateString('zh-CN');
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/30"
            onClick={onClose}
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="relative w-full max-w-sm bg-white shadow-xl h-full flex flex-col"
          >
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-semibold text-gray-900">会话列表</h2>
              <button
                onClick={onClose}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 border-b">
              <button
                onClick={() => onCreate()}
                disabled={isLoading}
                className="w-full py-2.5 bg-primary-600 text-white font-medium rounded-xl hover:bg-primary-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                新建会话
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-2">
              {threads.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>暂无会话</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {threads.map((thread) => (
                    <div
                      key={thread.thread_id}
                      className={`group flex items-center gap-3 p-3 rounded-xl transition-colors ${
                        thread.thread_id === currentThreadId
                          ? 'bg-primary-50 border border-primary-200'
                          : 'hover:bg-gray-50'
                      }`}
                    >
                      <button
                        onClick={() => onSwitch(thread.thread_id)}
                        className="flex-1 text-left min-w-0"
                      >
                        <div className="font-medium text-gray-900 truncate">
                          {thread.title || '默认对话'}
                        </div>
                        <div className="text-xs text-gray-400">
                          {formatDate(thread.last_active_at)}
                        </div>
                      </button>
                      {thread.thread_id !== currentThreadId && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(thread.thread_id);
                          }}
                          className="p-1.5 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
