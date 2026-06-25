import { motion, AnimatePresence } from 'framer-motion';
import { X, Command } from 'lucide-react';

interface CommandHelpProps {
  isOpen: boolean;
  onClose: () => void;
}

const commands = [
  { cmd: '/help', desc: '显示帮助信息' },
  { cmd: '/new', desc: '创建新会话' },
  { cmd: '/threads', desc: '查看所有会话' },
  { cmd: '/switch', desc: '切换会话' },
  { cmd: '/delete', desc: '删除当前会话' },
  { cmd: '/user', desc: '用户设置' },
];

export function CommandHelp({ isOpen, onClose }: CommandHelpProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/50"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="relative bg-white rounded-2xl shadow-xl w-full max-w-md p-6"
          >
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-1 text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                <Command className="w-5 h-5 text-gray-600" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900">可用命令</h2>
                <p className="text-sm text-gray-500">输入命令执行相应操作</p>
              </div>
            </div>

            <div className="space-y-2">
              {commands.map((item) => (
                <div
                  key={item.cmd}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-xl"
                >
                  <code className="font-mono text-primary-600 bg-primary-50 px-2 py-1 rounded">
                    {item.cmd}
                  </code>
                  <span className="text-gray-600">{item.desc}</span>
                </div>
              ))}
            </div>

            <p className="mt-4 text-xs text-gray-400 text-center">
              输入框中输入命令后按 Enter 执行
            </p>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
