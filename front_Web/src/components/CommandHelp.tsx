import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

interface CommandHelpProps {
  open: boolean;
  onClose: () => void;
}

const commands = [
  ['/help', '显示命令列表'],
  ['/new', '创建新会话并切换'],
  ['/threads', '打开会话列表'],
  ['/switch <id>', '切换到指定会话'],
  ['/delete', '删除当前会话'],
  ['/user', '编辑用户信息'],
  ['/exit', '关闭面板（Web 端）'],
];

export function CommandHelp({ open, onClose }: CommandHelpProps) {
  if (!open) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={onClose}>
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-sm bg-white rounded-2xl shadow-xl p-5"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between items-center mb-3">
            <h2 className="font-semibold text-brand-700">斜杠命令</h2>
            <button type="button" onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
              <X size={18} />
            </button>
          </div>
          <table className="w-full text-sm">
            <tbody>
              {commands.map(([cmd, desc]) => (
                <tr key={cmd} className="border-t border-gray-100">
                  <td className="py-2 font-mono text-brand-700 pr-3">{cmd}</td>
                  <td className="py-2 text-gray-600">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
