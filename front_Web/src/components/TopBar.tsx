import { HelpCircle, Menu, Settings } from 'lucide-react';
import { ReadinessDot } from './ReadinessDot';

interface TopBarProps {
  userId: string;
  threadTitle: string;
  dotColor: string;
  readinessLabel: string;
  onOpenDrawer: () => void;
  onOpenHelp: () => void;
  onOpenSettings: () => void;
}

export function TopBar({
  userId,
  threadTitle,
  dotColor,
  readinessLabel,
  onOpenDrawer,
  onOpenHelp,
  onOpenSettings,
}: TopBarProps) {
  return (
    <header className="h-12 shrink-0 border-b border-brand-500/20 bg-white flex items-center justify-between px-3 md:px-4">
      <div className="flex items-center gap-2 min-w-0">
        <button
          type="button"
          className="md:hidden p-1.5 rounded-lg hover:bg-brand-50 text-brand-700"
          onClick={onOpenDrawer}
          aria-label="打开会话"
        >
          <Menu size={20} />
        </button>
        <div className="min-w-0">
          <p className="text-sm font-semibold text-brand-700 truncate">智能导诊助手</p>
          <p className="text-xs text-gray-500 truncate">
            {userId || '未登录'} · {threadTitle}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <ReadinessDot color={dotColor} label={readinessLabel} />
        <button
          type="button"
          className="p-1.5 rounded-lg hover:bg-brand-50 text-brand-700"
          onClick={onOpenHelp}
          aria-label="帮助"
        >
          <HelpCircle size={20} />
        </button>
        <button
          type="button"
          className="p-1.5 rounded-lg hover:bg-brand-50 text-brand-700"
          onClick={onOpenSettings}
          aria-label="设置"
        >
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
}
