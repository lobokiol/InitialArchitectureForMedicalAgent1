import { ChevronRight } from 'lucide-react';
import { IntentBadge } from './IntentBadge';
import type { ChatResponse } from '../types';

interface InfoBarProps {
  response: ChatResponse | null;
  onOpenDetail: () => void;
}

export function InfoBar({ response, onOpenDetail }: InfoBarProps) {
  const conf = response?.dept_confidence;
  const passed = response?.dept_confidence_passed;
  const locked = response?.locked_department;

  return (
    <div className="shrink-0 h-10 border-t border-brand-500/20 bg-white px-3 md:px-6 flex items-center justify-between gap-2 text-xs overflow-hidden">
      <div className="flex items-center gap-2 min-w-0 overflow-hidden">
        <IntentBadge intent={response?.intent_result} />
        {conf != null && (
          <span className={`shrink-0 ${passed ? 'text-brand-600' : 'text-amber-600'}`}>
            置信度 {Math.round(conf)}
            {passed ? ' ✓' : ''}
          </span>
        )}
        {locked && <span className="text-gray-500 truncate hidden sm:inline">→ {locked}</span>}
      </div>
      <button
        type="button"
        onClick={onOpenDetail}
        disabled={!response}
        className="shrink-0 flex items-center gap-0.5 text-brand-600 hover:text-brand-700 disabled:opacity-40 font-medium"
      >
        详情
        <ChevronRight size={14} />
      </button>
    </div>
  );
}
