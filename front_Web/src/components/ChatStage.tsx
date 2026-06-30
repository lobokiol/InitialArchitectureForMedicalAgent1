import { ChevronLeft, ChevronRight } from 'lucide-react';
import { AppointmentCards } from './AppointmentCards';
import { ClarifyChoices } from './ClarifyChoices';
import { DeptChoices } from './DeptChoices';
import { MessageBubble } from './MessageBubble';
import type { ChatPhase, ClarifyChoice, DeptChoice, Turn } from '../types';

interface ChatStageProps {
  turn: Turn | null;
  viewIndex: number;
  totalTurns: number;
  phase: ChatPhase;
  onPrev: () => void;
  onNext: () => void;
  onPickChoice: (message: string) => void;
  onExpandFull: (text: string) => void;
}

export function ChatStage({
  turn,
  viewIndex,
  totalTurns,
  phase,
  onPrev,
  onNext,
  onPickChoice,
  onExpandFull,
}: ChatStageProps) {
  const busy = phase === 'loading';
  const canPick = phase === 'awaiting_clarify' || phase === 'awaiting_dept';
  const isLatest = viewIndex === totalTurns - 1;

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden px-3 md:px-6 py-3">
      <div className="flex items-center justify-center gap-3 shrink-0 mb-3">
        <button
          type="button"
          onClick={onPrev}
          disabled={viewIndex <= 0}
          className="p-1 rounded-lg hover:bg-brand-50 disabled:opacity-30 text-brand-700"
          aria-label="上一条"
        >
          <ChevronLeft size={20} />
        </button>
        <span className="text-xs text-gray-500 tabular-nums min-w-[3rem] text-center">
          {totalTurns === 0 ? '0/0' : `${viewIndex + 1}/${totalTurns}`}
        </span>
        <button
          type="button"
          onClick={onNext}
          disabled={viewIndex >= totalTurns - 1}
          className="p-1 rounded-lg hover:bg-brand-50 disabled:opacity-30 text-brand-700"
          aria-label="下一条"
        >
          <ChevronRight size={20} />
        </button>
      </div>

      <div className="flex-1 flex flex-col justify-center min-h-0 overflow-hidden gap-3">
        {!turn && (
          <p className="text-center text-sm text-gray-400">发送症状开始导诊，或输入 /help 查看命令</p>
        )}
        {turn && (
          <>
            <MessageBubble role="user" content={turn.userMessage} />
            <MessageBubble
              role="assistant"
              content={turn.assistantReply}
              loading={busy && !turn.assistantReply}
              onExpandFull={() => onExpandFull(turn.assistantReply)}
            />
            {turn.chatSnapshot?.recommended_department &&
              turn.chatSnapshot.oncall_appointments &&
              turn.chatSnapshot.oncall_appointments.length > 0 && (
                <AppointmentCards doctors={turn.chatSnapshot.oncall_appointments} />
              )}
            {isLatest && turn.choices?.type === 'clarify' && (
              <ClarifyChoices
                phase={turn.choices.phase}
                choices={turn.choices.items as ClarifyChoice[]}
                disabled={!canPick || busy}
                onPick={onPickChoice}
              />
            )}
            {isLatest && turn.choices?.type === 'dept' && (
              <DeptChoices
                choices={turn.choices.items as DeptChoice[]}
                multiSelect={turn.choices.multiSelect}
                disabled={!canPick || busy}
                onPick={onPickChoice}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
