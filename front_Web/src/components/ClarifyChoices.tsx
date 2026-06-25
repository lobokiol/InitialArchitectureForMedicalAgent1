import { motion } from 'framer-motion';
import type { ClarifyChoice } from '../types';

interface ClarifyChoicesProps {
  phase?: string;
  choices: ClarifyChoice[];
  disabled?: boolean;
  onPick: (label: string) => void;
}

export function ClarifyChoices({ phase, choices, disabled, onPick }: ClarifyChoicesProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-3 rounded-xl border border-brand-500/25 bg-brand-50/80 p-3"
    >
      <p className="text-xs font-medium text-brand-700 mb-2">澄清 · {phase ?? '信息'}</p>
      <div className="flex flex-wrap gap-2">
        {choices.map((c) => (
          <button
            key={c.id}
            type="button"
            disabled={disabled}
            onClick={() => onPick(c.label)}
            className="px-3 py-1.5 rounded-lg text-sm bg-white border border-brand-500/30 hover:border-brand-500 hover:bg-brand-100 disabled:opacity-50 transition-colors"
          >
            {c.label}
          </button>
        ))}
      </div>
    </motion.div>
  );
}
