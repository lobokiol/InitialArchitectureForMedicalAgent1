import { useState } from 'react';
import { motion } from 'framer-motion';
import { formatDeptSelection } from '../lib/chatLoop';
import type { DeptChoice } from '../types';

interface DeptChoicesProps {
  choices: DeptChoice[];
  multiSelect?: boolean;
  disabled?: boolean;
  onPick: (message: string) => void;
}

export function DeptChoices({ choices, multiSelect, disabled, onPick }: DeptChoicesProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const toggle = (index: number) => {
    if (!multiSelect) {
      onPick(choices[index].label);
      return;
    }
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const confirmMulti = () => {
    const indices = [...selected].sort((a, b) => a - b);
    if (indices.length === 0) return;
    onPick(formatDeptSelection(indices));
    setSelected(new Set());
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-3 rounded-xl border border-brand-500/25 bg-brand-50/80 p-3"
    >
      <p className="text-xs font-medium text-brand-700 mb-2">
        科室鉴别 · {multiSelect ? '可多选' : '单选'}
      </p>
      <div className="flex flex-wrap gap-2">
        {choices.map((c, i) => {
          const active = multiSelect && selected.has(i);
          return (
            <button
              key={c.id}
              type="button"
              disabled={disabled}
              onClick={() => toggle(i)}
              className={`px-3 py-1.5 rounded-lg text-sm border transition-colors disabled:opacity-50 ${
                active
                  ? 'bg-brand-500 text-white border-brand-500'
                  : 'bg-white border-brand-500/30 hover:border-brand-500 hover:bg-brand-100'
              }`}
            >
              {multiSelect && (active ? '✓ ' : '')}
              {c.label}
            </button>
          );
        })}
      </div>
      {multiSelect && (
        <button
          type="button"
          disabled={disabled || selected.size === 0}
          onClick={confirmMulti}
          className="mt-2 px-4 py-1.5 rounded-lg text-sm bg-brand-600 text-white disabled:opacity-50 hover:bg-brand-700"
        >
          确认选择
        </button>
      )}
    </motion.div>
  );
}
