import type { IntentResult } from '../types';

interface IntentBadgeProps {
  intent?: IntentResult;
}

const intentLabels: Record<string, string> = {
  symptom: '症状',
  process: '流程',
  mixed: '混合',
  non_medical: '非医疗',
};

const routeLabels: Record<string, string> = {
  disease: '疾病链',
  symptom: '症状链',
  reject: '拒绝',
};

export function IntentBadge({ intent }: IntentBadgeProps) {
  if (!intent) return <span className="text-xs text-gray-400">无意图</span>;
  return (
    <span className="inline-flex items-center gap-1 text-xs">
      <span className="px-2 py-0.5 rounded-full bg-brand-100 text-brand-700 font-medium">
        {intentLabels[intent.main_intent] ?? intent.main_intent}
      </span>
      {intent.triage_route && (
        <span className="text-gray-500">路由: {routeLabels[intent.triage_route] ?? intent.triage_route}</span>
      )}
    </span>
  );
}
