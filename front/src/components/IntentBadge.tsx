import { motion } from 'framer-motion';
import { Brain, FileText, Stethoscope, MessageCircle } from 'lucide-react';
import type { IntentResult, UsedDocs } from '../types';

interface IntentBadgeProps {
  intent?: IntentResult;
  usedDocs?: UsedDocs;
}

const intentConfig = {
  symptom: {
    icon: Stethoscope,
    label: '症状问诊',
    color: 'bg-red-100 text-red-700 border-red-200',
  },
  process: {
    icon: FileText,
    label: '流程咨询',
    color: 'bg-blue-100 text-blue-700 border-blue-200',
  },
  mixed: {
    icon: Brain,
    label: '混合意图',
    color: 'bg-purple-100 text-purple-700 border-purple-200',
  },
  non_medical: {
    icon: MessageCircle,
    label: '普通问答',
    color: 'bg-gray-100 text-gray-700 border-gray-200',
  },
};

export function IntentBadge({ intent, usedDocs }: IntentBadgeProps) {
  if (!intent) return null;

  const config = intentConfig[intent.intent] || intentConfig.non_medical;
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-2 space-y-2"
    >
      <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-sm ${config.color}`}>
        <Icon className="w-3.5 h-3.5" />
        <span>{config.label}</span>
        {intent.confidence > 0 && (
          <span className="text-xs opacity-70">
            ({Math.round(intent.confidence * 100)}%)
          </span>
        )}
      </div>

      {usedDocs && (usedDocs.medical.length > 0 || usedDocs.process.length > 0) && (
        <div className="text-xs text-gray-500 bg-gray-50 rounded-lg p-2 space-y-1">
          <div className="font-medium text-gray-600">📚 参考文档:</div>
          {usedDocs.medical.length > 0 && (
            <div className="pl-2">
              <span className="text-red-500">🏥 症状库:</span>
              {usedDocs.medical.slice(0, 2).map((doc, i) => (
                <div key={i} className="truncate">
                  • {doc.title}
                  {doc.score && <span className="text-gray-400 ml-1">({Math.round(doc.score * 100)}%)</span>}
                </div>
              ))}
            </div>
          )}
          {usedDocs.process.length > 0 && (
            <div className="pl-2">
              <span className="text-blue-500">📋 流程文档:</span>
              {usedDocs.process.slice(0, 2).map((doc, i) => (
                <div key={i} className="truncate">
                  • {doc.title}
                  {doc.score && <span className="text-gray-400 ml-1">({Math.round(doc.score * 100)}%)</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
