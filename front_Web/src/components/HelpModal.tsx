import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';

interface HelpModalProps {
  open: boolean;
  onClose: () => void;
}

const pages = [
  {
    title: '项目简介',
    body: `医院导诊 Agentic 助手 — 基于 FastAPI + LangGraph 的多轮导诊系统。

• OpenSearch 混合检索（rag_knowledge / disease_kb / rag_department_rules）
• Redis Checkpoint 持久化会话
• DashScope LLM 生成回复

本 Demo 页复刻 CLI 多轮澄清与科室鉴别能力，所有业务规则在后端 LangGraph。`,
  },
  {
    title: '架构简图',
    body: `Web Demo → POST /chat → chat_service
         → LangGraph StateGraph（14 节点）
         → OpenSearch + DashScope LLM
         → ChatResponse

症状链：混合召回 → symptom_clarify → dept_rules_disambiguation
      → dept_disambiguation → dept_confidence → 回复`,
  },
];

export function HelpModal({ open, onClose }: HelpModalProps) {
  const [page, setPage] = useState(0);

  if (!open) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={onClose}>
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-lg bg-white rounded-2xl shadow-xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between items-center px-4 py-3 border-b">
            <h2 className="font-semibold text-brand-700">{pages[page].title}</h2>
            <button type="button" onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
              <X size={18} />
            </button>
          </div>
          <div className="p-6 min-h-[200px]">
            <p className="text-sm text-gray-600 whitespace-pre-wrap leading-relaxed">{pages[page].body}</p>
          </div>
          <div className="flex justify-center items-center gap-4 py-3 border-t">
            <button
              type="button"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
              className="p-2 rounded-lg hover:bg-brand-50 disabled:opacity-30"
            >
              <ChevronLeft size={18} />
            </button>
            <span className="text-xs text-gray-500">
              {page + 1}/{pages.length}
            </span>
            <button
              type="button"
              disabled={page >= pages.length - 1}
              onClick={() => setPage((p) => p + 1)}
              className="p-2 rounded-lg hover:bg-brand-50 disabled:opacity-30"
            >
              <ChevronRight size={18} />
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
