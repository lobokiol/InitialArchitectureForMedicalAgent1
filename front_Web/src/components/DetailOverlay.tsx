import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import { paginateArray } from '../lib/paginate';
import type { ChatResponse, RetrievedDoc } from '../types';

interface DetailOverlayProps {
  open: boolean;
  response: ChatResponse | null;
  fullText?: string;
  onClose: () => void;
}

type Tab = 'docs' | 'trace' | 'debug';

function collectDocs(response: ChatResponse): RetrievedDoc[] {
  const med = response.used_docs.medical.slice(0, 3);
  const proc = response.used_docs.process.slice(0, 3);
  return [...med, ...proc];
}

export function DetailOverlay({ open, response, fullText, onClose }: DetailOverlayProps) {
  const [tab, setTab] = useState<Tab>('docs');
  const [page, setPage] = useState(0);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    setPage(0);
  }, [tab, open]);

  const pages = useMemo(() => {
    if (!response && !fullText) return [null];
    if (fullText) {
      const chunks: string[] = [];
      for (let i = 0; i < fullText.length; i += 400) chunks.push(fullText.slice(i, i + 400));
      return chunks.map((t) => ({ kind: 'text' as const, text: t }));
    }
    if (!response) return [null];
    if (tab === 'docs') {
      return collectDocs(response).map((d) => ({ kind: 'doc' as const, doc: d }));
    }
    if (tab === 'trace') {
      return paginateArray(response.node_trace, 5).map((nodes) => ({
        kind: 'trace' as const,
        nodes,
      }));
    }
    const keys = Object.keys(response.app_state ?? {});
    return keys.length
      ? keys.map((k) => ({ kind: 'debug' as const, key: k, value: response.app_state![k] }))
      : [{ kind: 'debug' as const, key: '(空)', value: null }];
  }, [response, tab, fullText]);

  const total = pages.length;
  const current = pages[page];

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            className="w-full max-w-2xl h-[min(100vh,640px)] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <div className="flex gap-2">
                {!fullText &&
                  (['docs', 'trace', 'debug'] as Tab[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setTab(t)}
                      className={`px-3 py-1 rounded-lg text-sm ${
                        tab === t ? 'bg-brand-500 text-white' : 'text-gray-600 hover:bg-brand-50'
                      }`}
                    >
                      {t === 'docs' ? '参考文档' : t === 'trace' ? '节点轨迹' : '调试'}
                    </button>
                  ))}
                {fullText && <span className="text-sm font-medium text-brand-700">回复全文</span>}
              </div>
              <button type="button" onClick={onClose} className="p-1 rounded-lg hover:bg-gray-100">
                <X size={20} />
              </button>
            </div>

            <div className="flex-1 min-h-0 p-6 overflow-hidden flex flex-col justify-center">
              {current && 'kind' in current && current.kind === 'doc' && (
                <div>
                  <p className="text-xs text-brand-600 mb-1">
                    {current.doc.source} · score {current.doc.score?.toFixed(2) ?? '-'}
                  </p>
                  <h3 className="font-semibold text-gray-800 mb-2">{current.doc.title ?? current.doc.id}</h3>
                  <p className="text-sm text-gray-600 whitespace-pre-wrap line-clamp-[12]">{current.doc.content}</p>
                </div>
              )}
              {current && 'kind' in current && current.kind === 'trace' && (
                <ul className="space-y-2">
                  {current.nodes.map((n) => (
                    <li key={n} className="text-sm font-mono bg-brand-50 px-3 py-2 rounded-lg text-brand-800">
                      {n}
                    </li>
                  ))}
                </ul>
              )}
              {current && 'kind' in current && current.kind === 'debug' && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">{current.key}</p>
                  <pre className="text-xs bg-gray-50 p-4 rounded-lg overflow-hidden line-clamp-[16]">
                    {JSON.stringify(current.value, null, 2)}
                  </pre>
                </div>
              )}
              {current && 'kind' in current && current.kind === 'text' && (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{current.text}</p>
              )}
            </div>

            <div className="shrink-0 flex items-center justify-center gap-4 py-3 border-t">
              <button
                type="button"
                disabled={page <= 0}
                onClick={() => setPage((p) => p - 1)}
                className="p-2 rounded-lg hover:bg-brand-50 disabled:opacity-30"
              >
                <ChevronLeft size={20} />
              </button>
              <span className="text-sm text-gray-500 tabular-nums">
                {total === 0 ? '0/0' : `${page + 1}/${total}`}
              </span>
              <button
                type="button"
                disabled={page >= total - 1}
                onClick={() => setPage((p) => p + 1)}
                className="p-2 rounded-lg hover:bg-brand-50 disabled:opacity-30"
              >
                <ChevronRight size={20} />
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
