import ReactMarkdown from 'react-markdown';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  loading?: boolean;
  onExpandFull?: () => void;
}

export function MessageBubble({ role, content, loading, onExpandFull }: MessageBubbleProps) {
  const isUser = role === 'user';
  const showExpand = !isUser && content.length > 200;

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? 'bg-brand-500 text-white rounded-br-md'
            : 'bg-white border border-brand-500/15 text-gray-800 rounded-bl-md shadow-sm'
        }`}
      >
        {loading ? (
          <div className="flex gap-1 py-1">
            <span className="w-2 h-2 rounded-full bg-brand-500/40 animate-pulse" />
            <span className="w-2 h-2 rounded-full bg-brand-500/40 animate-pulse [animation-delay:150ms]" />
            <span className="w-2 h-2 rounded-full bg-brand-500/40 animate-pulse [animation-delay:300ms]" />
          </div>
        ) : isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-1 line-clamp-6">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
        {showExpand && onExpandFull && (
          <button
            type="button"
            onClick={onExpandFull}
            className="mt-2 text-xs text-brand-600 hover:underline"
          >
            全文
          </button>
        )}
      </div>
    </div>
  );
}
