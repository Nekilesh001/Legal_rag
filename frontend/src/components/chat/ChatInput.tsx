import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Sparkles, CornerDownLeft } from 'lucide-react';
import Button from '../common/Button';

interface ChatInputProps {
  onSendMessage: (query: string) => void;
  onClearHistory: () => void;
  isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onClearHistory,
  isLoading,
}) => {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [query]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSendMessage(query.trim());
    setQuery('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative space-y-2">
      <div className="glass-panel p-2.5 rounded-2xl border-white/10 bg-[#0F111A]/90 focus-within:border-[#5227FF]/60 focus-within:ring-2 focus-within:ring-[#5227FF]/20 transition-all shadow-2xl">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask any question about statutes, acts, breach remedies, NDA clauses..."
          rows={1}
          disabled={isLoading}
          className="w-full bg-transparent border-none text-slate-100 placeholder-slate-500 text-sm focus:outline-none resize-none px-3 py-2 leading-relaxed"
        />

        <div className="flex items-center justify-between pt-2 border-t border-white/5 px-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClearHistory}
              className="text-xs text-slate-500 hover:text-rose-400 p-1.5 rounded-lg hover:bg-white/5 transition-colors flex items-center gap-1"
              title="Clear current conversation"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear Chat</span>
            </button>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-[11px] text-slate-500 font-mono flex items-center gap-1">
              <span>Press</span>
              <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-slate-300">↵</kbd>
              <span>to send</span>
            </span>

            <Button
              type="submit"
              size="sm"
              variant="accent"
              isLoading={isLoading}
              disabled={!query.trim() || isLoading}
              rightIcon={<Send className="w-3.5 h-3.5" />}
            >
              Send Query
            </Button>
          </div>
        </div>
      </div>
    </form>
  );
};

export default ChatInput;
