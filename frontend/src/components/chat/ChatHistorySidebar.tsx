import React from 'react';
import { Plus, MessageSquare, BookOpen, Clock } from 'lucide-react';
import { ChatSession } from '../../types/rag';
import Button from '../common/Button';

interface ChatHistorySidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
}

export const ChatHistorySidebar: React.FC<ChatHistorySidebarProps> = ({
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
}) => {
  return (
    <div className="w-64 glass-panel border-r border-white/10 p-4 flex flex-col justify-between hidden md:flex shrink-0 bg-[#0C0E17]/80">
      <div className="space-y-4">
        <Button
          onClick={onNewChat}
          variant="secondary"
          className="w-full justify-start text-xs font-semibold py-2.5"
          leftIcon={<Plus className="w-4 h-4 text-[#FF9FFC]" />}
        >
          New Legal Research Session
        </Button>

        <div className="space-y-2">
          <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-500 block px-1">
            Recent Research Sessions
          </span>

          <div className="space-y-1 max-h-[calc(100vh-22rem)] overflow-y-auto pr-1">
            {sessions.length === 0 ? (
              <div className="p-4 text-center text-xs text-slate-500 font-mono">
                No recent research sessions
              </div>
            ) : (
              sessions.map((sess) => {
                const isActive = sess.id === activeSessionId;
                const formattedTime = new Date(sess.updatedAt).toLocaleDateString(undefined, {
                  month: 'short',
                  day: 'numeric',
                });

                return (
                  <button
                    key={sess.id}
                    onClick={() => onSelectSession(sess.id)}
                    className={`w-full text-left p-2.5 rounded-xl border transition-all text-xs group ${
                      isActive
                        ? 'bg-[#5227FF]/15 border-[#5227FF]/40 text-white font-medium shadow-sm shadow-[#5227FF]/10'
                        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-white/5'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-[#FF9FFC]' : 'text-slate-500'}`} />
                      <span className="truncate">{sess.title}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono pl-5 block pt-0.5">
                      {formattedTime} · {sess.messages.length} msgs
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      </div>

      <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 text-[11px] text-slate-400 space-y-1">
        <div className="flex items-center gap-1.5 text-slate-200 font-medium">
          <BookOpen className="w-3.5 h-3.5 text-[#5227FF]" />
          <span>Statutory Knowledge Base</span>
        </div>
        <p className="text-[10px] leading-relaxed">
          Grounded over Indian Contract Act, Sale of Goods Act, TPA, CGST, and TN Shops Act.
        </p>
      </div>
    </div>
  );
};

export default ChatHistorySidebar;
