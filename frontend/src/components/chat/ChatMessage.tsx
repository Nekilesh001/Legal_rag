import React, { useState } from 'react';
import { User, Scale, ShieldCheck, ShieldAlert, BookOpen, Copy, Check, ExternalLink, AlertTriangle, RefreshCw, FileText } from 'lucide-react';
import { ChatMessage as ChatMessageType, Citation } from '../../types/rag';
import Badge from '../common/Badge';

interface ChatMessageProps {
  message: ChatMessageType;
  onOpenCitation: (citation: Citation, allCitations: Citation[]) => void;
  onRetry?: (query: string) => void;
}

function cleanDisplayAnswer(content: string, responseAnswer?: string): string {
  const target = responseAnswer || content || '';
  if (!target) return '';
  const trimmed = target.trim();

  // If raw JSON string is present, parse or extract "answer" key
  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed.answer === 'string') {
        return parsed.answer;
      }
    } catch {
      const match = trimmed.match(/"answer"\s*:\s*"([\s\S]*?)"(?:\s*,\s*"|\s*})/);
      if (match && match[1]) {
        return match[1]
          .replace(/\\n/g, '\n')
          .replace(/\\"/g, '"')
          .replace(/\\\\/g, '\\');
      }
    }
  }

  return target;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onOpenCitation, onRetry }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const displayContent = cleanDisplayAnswer(message.content, message.response?.answer);

  const handleCopy = () => {
    navigator.clipboard.writeText(displayContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-3 max-w-3xl ml-auto py-2 animate-in fade-in duration-300">
        <div className="glass-card-accent p-4 rounded-2xl rounded-tr-sm text-sm text-white max-w-xl shadow-lg shadow-[#5227FF]/10 leading-relaxed border border-[#5227FF]/40">
          <p>{message.content}</p>
          <span className="text-[10px] text-[#FF9FFC]/70 block text-right pt-1.5 font-mono">
            {message.timestamp}
          </span>
        </div>
        <div className="w-8 h-8 rounded-xl bg-[#5227FF] flex items-center justify-center text-white shrink-0 shadow-md shadow-[#5227FF]/30">
          <User className="w-4 h-4" />
        </div>
      </div>
    );
  }

  // Error State Display
  if (message.isError) {
    return (
      <div className="flex items-start gap-3.5 max-w-3xl py-3 animate-in fade-in duration-300">
        <div className="w-9 h-9 rounded-xl bg-rose-500/20 border border-rose-500/30 flex items-center justify-center text-rose-400 shrink-0">
          <AlertTriangle className="w-5 h-5" />
        </div>

        <div className="glass-panel p-5 rounded-2xl rounded-tl-sm space-y-3 bg-rose-950/20 border-rose-500/30 shadow-xl flex-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-rose-300 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" />
              Service Interrupted
            </span>
            <span className="text-[10px] text-slate-500 font-mono">{message.timestamp}</span>
          </div>

          <p className="text-xs text-rose-200 leading-relaxed font-sans">
            Unable to complete this legal query. The legal knowledge service encountered an internal error. Please try again.
          </p>

          {message.failedQuery && onRetry && (
            <div className="pt-2 flex justify-end">
              <button
                onClick={() => onRetry(message.failedQuery!)}
                className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 text-xs font-medium border border-rose-500/40 flex items-center gap-1.5 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Query</span>
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  const response = message.response;
  const citationsList = response?.citations || [];
  const isInsufficient = response?.evidence_status === 'insufficient' || response?.evidence_status === 'out_of_scope';

  return (
    <div className="flex items-start gap-3.5 max-w-4xl py-3 animate-in fade-in duration-300">
      <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#5227FF] to-[#FF9FFC] p-0.5 shrink-0 shadow-lg shadow-[#5227FF]/20">
        <div className="w-full h-full bg-[#090A0F] rounded-[10px] flex items-center justify-center text-[#FF9FFC]">
          <Scale className="w-5 h-5" />
        </div>
      </div>

      <div className="flex-1 space-y-4">
        <div className="glass-panel p-5 rounded-2xl rounded-tl-sm space-y-4 bg-[#0F111A]/80 border-white/10 shadow-xl">
          
          {/* Status & Confidence Header Badges */}
          {response && (
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <Badge
                  variant={response.evidence_status === 'supported' ? 'supported' : isInsufficient ? 'insufficient' : 'partially_supported'}
                  icon={isInsufficient ? <ShieldAlert className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                >
                  {response.evidence_status.replace('_', ' ').toUpperCase()}
                </Badge>

                <Badge variant={response.confidence === 'high' ? 'high' : response.confidence === 'medium' ? 'medium' : 'low'}>
                  CONFIDENCE: {response.confidence.toUpperCase()}
                </Badge>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1 px-2.5 py-1 rounded-lg hover:bg-white/5 transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            </div>
          )}

          {/* Insufficient Evidence Warning Banner */}
          {isInsufficient ? (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-200 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-sm text-rose-300">
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                <span>Insufficient Knowledge Base Support</span>
              </div>
              <p className="text-xs leading-relaxed">
                The provided corpus does not contain sufficient information to answer this query.
              </p>
            </div>
          ) : null}

          {/* Visually Dominant Primary Legal Answer Content */}
          <div className="text-sm text-slate-200 leading-relaxed font-sans space-y-3 whitespace-pre-wrap selection:bg-[#5227FF]">
            {displayContent}
          </div>

          {/* Grounded Citations & Sources Section (Count equals actual citations array length) */}
          {citationsList.length > 0 && (
            <div className="pt-4 border-t border-white/10 space-y-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5 text-[#5227FF]" />
                Grounded Legal Citations ({citationsList.length})
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {citationsList.map((cit, idx) => (
                  <button
                    key={cit.citation_id || idx}
                    onClick={() => onOpenCitation(cit, citationsList)}
                    className="group p-3 rounded-xl bg-white/5 hover:bg-[#5227FF]/20 border border-white/10 hover:border-[#5227FF]/50 text-left transition-all duration-200 flex flex-col justify-between gap-1.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#5227FF]/20 text-[#FF9FFC]">
                          [{idx + 1}]
                        </span>
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-white line-clamp-1">
                          {cit.document_title || cit.document || 'Legal Act'}
                        </span>
                      </div>
                      <ExternalLink className="w-3 h-3 text-slate-500 group-hover:text-[#FF9FFC] shrink-0 transition-colors" />
                    </div>

                    <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400 group-hover:text-indigo-200">
                      {cit.section && <span>{cit.section}</span>}
                      {cit.page !== undefined && cit.page !== null && (
                        <span>Page {cit.page}</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <span className="text-[10px] text-slate-500 font-mono pl-1">{message.timestamp}</span>
      </div>
    </div>
  );
};

export default ChatMessage;
