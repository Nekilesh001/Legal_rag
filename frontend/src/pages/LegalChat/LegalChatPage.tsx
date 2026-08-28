import React, { useRef, useEffect } from 'react';
import { MessageSquareCode, Sparkles, RefreshCw, Zap, ShieldCheck } from 'lucide-react';
import useRagChat from '../../hooks/useRagChat';
import ChatMessage from '../../components/chat/ChatMessage';
import ChatInput from '../../components/chat/ChatInput';
import SuggestedQuestions from '../../components/chat/SuggestedQuestions';
import ChatHistorySidebar from '../../components/chat/ChatHistorySidebar';
import EvidenceDrawer from '../../components/evidence/EvidenceDrawer';

export const LegalChatPage: React.FC = () => {
  const {
    sessions,
    activeSessionId,
    messages,
    modelMode,
    setModelMode,
    isLoading,
    statusMessage,
    selectedCitation,
    activeCitationsList,
    sendMessage,
    createNewSession,
    switchSession,
    clearCurrentChat,
    resetSession,
    openCitationDrawer,
    closeCitationDrawer,
  } = useRagChat();

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, statusMessage]);

  return (
    <div className="h-[calc(100vh-6.5rem)] flex rounded-2xl overflow-hidden glass-panel border-white/10 shadow-2xl">
      {/* Saved Sessions History Sidebar */}
      <ChatHistorySidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={createNewSession}
        onSelectSession={switchSession}
      />

      {/* Main Chat Workarea */}
      <div className="flex-1 flex flex-col min-w-0 h-full bg-[#090A0F]/60">
        
        {/* Workspace Sub-header with Model Mode Toggle */}
        <div className="px-6 py-3 border-b border-white/10 glass-panel flex flex-wrap items-center justify-between gap-3 bg-black/40">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-[#5227FF]/15 border border-[#5227FF]/30 flex items-center justify-center text-[#FF9FFC]">
              <MessageSquareCode className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                <span>Legal RAG AI Assistant</span>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Grounded Corpus v1.0
                </span>
              </h2>
              <p className="text-[11px] text-slate-400">
                Direct statutory lookup & contract rule intelligence with SSE token streaming
              </p>
            </div>
          </div>

          {/* Model Mode Toggle (Fast vs Quality) */}
          <div className="flex items-center gap-3">
            <div className="flex items-center p-1 rounded-xl bg-white/5 border border-white/10">
              <button
                onClick={() => setModelMode('fast')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  modelMode === 'fast'
                    ? 'bg-amber-500 text-slate-950 font-semibold shadow-md shadow-amber-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Faster responses for interactive legal research"
              >
                <Zap className="w-3.5 h-3.5" />
                <span>Fast</span>
              </button>

              <button
                onClick={() => setModelMode('quality')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  modelMode === 'quality'
                    ? 'bg-[#5227FF] text-white font-semibold shadow-md shadow-[#5227FF]/30'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Deeper reasoning for complex legal questions"
              >
                <ShieldCheck className="w-3.5 h-3.5 text-[#FF9FFC]" />
                <span>Quality</span>
              </button>
            </div>
            <button
              onClick={resetSession}
              className="text-xs text-slate-300 hover:text-white px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all flex items-center gap-1.5"
              title="Start a new research session"
            >
              <RefreshCw className="w-3.5 h-3.5 text-[#FF9FFC]" />
              <span>New Session</span>
            </button>
          </div>
        </div>

        {/* Conversation Stream Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <SuggestedQuestions onSelectQuestion={(q) => sendMessage(q)} />
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto">
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  onOpenCitation={openCitationDrawer}
                  onRetry={(q) => sendMessage(q)}
                />
              ))}

              {/* Status Indicator & Pipeline Progress Spinner */}
              {isLoading && statusMessage && (
                <div className="flex items-start gap-3.5 max-w-2xl py-2 animate-in fade-in duration-300">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#5227FF] to-[#FF9FFC] p-0.5 shrink-0 shadow-lg shadow-[#5227FF]/20 animate-pulse">
                    <div className="w-full h-full bg-[#090A0F] rounded-[10px] flex items-center justify-center text-[#FF9FFC]">
                      <Sparkles className="w-4 h-4 animate-spin" />
                    </div>
                  </div>

                  <div className="glass-panel px-4 py-2.5 rounded-xl bg-[#0F111A]/80 border-white/10 flex items-center gap-2 text-xs font-mono text-[#FF9FFC]">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>{statusMessage}</span>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Sticky Input Footer */}
        <div className="p-4 border-t border-white/10 bg-[#090A0F]/90 backdrop-blur-xl">
          <div className="max-w-4xl mx-auto">
            <ChatInput
              onSendMessage={sendMessage}
              onClearHistory={clearCurrentChat}
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Slide-over Legal Evidence Source Panel Drawer */}
      <EvidenceDrawer
        isOpen={selectedCitation !== null}
        onClose={closeCitationDrawer}
        citations={activeCitationsList}
        initialCitationIndex={activeCitationsList.findIndex(
          (c) => c.citation_id === selectedCitation?.citation_id || c.chunk_id === selectedCitation?.chunk_id
        )}
      />
    </div>
  );
};

export default LegalChatPage;
