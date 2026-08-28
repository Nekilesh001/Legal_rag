import React from 'react';
import { Sparkles, BookOpen, Clock, ShieldAlert, FileText, Globe } from 'lucide-react';
import GlassCard from '../common/GlassCard';

interface SuggestedQuestionsProps {
  onSelectQuestion: (query: string) => void;
}

export const SUGGESTED_QUESTIONS = [
  {
    category: 'Exact Legal Lookup',
    query: 'What does Section 73 of the Indian Contract Act say?',
    icon: BookOpen,
    description: 'Indian Contract Act 1872 — Compensation for loss or damage',
  },
  {
    category: 'Threshold & Statutory Notice',
    query: 'What is the notice period under the Tamil Nadu Shops Act?',
    icon: Clock,
    description: 'Section 41 — Notice period for employee discharge & dismissal',
  },
  {
    category: 'Broad Breach & Remedies',
    query: 'What happens if the seller breaches the contract?',
    icon: ShieldAlert,
    description: 'Sale of Goods Act 1930 — Remedies for breach of warranty & contract',
  },
  {
    category: 'Confidentiality & NDA',
    query: 'What are the mandatory clauses in an NDA agreement?',
    icon: FileText,
    description: 'NDA Core Provisions — Definition, Permitted Use, Non-Disclosure',
  },
  {
    category: 'Out-of-Corpus Abstention Test',
    query: 'What are the speed limit regulations under the Tokyo Traffic Law?',
    icon: Globe,
    description: 'Abstention Test — Validates grounding & zero hallucination behavior',
  },
];

export const SuggestedQuestions: React.FC<SuggestedQuestionsProps> = ({ onSelectQuestion }) => {
  return (
    <div className="max-w-3xl mx-auto space-y-6 py-6 animate-in fade-in duration-500">
      {/* Header Banner */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#5227FF]/15 border border-[#5227FF]/30 text-xs font-medium text-[#FF9FFC] shadow-lg shadow-[#5227FF]/10">
          <Sparkles className="w-4 h-4 text-[#FF9FFC]" />
          <span>Ask the Grounded Legal Knowledge Base</span>
        </div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          How can LegalIQ assist your contract research today?
        </h2>
        <p className="text-sm text-slate-400 max-w-xl mx-auto leading-relaxed">
          Query 52 canonical statutes, Indian contract rules, and NDA playbooks with 100% citation traceability. Select a benchmark query below or type your own.
        </p>
      </div>

      {/* Suggested Questions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
        {SUGGESTED_QUESTIONS.map((item, idx) => (
          <GlassCard
            key={idx}
            interactive
            onClick={() => onSelectQuestion(item.query)}
            className="p-4 flex flex-col justify-between group hover:border-[#5227FF]/50 transition-all duration-200"
          >
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#5227FF]/10 border border-[#5227FF]/20 flex items-center justify-center text-[#FF9FFC] group-hover:scale-105 transition-transform">
                <item.icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-300 font-medium block mb-1">
                  {item.category}
                </span>
                <p className="text-xs font-semibold text-slate-200 group-hover:text-white line-clamp-2 leading-snug">
                  "{item.query}"
                </p>
              </div>
            </div>
            <p className="text-[11px] text-slate-400 pt-3 border-t border-white/5 mt-3">
              {item.description}
            </p>
          </GlassCard>
        ))}
      </div>
    </div>
  );
};

export default SuggestedQuestions;
