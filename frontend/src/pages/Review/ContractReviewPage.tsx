import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileSearch, ShieldAlert, CheckCircle, MessageSquareCode, FileText, ChevronRight, Sparkles, BookOpen } from 'lucide-react';
import GlassCard from '../../components/common/GlassCard';
import Button from '../../components/common/Button';
import Badge from '../../components/common/Badge';
import { MOCK_CONTRACTS } from '../../data/mock/contracts';

export const ContractReviewPage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedContract, setSelectedContract] = useState(MOCK_CONTRACTS[0]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-8">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Contract Clause Review Workspace</h2>
          <p className="text-xs text-slate-400">
            Interactive clause-level risk identification & statutory compliance preview
          </p>
        </div>

        <Button
          onClick={() => navigate('/legal-chat')}
          variant="accent"
          leftIcon={<MessageSquareCode className="w-4 h-4" />}
        >
          Ask RAG AI About This Contract
        </Button>
      </div>

      {/* 3-Column Review Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Selected Contract Selector (3 Cols) */}
        <div className="lg:col-span-3 space-y-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 block px-1">
            Contract Document List
          </span>

          <div className="space-y-2">
            {MOCK_CONTRACTS.map((c) => (
              <GlassCard
                key={c.id}
                interactive
                onClick={() => setSelectedContract(c)}
                className={`p-3.5 ${selectedContract.id === c.id ? 'border-[#5227FF] bg-[#5227FF]/10' : ''}`}
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-1 min-w-0">
                    <h4 className="text-xs font-semibold text-white truncate">{c.title}</h4>
                    <span className="text-[10px] text-slate-400 font-mono block">{c.type}</span>
                  </div>
                  <Badge variant={c.riskLevel === 'High' ? 'insufficient' : c.riskLevel === 'Medium' ? 'medium' : 'supported'} size="sm">
                    {c.riskScore}
                  </Badge>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>

        {/* Center Column: Document Viewer Mock (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <GlassCard className="p-5 space-y-4 bg-[#0B0D16]/90 min-h-[500px]">
            <div className="pb-3 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-[#5227FF]" />
                <h3 className="text-xs font-semibold text-white truncate max-w-[240px]">
                  {selectedContract.title}
                </h3>
              </div>
              <span className="text-[10px] font-mono text-slate-400">Page 1 of 12</span>
            </div>

            {/* Document Text Mock Preview */}
            <div className="space-y-4 text-xs leading-relaxed text-slate-300 font-sans">
              <p className="font-semibold text-slate-100">
                MASTER ENTERPRISE AGREEMENT (Preview Mode)
              </p>
              <p>
                This Master Services Agreement ("Agreement") is entered into as of the Effective Date by and between Apex Cloud Solutions Private Limited ("Licensor") and Customer ("Licensee").
              </p>

              {/* Highlighted Risk Clause */}
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 space-y-1">
                <span className="text-[10px] font-mono text-rose-300 font-bold uppercase block">
                  FLAGGED CLAUSE 14.2 — INDEMNIFICATION CAP
                </span>
                <p className="text-rose-100 italic">
                  "Vendor liability shall not exceed INR 50,000 under any circumstances, notwithstanding statutory warranties under the Sale of Goods Act, 1930 or Indian Contract Act."
                </p>
              </div>

              <p>
                Clause 15.1 — Governing Law. This Agreement shall be governed by and construed in accordance with the laws of India, subject to exclusive jurisdiction of courts in New Delhi.
              </p>
            </div>
          </GlassCard>
        </div>

        {/* Right Column: Clause Analysis & Statutory Findings (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 block px-1">
            Clause Risk Findings
          </span>

          <GlassCard className="p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white">Risk Finding #1</span>
              <Badge variant="insufficient">High Risk (84)</Badge>
            </div>

            <div className="space-y-2">
              <span className="text-[11px] font-mono text-slate-400 block">STATUTORY CONFLICT</span>
              <p className="text-xs text-slate-200 leading-snug">
                Disproportionate liability cap conflicts with Section 59 remedies for breach of warranty under the Sale of Goods Act, 1930.
              </p>
            </div>

            <div className="p-3 rounded-xl bg-[#5227FF]/15 border border-[#5227FF]/30 space-y-1">
              <span className="text-[10px] font-mono text-[#FF9FFC] font-semibold block">RECOMMENDED REVISION</span>
              <p className="text-xs text-indigo-100">
                Replace flat cap with 2x annual contract value floor and carve out indemnification for statutory breaches.
              </p>
            </div>

            {/* Template Notice */}
            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 text-[11px] text-slate-400 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[#FF9FFC] shrink-0" />
              <span>Clause analysis UI preview. Use Legal Chat for grounded RAG queries.</span>
            </div>
          </GlassCard>
        </div>

      </div>
    </div>
  );
};

export default ContractReviewPage;
