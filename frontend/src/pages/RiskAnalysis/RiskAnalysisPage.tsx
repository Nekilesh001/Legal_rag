import React from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, ArrowUpRight, BarChart3, PieChart, FileText } from 'lucide-react';
import GlassCard from '../../components/common/GlassCard';
import Badge from '../../components/common/Badge';
import { MOCK_RISK_FINDINGS } from '../../data/mock/risk';

export const RiskAnalysisPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-8">
      {/* Top Header Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Portfolio Risk Analysis & Statutory Audit</h2>
          <p className="text-xs text-slate-400">
            Comprehensive risk distribution across 148 active contract covenants
          </p>
        </div>

        <div className="flex items-center gap-3">
          <GlassCard className="py-2 px-4 flex items-center gap-3 bg-[#5227FF]/10 border-[#5227FF]/30">
            <span className="text-xs font-medium text-slate-300">Portfolio Risk Score:</span>
            <span className="text-lg font-bold text-[#FF9FFC] font-mono">34 / 100</span>
            <Badge variant="supported">LOW RISK</Badge>
          </GlassCard>
        </div>
      </div>

      {/* Risk Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <GlassCard className="p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-slate-400">High Risk Findings</span>
            <div className="text-2xl font-bold text-rose-400 mt-1">4</div>
            <span className="text-[11px] text-slate-500 font-mono">Immediate attention required</span>
          </div>
          <div className="p-3 rounded-xl bg-rose-500/10 text-rose-400">
            <ShieldAlert className="w-6 h-6" />
          </div>
        </GlassCard>

        <GlassCard className="p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-slate-400">Medium Risk Findings</span>
            <div className="text-2xl font-bold text-amber-400 mt-1">12</div>
            <span className="text-[11px] text-slate-500 font-mono">Renegotiation recommended</span>
          </div>
          <div className="p-3 rounded-xl bg-amber-500/10 text-amber-400">
            <AlertTriangle className="w-6 h-6" />
          </div>
        </GlassCard>

        <GlassCard className="p-5 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-slate-400">Low Risk / Compliant</span>
            <div className="text-2xl font-bold text-emerald-400 mt-1">132</div>
            <span className="text-[11px] text-slate-500 font-mono">Verified compliant covenants</span>
          </div>
          <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
        </GlassCard>
      </div>

      {/* Category Breakdown & Key Findings List */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Risk Breakdown Chart Placeholder (1 Col) */}
        <GlassCard className="p-5 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <PieChart className="w-4 h-4 text-[#5227FF]" />
            <span>Risk Category Breakdown</span>
          </h3>

          <div className="space-y-3 pt-2">
            {[
              { category: 'Breach / Statutory Remedy', count: 5, pct: '38%' },
              { category: 'Unilateral Termination', count: 4, pct: '30%' },
              { category: 'Employment Non-Compete', count: 2, pct: '18%' },
              { category: 'Lease Forfeiture', count: 2, pct: '14%' },
            ].map((item, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>{item.category}</span>
                  <span className="font-mono text-slate-400">{item.pct}</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#5227FF] to-[#FF9FFC] rounded-full"
                    style={{ width: item.pct }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Clause Findings & Statutory Recommendations List (2 Cols) */}
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <FileText className="w-4 h-4 text-[#FF9FFC]" />
            <span>Detailed Risk Findings & Statutory Recommendations</span>
          </h3>

          <div className="space-y-4">
            {MOCK_RISK_FINDINGS.map((rf) => (
              <GlassCard key={rf.id} className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">{rf.contractTitle}</span>
                    <h4 className="text-xs font-semibold text-white">{rf.clauseName}</h4>
                  </div>
                  <Badge variant={rf.severity === 'High' ? 'insufficient' : rf.severity === 'Medium' ? 'medium' : 'supported'}>
                    {rf.severity} SEVERITY
                  </Badge>
                </div>

                <p className="text-xs text-slate-300 italic bg-black/30 p-3 rounded-xl border border-white/5">
                  "{rf.excerpt}"
                </p>

                <div className="p-3 rounded-xl bg-[#5227FF]/10 border border-[#5227FF]/20 space-y-1 text-xs">
                  <span className="font-semibold text-indigo-200 block">STATUTORY RECOMMENDATION</span>
                  <p className="text-slate-300">{rf.recommendation}</p>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default RiskAnalysisPage;
