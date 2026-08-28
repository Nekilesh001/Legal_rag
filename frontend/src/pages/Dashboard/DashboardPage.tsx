import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  ShieldAlert,
  CheckCircle2,
  Clock,
  ArrowUpRight,
  Sparkles,
  MessageSquareCode,
  TrendingUp,
  FileCheck,
} from 'lucide-react';
import GlassCard from '../../components/common/GlassCard';
import Button from '../../components/common/Button';
import Badge from '../../components/common/Badge';
import { MOCK_DASHBOARD_METRICS, MOCK_RECENT_ACTIVITIES } from '../../data/mock/dashboard';
import { MOCK_CONTRACTS } from '../../data/mock/contracts';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  const metricsCards = [
    {
      title: 'Contracts In Portfolio',
      value: MOCK_DASHBOARD_METRICS.totalContracts,
      change: '+12% this month',
      icon: FileText,
      color: 'text-indigo-400',
    },
    {
      title: 'Contracts Reviewed',
      value: MOCK_DASHBOARD_METRICS.reviewedContracts,
      change: '89.2% coverage',
      icon: CheckCircle2,
      color: 'text-emerald-400',
    },
    {
      title: 'High Risk Flagged',
      value: MOCK_DASHBOARD_METRICS.highRiskContracts,
      change: '4 action required',
      icon: ShieldAlert,
      color: 'text-rose-400',
    },
    {
      title: 'Pending Review',
      value: MOCK_DASHBOARD_METRICS.pendingReviewCount,
      change: 'Avg 24h turnaround',
      icon: Clock,
      color: 'text-amber-400',
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-8">
      {/* Top Welcome Banner */}
      <div className="glass-card-accent p-6 rounded-2xl border border-[#5227FF]/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl">
        <div className="space-y-1.5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#5227FF]/20 text-xs font-mono text-[#FF9FFC]">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Frozen Legal RAG Pipeline Ready</span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Legal Contract Intelligence Dashboard
          </h2>
          <p className="text-xs text-slate-300 max-w-xl">
            Real-time portfolio visibility, statutory compliance monitoring, and instant legal AI grounded research.
          </p>
        </div>

        <Button
          onClick={() => navigate('/legal-chat')}
          variant="accent"
          leftIcon={<MessageSquareCode className="w-4 h-4" />}
          rightIcon={<ArrowUpRight className="w-4 h-4" />}
        >
          Launch Functional Legal AI Chat
        </Button>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricsCards.map((m, idx) => (
          <GlassCard key={idx} className="p-5 flex flex-col justify-between space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">{m.title}</span>
              <div className={`p-2 rounded-xl bg-white/5 ${m.color}`}>
                <m.icon className="w-5 h-5" />
              </div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white tracking-tight">{m.value}</div>
              <span className="text-[11px] text-slate-400 font-mono">{m.change}</span>
            </div>
          </GlassCard>
        ))}
      </div>

      {/* Main Content Split Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Recent Contracts Table Widget (2 Cols) */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-[#5227FF]" />
              <span>Recent Contracts</span>
            </h3>
            <button
              onClick={() => navigate('/contracts')}
              className="text-xs text-[#FF9FFC] hover:underline flex items-center gap-1 font-medium"
            >
              <span>View All Repository</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <GlassCard className="p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-white/[0.03] text-slate-400 font-mono uppercase text-[10px] border-b border-white/10">
                  <tr>
                    <th className="p-4">Contract Title</th>
                    <th className="p-4">Type</th>
                    <th className="p-4">Counterparty</th>
                    <th className="p-4">Risk Level</th>
                    <th className="p-4 text-right">Updated</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {MOCK_CONTRACTS.slice(0, 5).map((c) => (
                    <tr key={c.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="p-4 font-medium text-white max-w-[200px] truncate">{c.title}</td>
                      <td className="p-4 text-slate-300">{c.type}</td>
                      <td className="p-4 text-slate-400 max-w-[160px] truncate">{c.counterparty}</td>
                      <td className="p-4">
                        <Badge variant={c.riskLevel === 'High' ? 'insufficient' : c.riskLevel === 'Medium' ? 'medium' : 'supported'}>
                          {c.riskLevel} ({c.riskScore})
                        </Badge>
                      </td>
                      <td className="p-4 text-right font-mono text-slate-400">{c.updatedAt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>

        {/* Recent Platform Activity & Risk Summary */}
        <div className="space-y-4">
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[#FF9FFC]" />
            <span>Platform Activity</span>
          </h3>

          <GlassCard className="p-5 space-y-4">
            <div className="space-y-4">
              {MOCK_RECENT_ACTIVITIES.map((act) => (
                <div key={act.id} className="flex items-start gap-3 pb-3 border-b border-white/5 last:border-0 last:pb-0">
                  <div className="w-2 h-2 rounded-full bg-[#5227FF] mt-1.5 shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-xs font-semibold text-slate-200 truncate">{act.title}</h4>
                    <p className="text-[11px] text-slate-400 leading-snug">{act.subtitle}</p>
                    <span className="text-[10px] text-slate-500 font-mono block pt-1">{act.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="p-3 rounded-xl bg-[#5227FF]/10 border border-[#5227FF]/20 text-xs text-indigo-200">
              <span className="font-semibold text-white block mb-1">RAG Grounded Research Active</span>
              Ask any question on Section 73, TN Shops Act, or Breach remedies via Legal Chat.
            </div>
          </GlassCard>
        </div>

      </div>
    </div>
  );
};

export default DashboardPage;
