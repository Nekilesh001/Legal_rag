import React, { useState } from 'react';
import { Settings, Cpu, ShieldCheck, Bell, Database, Lock, Sliders } from 'lucide-react';
import GlassCard from '../../components/common/GlassCard';
import Button from '../../components/common/Button';
import Badge from '../../components/common/Badge';
import { BRAND_CONFIG } from '../../config/brand';

export const SettingsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'ai' | 'general' | 'workspace' | 'privacy'>('ai');

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-8">
      {/* Top Title */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Platform Settings & Governance</h2>
        <p className="text-xs text-slate-400">
          Configure legal RAG parameters, vector store settings, and workspace governance
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/10 pb-3">
        {[
          { id: 'ai', label: 'AI & RAG Engine', icon: Cpu },
          { id: 'general', label: 'General & Branding', icon: Sliders },
          { id: 'workspace', label: 'Workspace & Corpus', icon: Database },
          { id: 'privacy', label: 'Privacy & Security', icon: Lock },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition-all ${
              activeTab === tab.id
                ? 'bg-[#5227FF] text-white shadow-lg shadow-[#5227FF]/25 font-semibold'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <tab.icon className="w-3.5 h-3.5" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* RAG Engine Settings Section */}
      {activeTab === 'ai' && (
        <div className="space-y-6">
          <GlassCard className="p-6 space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div className="space-y-0.5">
                <h3 className="text-sm font-semibold text-white">Frozen RAG Model Pipeline</h3>
                <p className="text-xs text-slate-400">Current active model configurations (FROZEN ARCHITECTURE)</p>
              </div>
              <Badge variant="supported">ACTIVE & FROZEN</Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                <span className="text-slate-500 font-mono text-[10px] block uppercase">Dense Embeddings</span>
                <span className="text-white font-semibold block">BAAI / bge-m3</span>
                <span className="text-slate-400 text-[11px]">1024-dim dense vector embeddings</span>
              </div>

              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                <span className="text-slate-500 font-mono text-[10px] block uppercase">Contextual Reranker</span>
                <span className="text-white font-semibold block">BAAI / bge-reranker-v2-m3</span>
                <span className="text-slate-400 text-[11px]">Cross-encoder reranking top-50 window</span>
              </div>

              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                <span className="text-slate-500 font-mono text-[10px] block uppercase">Vector Store Engine</span>
                <span className="text-white font-semibold block">Qdrant Local Persistent</span>
                <span className="text-slate-400 text-[11px]">20,748 1-to-1 child vector points</span>
              </div>

              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                <span className="text-slate-500 font-mono text-[10px] block uppercase">LLM Generation Model</span>
                <span className="text-white font-semibold block">nvidia / nemotron-3-super-120b</span>
                <span className="text-slate-400 text-[11px]">Grounded legal synthesis</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#5227FF]/10 border border-[#5227FF]/20 text-xs text-indigo-200">
              <span className="font-semibold text-white block mb-1">Architecture Freeze Enforcement</span>
              Retrieval algorithms, chunking strategies, RRF weights, and reranking pipelines are locked to guarantee benchmark stability.
            </div>
          </GlassCard>
        </div>
      )}

      {/* General Settings Section */}
      {activeTab === 'general' && (
        <GlassCard className="p-6 space-y-4 text-xs">
          <h3 className="text-sm font-semibold text-white">Application Identity</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-slate-400 block mb-1">Application Name</label>
              <input
                type="text"
                readOnly
                value={BRAND_CONFIG.appName}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white"
              />
            </div>
            <div>
              <label className="text-slate-400 block mb-1">Version</label>
              <input
                type="text"
                readOnly
                value={BRAND_CONFIG.version}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white"
              />
            </div>
          </div>
        </GlassCard>
      )}

      {activeTab === 'workspace' && (
        <GlassCard className="p-6 text-xs text-slate-300">
          Workspace indexing configuration details for Indian Statutes, Contract Rules, and NDA Playbooks.
        </GlassCard>
      )}

      {activeTab === 'privacy' && (
        <GlassCard className="p-6 text-xs text-slate-300">
          Zero data retention policy enabled. Grounded legal evidence generated strictly from local indexed corpus.
        </GlassCard>
      )}
    </div>
  );
};

export default SettingsPage;
