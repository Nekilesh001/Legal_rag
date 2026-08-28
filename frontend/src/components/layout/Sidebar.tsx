import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FileText,
  FileSearch,
  ShieldAlert,
  MessageSquareCode,
  Settings,
  Scale,
  Sparkles,
} from 'lucide-react';
import { BRAND_CONFIG } from '../../config/brand';

export const Sidebar: React.FC = () => {
  const navItems = [
    {
      name: 'Dashboard',
      path: '/dashboard',
      icon: LayoutDashboard,
      isFunctional: false,
    },
    {
      name: 'Contracts',
      path: '/contracts',
      icon: FileText,
      isFunctional: false,
    },
    {
      name: 'Contract Review',
      path: '/review',
      icon: FileSearch,
      isFunctional: false,
    },
    {
      name: 'Risk Analysis',
      path: '/risk',
      icon: ShieldAlert,
      isFunctional: false,
    },
    {
      name: 'Legal Chat',
      path: '/legal-chat',
      icon: MessageSquareCode,
      isFunctional: true,
      badge: 'LIVE AI',
    },
    {
      name: 'Settings',
      path: '/settings',
      icon: Settings,
      isFunctional: false,
    },
  ];

  return (
    <aside className="w-64 h-screen border-r border-white/10 glass-panel flex flex-col justify-between sticky top-0 z-40 bg-[#090A0F]/90 backdrop-blur-xl shrink-0">
      <div>
        {/* Brand Header */}
        <div className="h-16 px-6 border-b border-white/10 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#5227FF] to-[#FF9FFC] p-0.5 shadow-lg shadow-[#5227FF]/20 flex items-center justify-center">
            <div className="w-full h-full bg-[#090A0F] rounded-[10px] flex items-center justify-center">
              <Scale className="w-5 h-5 text-[#FF9FFC]" />
            </div>
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-wider flex items-center gap-1.5">
              <span>{BRAND_CONFIG.appName}</span>
              <span className="text-[10px] font-mono font-normal px-1.5 py-0.5 rounded bg-[#5227FF]/20 text-[#FF9FFC]">AI</span>
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">Contract Intelligence</p>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="p-4 space-y-1.5">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-[#5227FF]/20 to-[#FF9FFC]/10 text-white border border-[#5227FF]/40 shadow-lg shadow-[#5227FF]/10 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <item.icon className="w-4 h-4" />
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-[#5227FF] text-white font-bold tracking-wider shadow-sm shadow-[#5227FF]/50 animate-pulse">
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* RAG Frozen Status Card in Sidebar Footer */}
      <div className="p-4 m-4 rounded-xl glass-card-accent border border-[#5227FF]/30 space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-white">
          <Sparkles className="w-3.5 h-3.5 text-[#FF9FFC]" />
          <span>Legal RAG Engine</span>
        </div>
        <p className="text-[11px] text-slate-300 leading-tight">
          Grounding across 52 Canonical Statutes & Acts with BGE-M3 & BGE-Reranker-v2-m3.
        </p>
        <div className="text-[10px] font-mono text-[#FF9FFC] flex items-center gap-1 pt-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          <span>100% Citation Validity Verified</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
