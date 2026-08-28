import React from 'react';
import { Search, Bell, ShieldCheck, User, Sparkles } from 'lucide-react';
import { BRAND_CONFIG } from '../../config/brand';

interface HeaderProps {
  pageTitle: string;
  onOpenSearch?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ pageTitle, onOpenSearch }) => {
  return (
    <header className="h-16 border-b border-white/10 glass-panel sticky top-0 z-30 flex items-center justify-between px-6 bg-[#090A0F]/80 backdrop-blur-xl">
      {/* Left Title & Status */}
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-white tracking-tight">{pageTitle}</h1>
        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#5227FF]/10 border border-[#5227FF]/25 text-[11px] font-medium text-indigo-300">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Frozen RAG Engine v1.0 Active</span>
        </div>
      </div>

      {/* Right Search & Profile Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenSearch}
          className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <Search className="w-3.5 h-3.5 text-slate-400" />
          <span>Quick search statutes & contracts...</span>
          <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-[10px] text-slate-400">⌘K</kbd>
        </button>

        <button className="p-2 text-slate-400 hover:text-white rounded-xl hover:bg-white/5 transition-colors relative">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#FF9FFC]"></span>
        </button>

        <div className="h-6 w-px bg-white/10 mx-1"></div>

        {/* User Profile Placeholder */}
        <div className="flex items-center gap-3 pl-1">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#5227FF] to-[#FF9FFC] p-[1px]">
            <div className="w-full h-full rounded-[11px] bg-[#090A0F] flex items-center justify-center text-xs font-semibold text-white">
              JD
            </div>
          </div>
          <div className="hidden lg:block text-left text-xs">
            <div className="font-medium text-slate-200">Legal Counsel</div>
            <div className="text-[10px] text-slate-400">Enterprise Counsel</div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
