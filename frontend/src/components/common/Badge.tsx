import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'supported' | 'partially_supported' | 'insufficient' | 'high' | 'medium' | 'low' | 'neutral' | 'accent';
  size?: 'sm' | 'md';
  icon?: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  size = 'md',
  icon,
}) => {
  const variantStyles = {
    supported: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    partially_supported: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    insufficient: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    high: 'bg-[#5227FF]/15 text-indigo-300 border-[#5227FF]/40',
    medium: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    low: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
    neutral: 'bg-white/5 text-slate-300 border-white/10',
    accent: 'bg-[#FF9FFC]/10 text-[#FF9FFC] border-[#FF9FFC]/30',
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-[11px] font-medium tracking-wide rounded-md',
    md: 'px-2.5 py-1 text-xs font-medium tracking-wide rounded-lg',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 border font-mono uppercase ${sizeStyles[size]} ${variantStyles[variant]}`}
    >
      {icon}
      {children}
    </span>
  );
};

export default Badge;
