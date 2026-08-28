import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  interactive?: boolean;
  accent?: boolean;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  interactive = false,
  accent = false,
  onClick,
}) => {
  const baseClass = accent
    ? 'glass-card-accent'
    : interactive
    ? 'glass-panel-interactive'
    : 'glass-panel';

  return (
    <div
      onClick={onClick}
      className={`rounded-xl p-5 ${baseClass} ${interactive ? 'cursor-pointer' : ''} ${className}`}
    >
      {children}
    </div>
  );
};

export default GlassCard;
