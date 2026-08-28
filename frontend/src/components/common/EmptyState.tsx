import React from 'react';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className = '',
}) => {
  return (
    <div className={`flex flex-col items-center justify-center text-center p-8 glass-panel rounded-2xl ${className}`}>
      {icon && (
        <div className="w-14 h-14 rounded-2xl bg-[#5227FF]/10 border border-[#5227FF]/20 flex items-center justify-center text-[#FF9FFC] mb-4 shadow-lg shadow-[#5227FF]/10">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-sm text-slate-400 max-w-md mb-6 leading-relaxed">{description}</p>
      {action}
    </div>
  );
};

export default EmptyState;
