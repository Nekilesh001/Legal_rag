import React from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'accent' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  className = '',
  disabled,
  ...props
}) => {
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs font-medium rounded-lg',
    md: 'px-4 py-2 text-sm font-medium rounded-xl',
    lg: 'px-6 py-3 text-base font-semibold rounded-xl',
  };

  const variantClasses = {
    primary:
      'bg-[#5227FF] hover:bg-[#401BE6] text-white shadow-lg shadow-[#5227FF]/25 border border-[#5227FF]/50 transition-all duration-200 active:scale-[0.98]',
    accent:
      'bg-gradient-to-r from-[#5227FF] to-[#FF9FFC] hover:opacity-95 text-white font-semibold shadow-lg shadow-[#FF9FFC]/20 transition-all duration-200 active:scale-[0.98]',
    secondary:
      'bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 transition-all duration-200 active:scale-[0.98]',
    ghost:
      'bg-transparent hover:bg-white/5 text-slate-400 hover:text-white transition-all duration-200',
    danger:
      'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-all duration-200',
  };

  return (
    <button
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed select-none ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {isLoading ? <Loader2 className="w-4 h-4 animate-spin text-current" /> : leftIcon}
      <span>{children}</span>
      {!isLoading && rightIcon}
    </button>
  );
};

export default Button;
