import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import MoltenMetal from '../background/MoltenMetal/MoltenMetal';

export const AppShell: React.FC = () => {
  const location = useLocation();

  const getPageTitle = (path: string) => {
    switch (path) {
      case '/':
      case '/dashboard':
        return 'Dashboard & Portfolio Overview';
      case '/contracts':
        return 'Contract Repository & Management';
      case '/review':
        return 'Contract Review & Clause Intelligence';
      case '/risk':
        return 'Risk Analysis & Statutory Audit';
      case '/legal-chat':
        return 'Legal AI Assistant & Knowledge Base';
      case '/settings':
        return 'Platform Settings & Governance';
      default:
        return 'Legal Intelligence Platform';
    }
  };

  // Determine WebGL background intensity based on route (Section 7)
  const getBackgroundIntensity = (path: string): 'high' | 'medium' | 'subtle' => {
    if (path === '/' || path === '/dashboard') return 'medium';
    if (path === '/legal-chat') return 'subtle';
    return 'subtle';
  };

  return (
    <div className="relative min-h-screen bg-[#090A0F] text-[#E2E8F0] flex overflow-hidden">
      {/* WebGL MoltenMetal Background Effect */}
      <MoltenMetal intensity={getBackgroundIntensity(location.pathname)} />

      {/* Persistent Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden z-10">
        <Header pageTitle={getPageTitle(location.pathname)} />

        <main className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppShell;
