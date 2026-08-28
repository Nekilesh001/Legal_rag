import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppShell from '../components/layout/AppShell';
import DashboardPage from '../pages/Dashboard/DashboardPage';
import ContractsPage from '../pages/Contracts/ContractsPage';
import ContractReviewPage from '../pages/Review/ContractReviewPage';
import RiskAnalysisPage from '../pages/RiskAnalysis/RiskAnalysisPage';
import LegalChatPage from '../pages/LegalChat/LegalChatPage';
import SettingsPage from '../pages/Settings/SettingsPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<Navigate to="/legal-chat" replace />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="contracts" element={<ContractsPage />} />
          <Route path="review" element={<ContractReviewPage />} />
          <Route path="risk" element={<RiskAnalysisPage />} />
          <Route path="legal-chat" element={<LegalChatPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/legal-chat" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
