import { DashboardMetrics } from '../../types';

export const MOCK_DASHBOARD_METRICS: DashboardMetrics = {
  totalContracts: 148,
  reviewedContracts: 132,
  highRiskContracts: 12,
  pendingReviewCount: 16,
  avgRiskScore: 34,
};

export const MOCK_RECENT_ACTIVITIES = [
  {
    id: 'act_1',
    title: 'Legal RAG Engine Verified Section 73 Query',
    subtitle: 'Indian Contract Act, 1872 — Grounded Citation verified',
    timestamp: '10 mins ago',
    type: 'chat',
    status: 'success',
  },
  {
    id: 'act_2',
    title: 'High Risk Flagged in SaaS Vendor Subcontract',
    subtitle: 'Limitation of liability capped below 1x annual fee',
    timestamp: '1 hour ago',
    type: 'risk',
    status: 'warning',
  },
  {
    id: 'act_3',
    title: 'Executive Employment Agreement Uploaded',
    subtitle: 'VP Engineering — Notice Period Clause Analysis Pending',
    timestamp: '3 hours ago',
    type: 'contract',
    status: 'info',
  },
  {
    id: 'act_4',
    title: 'Lease Deed Covenant Audit Completed',
    subtitle: 'Transfer of Property Act Sec 105 Compliance Verified',
    timestamp: 'Yesterday',
    type: 'review',
    status: 'success',
  },
];
