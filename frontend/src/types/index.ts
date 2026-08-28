export * from './rag';

export type ContractType = 'NDA' | 'Employment' | 'Vendor Services' | 'Lease' | 'IP Assignment' | 'Master Services';
export type ContractStatus = 'Active' | 'Under Review' | 'Draft' | 'Expired' | 'Terminated';
export type RiskLevel = 'High' | 'Medium' | 'Low';

export interface ContractItem {
  id: string;
  title: string;
  type: ContractType;
  counterparty: string;
  status: ContractStatus;
  riskScore: number; // 0-100
  riskLevel: RiskLevel;
  updatedAt: string;
  effectiveDate: string;
  expirationDate: string;
  governingLaw: string;
  keyClausesCount: number;
  flaggedRisksCount: number;
}

export interface RiskFinding {
  id: string;
  contractId: string;
  contractTitle: string;
  clauseName: string;
  severity: RiskLevel;
  category: string;
  excerpt: string;
  riskDescription: string;
  recommendation: string;
}

export interface DashboardMetrics {
  totalContracts: number;
  reviewedContracts: number;
  highRiskContracts: number;
  pendingReviewCount: number;
  avgRiskScore: number;
}
