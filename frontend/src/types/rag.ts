export type EvidenceStatus = 'supported' | 'partially_supported' | 'insufficient' | 'out_of_scope';
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown';
export type QueryIntent = 
  | 'specific_clause_lookup'
  | 'definition_inquiry'
  | 'obligation_query'
  | 'threshold_parameter'
  | 'cross_contract_comparison'
  | 'compliance_audit'
  | 'general_legal'
  | 'unknown';

export interface Citation {
  citation_id: string;
  document_id: string;
  document_title?: string;
  document?: string;
  chunk_id: string;
  section?: string;
  section_title?: string;
  page?: number;
  category?: string;
  excerpt: string;
}


export interface RetrievalResult {
  chunk_id: string;
  parent_id?: string;
  document_id: string;
  document_title?: string;
  category?: string;
  section_number?: string;
  section_title?: string;
  page_start?: number;
  page_end?: number;
  text: string;
  dense_score?: number;
  sparse_score?: number;
  rrf_score?: number;
  reranker_score?: number;
  source?: string;
}

export interface QueryAnalysis {
  original_query: string;
  normalized_query?: string;
  intent?: QueryIntent;
  section_refs?: string[];
  act_names?: string[];
  exact_terms?: string[];
  jurisdictions?: string[];
  category_hints?: string[];
}

export interface QueryRequest {
  query: string;
  filters?: Record<string, any>;
  model_mode?: 'quality' | 'fast';
  conversation_context?: Array<Record<string, any>>;
}


export interface QueryResponse {
  query: string;
  answer: string;
  evidence_status: EvidenceStatus;
  confidence: ConfidenceLevel;
  citations: Citation[];
  supporting_chunks: RetrievalResult[];
  query_analysis?: QueryAnalysis;
  retrieval_attempts?: number;
  rewritten_query?: string;
  processing_metadata?: Record<string, any>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  response?: QueryResponse;
  isLoading?: boolean;
  isError?: boolean;
  errorMessage?: string;
  failedQuery?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}
