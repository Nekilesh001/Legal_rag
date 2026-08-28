import { Citation, QueryRequest, QueryResponse } from '../types/rag';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function normalizeCitation(c: any, idx: number): Citation {
  if (!c || typeof c !== 'object') {
    return {
      citation_id: `C${idx + 1}`,
      document_id: 'doc',
      document_title: 'Legal Document',
      chunk_id: `chunk_${idx}`,
      section: undefined,
      page: undefined,
      excerpt: '',
    };
  }

  const docTitle = c.document_title || c.document || c.document_id || 'Legal Document';
  const sec = c.section || (c.section_number ? `Section ${c.section_number}` : undefined);
  const pg = c.page ?? c.page_start ?? undefined;
  const exc = c.excerpt || c.text || '';

  return {
    citation_id: c.citation_id || `C${idx + 1}`,
    document_id: c.document_id || 'doc',
    document_title: docTitle,
    chunk_id: c.chunk_id || `chunk_${idx}`,
    section: sec,
    section_title: c.section_title,
    page: pg,
    category: c.category,
    excerpt: exc,
  };
}

export function cleanAnswerText(rawText: string): string {
  if (!rawText) return '';
  const trimmed = rawText.trim();
  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed.answer === 'string') {
        return parsed.answer;
      }
    } catch {
      const match = trimmed.match(/"answer"\s*:\s*"([\s\S]*?)"(?:\s*,\s*"|\s*})/);
      if (match && match[1]) {
        return match[1]
          .replace(/\\n/g, '\n')
          .replace(/\\"/g, '"')
          .replace(/\\\\/g, '\\');
      }
    }
  }
  return rawText;
}

export interface StreamCallbacks {
  onStatus?: (message: string) => void;
  onMetadata?: (meta: { citations: Citation[]; confidence: any; evidence_status: any }) => void;
  onToken?: (token: string) => void;
  onComplete?: (data: Partial<QueryResponse>) => void;
  onError?: (err: Error) => void;
}

export class RagApiService {
  /**
   * Send a query to the frozen Legal RAG backend (synchronous).
   */
  static async query(request: QueryRequest): Promise<QueryResponse> {
    try {
      const response = await fetch(`${BASE_URL}/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data: QueryResponse = await response.json();
      const rawCites = Array.isArray(data.citations) ? data.citations : [];
      const normalizedCites = rawCites
        .filter(c => c && typeof c === 'object')
        .map((c, i) => normalizeCitation(c, i));

      return {
        ...data,
        answer: cleanAnswerText(data.answer),
        citations: normalizedCites,
      };
    } catch (error: any) {
      console.error('RAG API query error:', error);
      throw new Error(error.message || 'Failed to communicate with the Legal RAG Engine');
    }
  }

  /**
   * Stream a legal query via SSE (real-time token delivery).
   */
  static async queryStream(request: QueryRequest, callbacks: StreamCallbacks): Promise<void> {
    try {
      const response = await fetch(`${BASE_URL}/rag/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('ReadableStream not supported by response');

      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const lines = part.split('\n');
          let eventType = 'message';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              dataStr += line.slice(6);
            }
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventType === 'status' || parsed.type === 'status') {
              callbacks.onStatus?.(parsed.message);
            } else if (eventType === 'metadata' || parsed.type === 'metadata') {
              const rawCites = Array.isArray(parsed.citations) ? parsed.citations : [];
              const normalizedCites = rawCites.map((c: any, i: number) => normalizeCitation(c, i));
              callbacks.onMetadata?.({
                citations: normalizedCites,
                confidence: parsed.confidence || 'medium',
                evidence_status: parsed.evidence_status || 'supported',
              });
            } else if (eventType === 'token' || parsed.type === 'token') {
              callbacks.onToken?.(parsed.token || '');
            } else if (eventType === 'complete' || parsed.type === 'complete') {
              const rawCites = Array.isArray(parsed.citations) ? parsed.citations : [];
              const normalizedCites = rawCites.map((c: any, i: number) => normalizeCitation(c, i));
              const cleanAns = cleanAnswerText(parsed.answer || '');

              callbacks.onComplete?.({
                query: request.query,
                answer: cleanAns,
                citations: normalizedCites,
                confidence: parsed.confidence || 'medium',
                evidence_status: parsed.evidence_status || 'supported',
                supporting_chunks: [],
              });
            } else if (eventType === 'error' || parsed.type === 'error') {
              callbacks.onError?.(new Error(parsed.error || 'Server error during streaming'));
            }
          } catch (e) {
            console.warn('Malformed SSE data chunk:', dataStr);
          }
        }
      }
    } catch (error: any) {
      console.error('RAG API stream error:', error);
      callbacks.onError?.(new Error(error.message || 'Failed to stream from Legal RAG Engine'));
    }
  }

  /**
   * Check health of backend RAG Engine.
   */
  static async checkHealth(): Promise<{ status: string; engine_ready: boolean }> {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (!response.ok) return { status: 'error', engine_ready: false };
      return await response.json();
    } catch (error) {
      return { status: 'offline', engine_ready: false };
    }
  }
}

export default RagApiService;
