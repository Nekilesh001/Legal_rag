import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, ChatSession, Citation, QueryResponse } from '../types/rag';
import { RagApiService } from '../services/ragApi';

const STORAGE_KEY_SESSIONS = 'legal_rag_chat_sessions_v1';
const STORAGE_KEY_ACTIVE_ID = 'legal_rag_active_session_id_v1';
const STORAGE_KEY_MODEL_MODE = 'legal_rag_model_mode_v1';

function generateSessionTitle(query: string): string {
  const clean = query.trim();
  if (clean.length <= 45) return clean;
  return clean.slice(0, 45) + '...';
}

function loadSessionsFromStorage(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_SESSIONS);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.filter(s => s && typeof s.id === 'string' && Array.isArray(s.messages));
    }
    return [];
  } catch (err) {
    console.warn('LocalStorage error loading sessions, resetting safely:', err);
    return [];
  }
}

function saveSessionsToStorage(sessions: ChatSession[]): void {
  try {
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
  } catch (err) {
    console.warn('Failed to save chat sessions to localStorage:', err);
  }
}

export function useRagChat() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => loadSessionsFromStorage());
  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY_ACTIVE_ID);
    } catch {
      return null;
    }
  });

  const [modelMode, setModelMode] = useState<'quality' | 'fast'>(() => {
    try {
      return (localStorage.getItem(STORAGE_KEY_MODEL_MODE) as 'quality' | 'fast') || 'fast';
    } catch {
      return 'fast';
    }
  });


  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [activeCitationsList, setActiveCitationsList] = useState<Citation[]>([]);

  // Ensure active session exists
  useEffect(() => {
    if (sessions.length > 0 && (!activeSessionId || !sessions.some(s => s.id === activeSessionId))) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  // Sync activeSessionId to localStorage
  useEffect(() => {
    if (activeSessionId) {
      try {
        localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeSessionId);
      } catch {}
    }
  }, [activeSessionId]);

  // Sync modelMode to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_MODEL_MODE, modelMode);
    } catch {}
  }, [modelMode]);

  const activeSession = sessions.find(s => s.id === activeSessionId) || null;
  const messages = activeSession ? activeSession.messages : [];

  const createNewSession = useCallback((): string => {
    const newId = `sess_${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: 'New Legal Research Session',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
    };

    setSessions(prev => {
      const updated = [newSession, ...prev];
      saveSessionsToStorage(updated);
      return updated;
    });

    setActiveSessionId(newId);
    setError(null);
    setStatusMessage(null);
    return newId;
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    if (sessions.some(s => s.id === sessionId)) {
      setActiveSessionId(sessionId);
      setError(null);
      setStatusMessage(null);
    }
  }, [sessions]);

  const sendMessage = useCallback(async (queryText: string) => {
    if (!queryText.trim()) return;

    setError(null);
    setStatusMessage('Analyzing legal sources...');

    let targetSessionId = activeSessionId;
    if (!targetSessionId || !sessions.some(s => s.id === targetSessionId)) {
      targetSessionId = createNewSession();
    }

    const currentSess = sessions.find(s => s.id === targetSessionId);
    const existingMsgs = currentSess ? currentSess.messages : [];

    const userMsgId = `user_${Date.now()}`;
    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: queryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const assistantMsgId = `assistant_${Date.now()}`;
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isLoading: true,
    };

    // Add user message & placeholder assistant message
    setSessions(prevSessions => {
      const updated = prevSessions.map(sess => {
        if (sess.id === targetSessionId) {
          const isFirstUserMsg = sess.messages.filter(m => m.role === 'user').length === 0;
          const newTitle = isFirstUserMsg ? generateSessionTitle(queryText) : sess.title;
          return {
            ...sess,
            title: newTitle,
            updatedAt: new Date().toISOString(),
            messages: [...sess.messages, userMessage, initialAssistantMsg],
          };
        }
        return sess;
      });
      saveSessionsToStorage(updated);
      return updated;
    });

    setIsLoading(true);

    // Prepare conversation context (last 6 turns for controlled history window - Part 12)
    const contextTurns = existingMsgs.slice(-6).map(m => ({
      role: m.role,
      content: m.content,
      response: m.response ? {
        citations: m.response.citations,
        supporting_chunks: m.response.supporting_chunks,
      } : undefined,
    }));

    let accumulatedAnswer = '';
    let currentMetadata: { citations: Citation[]; confidence: any; evidence_status: any } = {
      citations: [],
      confidence: 'medium',
      evidence_status: 'supported',
    };

    await RagApiService.queryStream(
      {
        query: queryText,
        model_mode: modelMode,
        conversation_context: contextTurns,
      },
      {
        onStatus: (statusMsg) => {
          setStatusMessage(statusMsg);
        },
        onMetadata: (meta) => {
          currentMetadata = meta;
          setSessions(prevSessions => {
            const updated = prevSessions.map(sess => {
              if (sess.id === targetSessionId) {
                const updatedMsgs = sess.messages.map(m => {
                  if (m.id === assistantMsgId) {
                    return {
                      ...m,
                      response: {
                        query: queryText,
                        answer: accumulatedAnswer,
                        evidence_status: meta.evidence_status,
                        confidence: meta.confidence,
                        citations: meta.citations,
                        supporting_chunks: [],
                      },
                    };
                  }
                  return m;
                });
                return { ...sess, messages: updatedMsgs };
              }
              return sess;
            });
            return updated;
          });
        },
        onToken: (token) => {
          accumulatedAnswer += token;
          setSessions(prevSessions => {
            const updated = prevSessions.map(sess => {
              if (sess.id === targetSessionId) {
                const updatedMsgs = sess.messages.map(m => {
                  if (m.id === assistantMsgId) {
                    return {
                      ...m,
                      content: accumulatedAnswer,
                      isLoading: false,
                    };
                  }
                  return m;
                });
                return { ...sess, messages: updatedMsgs };
              }
              return sess;
            });
            return updated;
          });
        },
        onComplete: (data) => {
          setIsLoading(false);
          setStatusMessage(null);
          setSessions(prevSessions => {
            const updated = prevSessions.map(sess => {
              if (sess.id === targetSessionId) {
                const updatedMsgs = sess.messages.map(m => {
                  if (m.id === assistantMsgId) {
                    return {
                      ...m,
                      content: data.answer || accumulatedAnswer,
                      isLoading: false,
                      response: {
                        query: queryText,
                        answer: data.answer || accumulatedAnswer,
                        evidence_status: data.evidence_status || currentMetadata.evidence_status,
                        confidence: data.confidence || currentMetadata.confidence,
                        citations: data.citations || currentMetadata.citations,
                        supporting_chunks: [],
                      },
                    };
                  }
                  return m;
                });
                return { ...sess, updatedAt: new Date().toISOString(), messages: updatedMsgs };
              }
              return sess;
            });
            saveSessionsToStorage(updated);
            return updated;
          });
        },
        onError: (err) => {
          setIsLoading(false);
          setStatusMessage(null);
          console.error('RAG Query Error:', err);
          const userFacingError = 'Unable to complete this legal query. The legal knowledge service encountered an internal error. Please try again.';
          setError(userFacingError);

          setSessions(prevSessions => {
            const updated = prevSessions.map(sess => {
              if (sess.id === targetSessionId) {
                const updatedMsgs = sess.messages.map(m => {
                  if (m.id === assistantMsgId) {
                    return {
                      ...m,
                      content: userFacingError,
                      isLoading: false,
                      isError: true,
                      errorMessage: userFacingError,
                      failedQuery: queryText,
                    };
                  }
                  return m;
                });
                return { ...sess, messages: updatedMsgs };
              }
              return sess;
            });
            saveSessionsToStorage(updated);
            return updated;
          });
        },
      }
    );
  }, [activeSessionId, sessions, createNewSession, modelMode]);

  const clearCurrentChat = useCallback(() => {
    if (!activeSessionId) return;
    setSessions(prevSessions => {
      const updated = prevSessions.map(sess => {
        if (sess.id === activeSessionId) {
          return {
            ...sess,
            messages: [],
            updatedAt: new Date().toISOString(),
          };
        }
        return sess;
      });
      saveSessionsToStorage(updated);
      return updated;
    });
    setError(null);
    setStatusMessage(null);
  }, [activeSessionId]);

  const resetSession = useCallback(() => {
    createNewSession();
  }, [createNewSession]);

  const openCitationDrawer = useCallback((citation: Citation, citationsList: Citation[]) => {
    setSelectedCitation(citation);
    setActiveCitationsList(citationsList);
  }, []);

  const closeCitationDrawer = useCallback(() => {
    setSelectedCitation(null);
  }, []);

  return {
    sessions,
    activeSessionId,
    activeSession,
    messages,
    modelMode,
    setModelMode,
    isLoading,
    statusMessage,
    error,
    selectedCitation,
    activeCitationsList,
    sendMessage,
    createNewSession,
    switchSession,
    clearCurrentChat,
    resetSession,
    openCitationDrawer,
    closeCitationDrawer,
  };
}

export default useRagChat;
