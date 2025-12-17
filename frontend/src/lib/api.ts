const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  headers?: Record<string, string>;
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;

  const headers: Record<string, string> = {
    ...options.headers,
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: options.method || "GET",
    headers,
    body: options.body instanceof FormData
      ? options.body
      : options.body
        ? JSON.stringify(options.body)
        : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// Types
export interface Contract {
  id: number;
  title: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  risk_level: string | null;
  created_at: string;
}

export interface ContractDetail extends Contract {
  file_url: string;
  extracted_text: string | null;
  analysis_result: AnalysisResult | null;
}

export interface ContractStats {
  total: number;
  completed: number;
  processing: number;
  failed: number;
}

export interface ContractListResponse {
  items: Contract[];
  total: number;
  skip: number;
  limit: number;
  stats: ContractStats;
}

// Legacy interface for backward compatibility
export interface RiskClause {
  text: string;
  level: "High" | "Medium" | "Low";
  explanation?: string;
  suggestion?: string;
  legal_basis?: string;
}

// Actual backend response structure
export interface RedliningChange {
  type: "modify" | "delete" | "add";
  original?: string;
  revised?: string;
  reason?: string;
  severity?: "High" | "Medium" | "Low";
}

export interface RedliningResult {
  changes?: RedliningChange[];
  change_count?: number;
}

export interface JudgmentResult {
  is_reliable?: boolean;
  overall_score?: number;
  confidence_level?: string;
  verdict?: string;
  recommendations?: string[];
}

export interface StressTestViolation {
  type?: string;
  severity?: string;
  description?: string;
  legal_basis?: string;
  current_value?: unknown;
  legal_standard?: unknown;
  suggestion?: string;  // V2: 수정 제안
  suggested_text?: string;  // V2: 수정된 조항 텍스트 (대체용)
  // V2: LLM 조항 분석 추가 필드
  clause_number?: string;
  sources?: string[];  // CRAG 검색 출처
  original_text?: string;  // 원본 조항 텍스트 (하이라이팅용)
  matched_text?: string;  // 하이라이팅할 실제 텍스트 (텍스트 기반 매칭용)
  start_index?: number;  // 원본 텍스트 시작 위치
  end_index?: number;  // 원본 텍스트 끝 위치
}

export interface StressTestResult {
  violations?: StressTestViolation[];
  total_underpayment?: number;
  annual_underpayment?: number;
}

export interface ConstitutionalCritique {
  principle?: string;
  violation_detected?: boolean;
  critique?: string;
  severity?: string;
  suggestion?: string;
}

export interface ConstitutionalReview {
  is_constitutional?: boolean;
  has_violations?: boolean;
  high_severity_count?: number;
  critiques?: ConstitutionalCritique[];
  revised_response?: string;
}

export interface ReasoningNode {
  id: string;
  type: string;
  label: string;
  content: string;
  metadata?: Record<string, unknown>;
  position?: { x: number; y: number };
  confidence?: number;
}

export interface ReasoningEdge {
  type: string;
  label: string;
  source: string;
  target: string;
  weight?: number;
}

export interface ReasoningTrace {
  nodes?: ReasoningNode[];
  edges?: ReasoningEdge[];
  summary?: string;
  conclusion?: string;
}

export interface AnalysisResult {
  // Main fields
  risk_level?: string;
  risk_score?: number;
  analysis_summary?: string;

  // Detailed results
  judgment?: JudgmentResult;
  redlining?: RedliningResult;
  stress_test?: StressTestResult;
  constitutional_review?: ConstitutionalReview;
  reasoning_trace?: ReasoningTrace;
  retrieved_docs?: Array<unknown>;

  // Metadata
  contract_id?: string;
  timestamp?: string;
  processing_time?: number;
  pipeline_version?: string;  // V2: "2.0.0" for LLM-based analysis

  // Legacy fields (for backward compatibility)
  summary?: string;
  risk_clauses?: RiskClause[];
  redlining_result?: RedliningResult;
  crag_result?: CRAGResult;
  constitutional_ai_result?: ConstitutionalAIResult;
}

export interface CRAGResult {
  query?: string;
  retrieved_documents?: Array<{
    source: string;
    text: string;
    relevance_score?: number;
  }>;
  final_answer?: string;
}

export interface ConstitutionalAIResult {
  original_response?: string;
  critique?: string;
  revised_response?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  conversation_id: string;
  message_id: string;
  sources?: Array<{
    source: string;
    text: string;
  }>;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

export interface Notification {
  id: string;
  type: "analysis_complete" | "analysis_failed" | "system";
  title: string;
  message: string;
  contract_id?: number;
  contract_title?: string;
  read: boolean;
  created_at: string;
}

// Auth API
export const authApi = {
  login: async (email: string, password: string): Promise<AuthResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Login failed" }));
      throw new Error(error.detail);
    }

    const data = await response.json();
    localStorage.setItem("access_token", data.access_token);
    return data;
  },

  register: async (email: string, password: string, username: string): Promise<User> => {
    return request<User>("/api/v1/auth/signup", {
      method: "POST",
      body: { email, password, username },
    });
  },

  logout: () => {
    localStorage.removeItem("access_token");
  },

  getMe: async (): Promise<User> => {
    return request<User>("/api/v1/users/me");
  },

  isAuthenticated: (): boolean => {
    if (typeof window === "undefined") return false;
    return !!localStorage.getItem("access_token");
  },
};

// Contracts API
export const contractsApi = {
  list: async (skip = 0, limit = 10): Promise<ContractListResponse> => {
    return request<ContractListResponse>(`/api/v1/contracts?skip=${skip}&limit=${limit}`);
  },

  get: async (id: number): Promise<ContractDetail> => {
    return request<ContractDetail>(`/api/v1/contracts/${id}`);
  },

  upload: async (file: File): Promise<{ message: string; contract_id: number; status: string }> => {
    const formData = new FormData();
    formData.append("file", file);

    return request("/api/v1/contracts/", {
      method: "POST",
      body: formData,
    });
  },

  delete: async (id: number): Promise<void> => {
    return request(`/api/v1/contracts/${id}`, {
      method: "DELETE",
    });
  },

  // 버전 관리 API
  getVersions: async (contractId: number): Promise<DocumentVersionListResponse> => {
    return request<DocumentVersionListResponse>(`/api/v1/contracts/${contractId}/versions`);
  },

  createVersion: async (
    contractId: number,
    data: DocumentVersionCreate
  ): Promise<DocumentVersion> => {
    return request<DocumentVersion>(`/api/v1/contracts/${contractId}/versions`, {
      method: "POST",
      body: data,
    });
  },

  getVersion: async (contractId: number, versionNumber: number): Promise<DocumentVersion> => {
    return request<DocumentVersion>(`/api/v1/contracts/${contractId}/versions/${versionNumber}`);
  },

  restoreVersion: async (contractId: number, versionNumber: number): Promise<DocumentVersion> => {
    return request<DocumentVersion>(`/api/v1/contracts/${contractId}/versions/${versionNumber}/restore`, {
      method: "POST",
    });
  },
};

// 버전 관리 타입 정의
export interface DocumentVersion {
  id: number;
  contract_id: number;
  version_number: number;
  content: string;
  changes?: Record<string, unknown>;
  change_summary?: string;
  is_current: boolean;
  created_at: string;
  created_by?: string;
}

export interface DocumentVersionCreate {
  content: string;
  changes?: Record<string, unknown>;
  change_summary?: string;
  created_by?: string;
}

export interface DocumentVersionListResponse {
  versions: DocumentVersion[];
  current_version: number;
}

// Chat API
export const chatApi = {
  send: async (
    contractId: number,
    message: string,
    conversationId?: string
  ): Promise<ChatResponse> => {
    return request<ChatResponse>(`/api/v1/chat/${contractId}`, {
      method: "POST",
      body: {
        message,
        conversation_id: conversationId,
      },
    });
  },

  getConversations: async (contractId: number) => {
    return request(`/api/v1/chat/${contractId}/conversations`);
  },

  getMessages: async (contractId: number, conversationId: string) => {
    return request(`/api/v1/chat/${contractId}/conversations/${conversationId}/messages`);
  },
};

// Direct AI Question (without Dify)
export const directAiApi = {
  askQuestion: async (
    contractId: number,
    selectedText: string,
    question: string
  ): Promise<{ answer: string }> => {
    // Dify가 설정되지 않았을 때 사용하는 직접 질문 API
    // 추후 백엔드에 별도 엔드포인트 추가 필요
    // 현재는 chat API를 활용
    const response = await chatApi.send(
      contractId,
      `선택된 텍스트: "${selectedText}"\n\n질문: ${question}`
    );
    return { answer: response.answer };
  },
};

// LangGraph Agent Chat API (Streaming)
export interface AgentStreamEvent {
  type: "step" | "tool" | "token" | "done" | "error" | "browser_guide";
  step?: string;
  message?: string;
  tool?: string;
  status?: string;
  content?: string;
  full_response?: string;
  // Browser guide specific fields
  session_id?: string;
  url?: string;
  procedure_type?: string;
}

export interface AgentChatMessage {
  role: "user" | "assistant";
  content: string;
}

export const agentChatApi = {
  /**
   * Stream chat response using Server-Sent Events
   */
  streamChat: (
    contractId: number,
    message: string,
    onEvent: (event: AgentStreamEvent) => void,
    onError?: (error: Error) => void,
    onComplete?: () => void
  ): (() => void) => {
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
    const abortController = new AbortController();

    const fetchStream = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/agent/${contractId}/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token && { Authorization: `Bearer ${token}` }),
          },
          body: JSON.stringify({
            message,
            include_contract: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No reader available");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                onEvent(data as AgentStreamEvent);
              } catch (e) {
                // Ignore parse errors
              }
            }
          }
        }

        onComplete?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          onError?.(error as Error);
        }
      }
    };

    fetchStream();

    // Return abort function
    return () => abortController.abort();
  },

  /**
   * Stream chat with conversation history
   */
  streamChatWithHistory: (
    contractId: number,
    message: string,
    history: AgentChatMessage[],
    onEvent: (event: AgentStreamEvent) => void,
    onError?: (error: Error) => void,
    onComplete?: () => void
  ): (() => void) => {
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
    const abortController = new AbortController();

    const fetchStream = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/agent/${contractId}/stream/history`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token && { Authorization: `Bearer ${token}` }),
          },
          body: JSON.stringify({
            message,
            history,
            include_contract: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No reader available");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                onEvent(data as AgentStreamEvent);
              } catch (e) {
                // Ignore parse errors
              }
            }
          }
        }

        onComplete?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          onError?.(error as Error);
        }
      }
    };

    fetchStream();

    return () => abortController.abort();
  },

  /**
   * Non-streaming fallback
   */
  chat: async (contractId: number, message: string): Promise<{ answer: string; tools_used: string[] }> => {
    return request(`/api/v1/agent/${contractId}`, {
      method: "POST",
      body: { message, include_contract: true },
    });
  },
};
