"use client";

import { useState, useEffect, useCallback, use, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import dynamic from "next/dynamic";
import { ContractDetail, contractsApi, authApi, RiskClause, StressTestViolation, agentChatApi, AgentStreamEvent, AgentChatMessage } from "@/lib/api";
import {
  IconArrowLeft,
  IconLoading,
  IconCheck,
  IconWarning,
  IconDanger,
  IconChevronRight,
  IconChat,
  IconSend,
  IconClose,
  IconInfo,
} from "@/components/icons";
import { cn } from "@/lib/utils";

// PDF 뷰어는 클라이언트 사이드에서만 로드
const PDFViewer = dynamic(() => import("@/components/pdf-viewer"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-white">
      <div className="flex flex-col items-center gap-3">
        <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-gray-500">PDF 뷰어 로딩 중...</p>
      </div>
    </div>
  ),
});

interface AnalysisPageProps {
  params: Promise<{
    id: string;
  }>;
}

function RiskLevelBadge({ level }: { level: string }) {
  const normalizedLevel = level.toLowerCase();

  if (normalizedLevel === "high" || normalizedLevel === "danger") {
    return (
      <span className="badge badge-danger">
        <IconDanger size={12} />
        High Risk
      </span>
    );
  }
  if (normalizedLevel === "medium" || normalizedLevel === "warning") {
    return (
      <span className="badge badge-warning">
        <IconWarning size={12} />
        Medium Risk
      </span>
    );
  }
  return (
    <span className="badge badge-success">
      <IconCheck size={12} />
      Low Risk
    </span>
  );
}

// Normalized clause item that works with both old and new data structures
interface NormalizedClause {
  text: string;
  level: string;
  explanation?: string;
  suggestion?: string;
  clauseNumber?: string;      // V2: 조항 번호
  sources?: string[];         // V2: CRAG 검색 출처
  legalBasis?: string;        // V2: 법적 근거
  originalText?: string;      // 원본 계약서에서 매칭할 텍스트
}

interface RiskClauseItemProps {
  clause: NormalizedClause;
  index: number;
}

function RiskClauseItem({ clause, index }: RiskClauseItemProps) {
  const [expanded, setExpanded] = useState(false);

  const getLevelStyles = (level: string) => {
    const l = level.toLowerCase();
    if (l === "high") return "border-l-red-500 bg-gradient-to-r from-red-50/80 to-transparent";
    if (l === "medium") return "border-l-amber-500 bg-gradient-to-r from-amber-50/80 to-transparent";
    return "border-l-green-500 bg-gradient-to-r from-green-50/80 to-transparent";
  };

  return (
    <div className={cn(
      "border-l-4 rounded-xl overflow-hidden transition-all duration-200",
      getLevelStyles(clause.level),
      expanded && "shadow-sm"
    )}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 text-left flex items-start gap-3 hover:bg-white/50 transition-colors"
      >
        <span className={cn(
          "flex-shrink-0 mt-0.5 transition-transform duration-200",
          expanded && "rotate-90"
        )}>
          <IconChevronRight size={16} className="text-gray-400" />
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {clause.clauseNumber ? (
              <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                {clause.clauseNumber}
              </span>
            ) : (
              <span className="text-xs font-semibold text-gray-400">#{index + 1}</span>
            )}
            <RiskLevelBadge level={clause.level} />
          </div>
          <p className="text-sm text-gray-800 line-clamp-2">{clause.text}</p>
        </div>
      </button>
      <div className={cn(
        "overflow-hidden transition-all duration-300",
        expanded ? "max-h-[600px] opacity-100" : "max-h-0 opacity-0"
      )}>
        <div className="px-4 pb-4 pl-10 space-y-3">
          {clause.explanation && (
            <div className="bg-white/60 rounded-lg p-3">
              <p className="text-xs font-semibold text-gray-500 mb-1">위험 사유</p>
              <p className="text-sm text-gray-700">{clause.explanation}</p>
            </div>
          )}
          {clause.legalBasis && (
            <div className="bg-blue-50/60 rounded-lg p-3">
              <p className="text-xs font-semibold text-blue-600 mb-1">법적 근거</p>
              <p className="text-sm text-gray-700">{clause.legalBasis}</p>
            </div>
          )}
          {clause.suggestion && (
            <div className="bg-white/60 rounded-lg p-3">
              <p className="text-xs font-semibold text-gray-500 mb-1">수정 제안</p>
              <p className="text-sm text-gray-700">{clause.suggestion}</p>
            </div>
          )}
          {clause.sources && clause.sources.length > 0 && (
            <div className="bg-gray-50/60 rounded-lg p-3">
              <p className="text-xs font-semibold text-gray-500 mb-2">참조 출처 (CRAG)</p>
              <div className="flex flex-wrap gap-1.5">
                {clause.sources.map((source, idx) => (
                  <span
                    key={idx}
                    className="text-xs bg-white px-2 py-1 rounded border border-gray-200 text-gray-600"
                  >
                    {source}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper function to normalize different data structures
interface RedliningChange {
  original?: string;
  severity?: string;
  reason?: string;
  revised?: string;
}

function normalizeRiskClauses(analysis: ContractDetail["analysis_result"]): NormalizedClause[] {
  if (!analysis) return [];

  const allClauses: NormalizedClause[] = [];

  // 1. StressTest violations - V2 LLM 분석 또는 Legacy 정규식 분석 결과
  if (analysis.stress_test?.violations && analysis.stress_test.violations.length > 0) {
    analysis.stress_test.violations.forEach((v: StressTestViolation) => {
      const severity = v.severity?.toUpperCase();
      allClauses.push({
        text: v.type || "법적 기준 위반",
        level: severity === "CRITICAL" || severity === "HIGH" ? "High" : severity === "MEDIUM" ? "Medium" : "Low",
        explanation: v.description || `현재 값: ${v.current_value}, 법적 기준: ${v.legal_standard}`,
        suggestion: v.suggestion,
        clauseNumber: v.clause_number,
        sources: v.sources,
        legalBasis: v.legal_basis,
        // API에서 받은 원본 조항 텍스트 사용 (정확한 하이라이팅용)
        originalText: v.original_text || "",
      });
    });
  }

  // 2. Redlining changes - 계약서 조항의 불공정 분석
  if (analysis.redlining?.changes && analysis.redlining.changes.length > 0) {
    analysis.redlining.changes.forEach((change: RedliningChange) => {
      // stress_test와 중복되지 않는 항목만 추가
      const isDuplicate = allClauses.some(c => c.text === change.original);
      if (!isDuplicate) {
        allClauses.push({
          text: change.original || "",
          level: change.severity || "Medium",
          explanation: change.reason,
          suggestion: change.revised,
          originalText: change.original || "",
        });
      }
    });
  }

  // 3. Fall back to legacy structure: risk_clauses
  if (allClauses.length === 0 && analysis.risk_clauses && analysis.risk_clauses.length > 0) {
    return analysis.risk_clauses.map((clause: RiskClause) => ({
      text: clause.text,
      level: clause.level,
      explanation: clause.explanation,
      suggestion: clause.suggestion,
      originalText: clause.text,
    }));
  }

  return allClauses;
}

interface TextSelectionTooltipProps {
  position: { x: number; y: number };
  onAsk: () => void;
  onClose: () => void;
}

function TextSelectionTooltip({ position, onAsk, onClose }: TextSelectionTooltipProps) {
  return (
    <div
      className="fixed z-50 bg-gray-900 text-white rounded-xl shadow-strong px-4 py-2.5 text-sm animate-scaleIn"
      style={{
        left: position.x,
        top: position.y,
        transform: "translate(-50%, -100%)",
        marginTop: "-12px",
      }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={onAsk}
          className="flex items-center gap-1.5 hover:text-gray-300 transition-colors"
        >
          <IconChat size={14} />
          AI에게 질문하기
        </button>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <IconClose size={14} />
        </button>
      </div>
      <div className="absolute left-1/2 top-full -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[6px] border-transparent border-t-gray-900" />
    </div>
  );
}

interface ToolStatus {
  tool: string;
  status: "searching" | "complete";
  message: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

interface ChatPanelProps {
  contractId: number;
  initialQuestion?: string;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  onClose: () => void;
}

function ChatPanel({ contractId, initialQuestion, messages, setMessages, onClose }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [toolStatuses, setToolStatuses] = useState<ToolStatus[]>([]);
  const [streamingContent, setStreamingContent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const initialQuestionProcessedRef = useRef<string | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, scrollToBottom]);

  useEffect(() => {
    // Prevent duplicate: only send if this exact question hasn't been processed
    if (initialQuestion && initialQuestionProcessedRef.current !== initialQuestion) {
      initialQuestionProcessedRef.current = initialQuestion;
      sendMessage(initialQuestion);
    }
  }, [initialQuestion]);

  const handleStreamEvent = useCallback((event: AgentStreamEvent) => {
    switch (event.type) {
      case "step":
        setCurrentStep(event.message || null);
        break;

      case "tool":
        if (event.tool && event.status && event.message) {
          const toolName = event.tool;
          const toolStatus = event.status as "searching" | "complete";
          const toolMessage = event.message;
          setToolStatuses((prev) => {
            const existing = prev.find((t) => t.tool === toolName);
            if (existing) {
              return prev.map((t) =>
                t.tool === toolName
                  ? { ...t, status: toolStatus, message: toolMessage }
                  : t
              );
            }
            return [...prev, { tool: toolName, status: toolStatus, message: toolMessage }];
          });
        }
        break;

      case "token":
        if (event.content) {
          setStreamingContent((prev) => prev + event.content);
        }
        break;

      case "done":
        const finalContent = event.full_response || streamingContent;
        setMessages((prev) => [
          ...prev.filter((m) => !m.isStreaming),
          { role: "assistant", content: finalContent },
        ]);
        setStreamingContent("");
        setIsStreaming(false);
        setCurrentStep(null);
        setToolStatuses([]);
        break;

      case "error":
        setMessages((prev) => [
          ...prev.filter((m) => !m.isStreaming),
          { role: "assistant", content: `오류가 발생했습니다: ${event.message}` },
        ]);
        setStreamingContent("");
        setIsStreaming(false);
        setCurrentStep(null);
        setToolStatuses([]);
        break;
    }
  }, [streamingContent]);

  function sendMessage(text: string) {
    if (!text.trim() || isStreaming) return;

    // Add user message
    const userMessage: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);
    setStreamingContent("");
    setToolStatuses([]);

    // Build history for context
    const history: AgentChatMessage[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Start streaming
    const abort = agentChatApi.streamChatWithHistory(
      contractId,
      text,
      history,
      handleStreamEvent,
      (error) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `연결 오류: ${error.message}` },
        ]);
        setIsStreaming(false);
        setCurrentStep(null);
        setToolStatuses([]);
      },
      () => {
        // Complete callback
      }
    );

    abortRef.current = abort;
  }

  function stopGeneration() {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    if (streamingContent) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: streamingContent + "\n\n_(생성 중단됨)_" },
      ]);
    }
    setStreamingContent("");
    setIsStreaming(false);
    setCurrentStep(null);
    setToolStatuses([]);
  }

  return (
    <div className="flex flex-col h-full animate-slideInRight safe-area-inset">
      {/* Header */}
      <div className="flex items-center justify-between px-4 sm:px-5 py-3 sm:py-4 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl flex items-center justify-center shadow-sm flex-shrink-0">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-white">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2Z" fill="currentColor" fillOpacity="0.2"/>
              <path d="M12 6C8.69 6 6 8.69 6 12C6 13.1 6.3 14.12 6.81 15.1L6 18L8.9 17.19C9.88 17.7 10.9 18 12 18C15.31 18 18 15.31 18 12C18 8.69 15.31 6 12 6Z" fill="currentColor"/>
              <circle cx="9" cy="12" r="1" fill="#1f2937"/>
              <circle cx="12" cy="12" r="1" fill="#1f2937"/>
              <circle cx="15" cy="12" r="1" fill="#1f2937"/>
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900 tracking-tight">AI 어시스턴트</h3>
            <p className="text-[10px] text-gray-400 tracking-tight">LangGraph Agent</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-200 min-w-[44px] min-h-[44px] flex items-center justify-center"
        >
          <IconClose size={18} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
        {messages.length === 0 && !isStreaming && (
          <div className="text-center py-8 sm:py-12 animate-fadeIn">
            <div className="inline-flex items-center justify-center w-12 h-12 sm:w-14 sm:h-14 bg-gray-100 rounded-2xl mb-3 sm:mb-4">
              <IconChat size={24} className="text-gray-400 sm:w-7 sm:h-7" />
            </div>
            <p className="text-sm text-gray-500 tracking-tight">이 계약서에 대해 질문하세요</p>
            <p className="text-xs text-gray-400 mt-1 tracking-tight">AI가 법령/판례를 검색하고 분석합니다</p>

            {/* Quick prompts */}
            <div className="mt-5 sm:mt-6 space-y-2">
              {[
                "이 계약서의 주요 위험 요소는?",
                "위약금 조항이 적법한가요?",
                "노동청에 신고하려면 어떻게 해야 하나요?",
              ].map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="block w-full text-left px-4 py-3 sm:py-2.5 text-sm text-gray-600 bg-white border border-gray-200 rounded-xl hover:border-gray-300 hover:bg-gray-50 transition-all duration-200 tracking-tight min-h-[48px] sm:min-h-0"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "animate-fadeInUp",
              msg.role === "user" ? "flex justify-end" : ""
            )}
            style={{ animationDelay: `${i * 30}ms` }}
          >
            {msg.role === "user" ? (
              <div className="max-w-[85%] px-4 py-3 bg-gray-900 text-white text-sm rounded-2xl rounded-br-md">
                {msg.content}
              </div>
            ) : (
              <div className="max-w-full">
                <div className="prose prose-sm prose-gray max-w-none bg-gray-50/80 rounded-2xl rounded-bl-md px-4 py-3">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Streaming state */}
        {isStreaming && (
          <div className="space-y-3 animate-fadeIn">
            {/* Tool execution status */}
            {(currentStep || toolStatuses.length > 0) && (
              <div className="bg-blue-50/80 rounded-xl p-3 space-y-2">
                {currentStep && (
                  <div className="flex items-center gap-2 text-xs text-blue-600">
                    <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                    {currentStep}
                  </div>
                )}
                {toolStatuses.map((tool, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "flex items-center gap-2 text-xs transition-all duration-300",
                      tool.status === "complete" ? "text-green-600" : "text-blue-600"
                    )}
                  >
                    {tool.status === "searching" ? (
                      <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <IconCheck size={12} />
                    )}
                    <span>{tool.message}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Streaming content */}
            {streamingContent && (
              <div className="max-w-full">
                <div className="prose prose-sm prose-gray max-w-none bg-gray-50/80 rounded-2xl rounded-bl-md px-4 py-3">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {streamingContent}
                  </ReactMarkdown>
                  <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5" />
                </div>
              </div>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 sm:p-4 border-t border-gray-100 bg-white pb-safe">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="input-field text-sm h-11 sm:h-auto"
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button
              type="button"
              onClick={stopGeneration}
              className="px-4 py-2.5 bg-red-500 text-white rounded-xl shadow-sm hover:bg-red-600 transition-all duration-200 min-w-[48px] min-h-[44px] flex items-center justify-center flex-shrink-0"
            >
              <IconClose size={18} />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-4 py-2.5 bg-gray-900 text-white rounded-xl shadow-sm hover:bg-gray-800 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 min-w-[48px] min-h-[44px] flex items-center justify-center flex-shrink-0"
            >
              <IconSend size={18} />
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

export default function AnalysisPage({ params }: AnalysisPageProps) {
  const { id } = use(params);
  const router = useRouter();
  const [contract, setContract] = useState<ContractDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "clauses" | "text">("overview");
  const [showChat, setShowChat] = useState(false);
  const [chatInitialQuestion, setChatInitialQuestion] = useState<string | undefined>();
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]); // Persistent chat messages
  const [chatWidth, setChatWidth] = useState(384); // Default: 24rem = 384px
  const [isResizing, setIsResizing] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [sortBy, setSortBy] = useState<"default" | "risk" | "clause">("default");
  const [mobileView, setMobileView] = useState<"pdf" | "analysis">("analysis"); // Mobile view switcher
  const [isMobile, setIsMobile] = useState(false); // For SSR-safe mobile detection

  useEffect(() => {
    if (!authApi.isAuthenticated()) {
      router.push("/login");
      return;
    }

    loadContract();
  }, [id, router]);

  // Mobile detection for SSR compatibility
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  async function loadContract() {
    try {
      setLoading(true);
      const data = await contractsApi.get(parseInt(id));
      setContract(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "계약서를 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }

  const handleAskAboutSelection = () => {
    if (selectedText) {
      setChatInitialQuestion(`"${selectedText.substring(0, 200)}${selectedText.length > 200 ? '...' : ''}" - 이 조항은 무슨 의미이고, 위험한가요?`);
      setShowChat(true);
      setSelectedText(null);
      setTooltipPosition(null);
    }
  };

  // 오른쪽 패널 텍스트 선택 핸들러
  const handleRightPanelTextSelect = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const text = selection.toString().trim();
    if (text.length < 10) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    setSelectedText(text);
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  }, []);

  // 위험 조항 정렬 함수
  const sortRiskClauses = useCallback((clauses: NormalizedClause[]) => {
    if (sortBy === "default") return clauses;

    return [...clauses].sort((a, b) => {
      if (sortBy === "risk") {
        // 위험도순: High > Medium > Low
        const riskOrder: Record<string, number> = { high: 0, medium: 1, low: 2 };
        const aRisk = riskOrder[a.level.toLowerCase()] ?? 3;
        const bRisk = riskOrder[b.level.toLowerCase()] ?? 3;
        return aRisk - bRisk;
      }
      if (sortBy === "clause") {
        // 조항 번호순
        const aNum = a.clauseNumber || "zzz";
        const bNum = b.clauseNumber || "zzz";
        // 숫자 부분 추출해서 비교
        const aMatch = aNum.match(/^(\d+)/);
        const bMatch = bNum.match(/^(\d+)/);
        if (aMatch && bMatch) {
          return parseInt(aMatch[1]) - parseInt(bMatch[1]);
        }
        return aNum.localeCompare(bNum);
      }
      return 0;
    });
  }, [sortBy]);

  // Chat panel resize handlers
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);

    const startX = e.clientX;
    const startWidth = chatWidth;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = startX - e.clientX;
      const newWidth = Math.min(Math.max(startWidth + delta, 320), 640); // Min 320px, Max 640px
      setChatWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, [chatWidth]);

  if (loading) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center bg-gradient-to-b from-gray-50 to-white">
        <div className="flex flex-col items-center gap-3 animate-fadeIn">
          <IconLoading size={32} className="text-gray-400" />
          <p className="text-sm text-gray-500 tracking-tight">분석 결과를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !contract) {
    return (
      <div className="min-h-[100dvh] bg-gradient-to-b from-gray-50/50 to-white">
        <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200/80 sticky top-0 z-10">
          <div className="px-4 sm:px-5 h-14 flex items-center">
            <Link href="/" className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 transition-colors min-h-[44px]">
              <IconArrowLeft size={16} />
              <span className="tracking-tight">돌아가기</span>
            </Link>
          </div>
        </header>
        <main className="px-4 sm:px-8 py-12 sm:py-16 text-center animate-fadeIn">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-red-100 rounded-2xl mb-4">
            <IconDanger size={28} className="text-red-500" />
          </div>
          <p className="text-red-600 font-medium tracking-tight">{error || "계약서를 찾을 수 없습니다"}</p>
        </main>
      </div>
    );
  }

  const analysis = contract.analysis_result;
  const riskClauses = normalizeRiskClauses(analysis);
  const summary = analysis?.analysis_summary || analysis?.summary;

  return (
    <div className={cn("min-h-[100dvh] bg-gray-50/30 flex flex-col md:flex-row", isResizing && "select-none cursor-ew-resize")}>
      {/* Main Content */}
      <div
        className="flex-1 flex flex-col transition-all duration-300"
        style={{ marginRight: showChat && !isMobile ? chatWidth : 0 }}
      >
        <header className="bg-white/90 backdrop-blur-sm border-b border-gray-200/80 sticky top-0 z-10">
          <div className="px-3 sm:px-5 h-14 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1">
              <Link
                href="/"
                className="p-2.5 -ml-1 sm:-ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-200 flex-shrink-0"
              >
                <IconArrowLeft size={18} />
              </Link>
              <div className="min-w-0 flex-1">
                <h1 className="text-sm font-semibold text-gray-900 truncate tracking-tight">
                  {contract.title}
                </h1>
              </div>
            </div>
            <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
              {analysis?.risk_level && <RiskLevelBadge level={analysis.risk_level} />}
              <button
                onClick={() => setShowChat(true)}
                className="inline-flex items-center justify-center gap-2 px-3 sm:px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-xl shadow-sm hover:bg-gray-50 hover:shadow transition-all duration-200 min-h-[44px]"
              >
                <IconChat size={16} />
                <span className="hidden xs:inline tracking-tight">AI 질문</span>
              </button>
            </div>
          </div>
        </header>

        {/* Mobile View Switcher */}
        <div className="md:hidden flex border-b border-gray-200 bg-white">
          <button
            onClick={() => setMobileView("pdf")}
            className={cn(
              "flex-1 py-3 text-sm font-medium transition-all duration-200 relative tracking-tight",
              mobileView === "pdf" ? "text-gray-900" : "text-gray-500"
            )}
          >
            원본 문서
            {mobileView === "pdf" && (
              <div className="absolute bottom-0 left-4 right-4 h-0.5 bg-gray-900 rounded-full" />
            )}
          </button>
          <button
            onClick={() => setMobileView("analysis")}
            className={cn(
              "flex-1 py-3 text-sm font-medium transition-all duration-200 relative tracking-tight",
              mobileView === "analysis" ? "text-gray-900" : "text-gray-500"
            )}
          >
            분석 결과
            {mobileView === "analysis" && (
              <div className="absolute bottom-0 left-4 right-4 h-0.5 bg-gray-900 rounded-full" />
            )}
          </button>
        </div>

        <div className="flex-1 flex animate-fadeIn overflow-hidden">
          {/* Left Panel - PDF Viewer (Desktop: always visible, Mobile: conditional) */}
          <div className={cn(
            "flex-col bg-white",
            // Desktop: 50% width
            "md:flex md:w-1/2 md:border-r md:border-gray-200/80",
            // Mobile: full width or hidden
            mobileView === "pdf" ? "flex w-full" : "hidden"
          )}>
            <PDFViewer
              fileUrl={contract.file_url}
              extractedText={contract.extracted_text}
              onTextSelect={(text, position) => {
                setSelectedText(text);
                setTooltipPosition(position);
              }}
              className="flex-1"
            />
          </div>

          {/* Right Panel - Analysis (Desktop: always visible, Mobile: conditional) */}
          <div className={cn(
            "flex-col bg-white",
            // Desktop: 50% width
            "md:flex md:w-1/2",
            // Mobile: full width or hidden
            mobileView === "analysis" ? "flex w-full" : "hidden"
          )}>
            <div className="flex border-b border-gray-200 px-1 sm:px-2 overflow-x-auto scrollbar-hide">
              {[
                { key: "overview", label: "개요" },
                { key: "clauses", label: `위험 조항 (${riskClauses.length})` },
                { key: "text", label: "전체 텍스트" },
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as typeof activeTab)}
                  className={cn(
                    "flex-1 px-3 sm:px-4 py-3 text-sm font-medium transition-all duration-200 relative whitespace-nowrap tracking-tight",
                    activeTab === tab.key
                      ? "text-gray-900"
                      : "text-gray-500 hover:text-gray-700"
                  )}
                >
                  {tab.label}
                  {activeTab === tab.key && (
                    <div className="absolute bottom-0 left-2 right-2 h-0.5 bg-gray-900 rounded-full" />
                  )}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-auto p-3 sm:p-5" onMouseUp={handleRightPanelTextSelect}>
              {activeTab === "overview" && (
                <div className="space-y-5 animate-fadeIn">
                  {summary && (
                    <div className="card p-4">
                      <h3 className="text-sm font-semibold text-gray-900 mb-2">요약</h3>
                      <p className="text-sm text-gray-700 leading-relaxed">{summary}</p>
                    </div>
                  )}

                  <div className="grid grid-cols-3 gap-2 sm:gap-3">
                    <div className="card p-3 sm:p-4 text-center">
                      <p className="text-[10px] sm:text-xs font-medium text-gray-500 mb-0.5 sm:mb-1 tracking-tight">위험도</p>
                      <p className="text-base sm:text-lg font-semibold text-gray-900">
                        {analysis?.risk_level || "N/A"}
                      </p>
                    </div>
                    <div className="card p-3 sm:p-4 text-center">
                      <p className="text-[10px] sm:text-xs font-medium text-gray-500 mb-0.5 sm:mb-1 tracking-tight">위험 조항</p>
                      <p className="text-base sm:text-lg font-semibold text-gray-900">{riskClauses.length}</p>
                    </div>
                    <div className="card p-3 sm:p-4 text-center">
                      <p className="text-[10px] sm:text-xs font-medium text-gray-500 mb-0.5 sm:mb-1 tracking-tight">고위험</p>
                      <p className="text-base sm:text-lg font-semibold text-red-600">
                        {riskClauses.filter((c: NormalizedClause) => c.level.toLowerCase() === "high").length}
                      </p>
                    </div>
                  </div>

                  {riskClauses.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-900 mb-3">주요 이슈</h3>
                      <div className="space-y-3">
                        {riskClauses.slice(0, 3).map((clause, i) => (
                          <RiskClauseItem
                            key={i}
                            clause={clause}
                            index={i}
                          />
                        ))}
                        {riskClauses.length > 3 && (
                          <button
                            onClick={() => setActiveTab("clauses")}
                            className="text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
                          >
                            전체 {riskClauses.length}개 조항 보기 →
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* 체불 예상액 */}
                  {analysis?.stress_test && (analysis.stress_test.annual_underpayment || 0) > 0 && (
                    <div className="card p-4 border-l-4 border-l-red-500">
                      <h3 className="text-sm font-semibold text-gray-900 mb-3">체불 예상액</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-gray-500 mb-1">월간 체불 예상</p>
                          <p className="text-lg font-bold text-red-600">
                            {(analysis.stress_test.total_underpayment || 0).toLocaleString()}원
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500 mb-1">연간 체불 예상</p>
                          <p className="text-lg font-bold text-red-600">
                            {(analysis.stress_test.annual_underpayment || 0).toLocaleString()}원
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* AI 분석 품질 - 내부 품질 지표로 덜 강조 */}
                  {analysis?.judgment && (
                    <div className="card p-4 bg-gray-50/50 border-gray-100">
                      <details className="group">
                        <summary className="flex items-center justify-between cursor-pointer list-none">
                          <h3 className="text-xs font-medium text-gray-500">분석 품질 지표</h3>
                          <span className={cn(
                            "px-2 py-0.5 text-xs rounded",
                            analysis.judgment.is_reliable ? "bg-green-100 text-green-600" : "bg-amber-100 text-amber-600"
                          )}>
                            {Math.round((analysis.judgment.overall_score || 0) * 100)}%
                          </span>
                        </summary>
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <p className="text-xs text-gray-500 mb-2">
                            AI 분석의 내부 품질 검증 결과입니다. 계약서 위험도와는 별개입니다.
                          </p>
                          {analysis.judgment.verdict && (
                            <p className="text-xs text-gray-600">{analysis.judgment.verdict}</p>
                          )}
                        </div>
                      </details>
                    </div>
                  )}


                  {analysis?.crag_result?.final_answer && (
                    <div className="card p-4">
                      <h3 className="text-sm font-semibold text-gray-900 mb-2 flex items-center gap-1.5">
                        <IconInfo size={14} className="text-blue-500" />
                        법률 참조
                      </h3>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {analysis.crag_result.final_answer}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "clauses" && (
                <div className="space-y-3 animate-fadeIn">
                  {riskClauses.length === 0 ? (
                    <div className="text-center py-12">
                      <div className="inline-flex items-center justify-center w-14 h-14 bg-green-100 rounded-2xl mb-4">
                        <IconCheck size={28} className="text-green-600" />
                      </div>
                      <p className="text-sm font-medium text-gray-600">위험 조항이 발견되지 않았습니다</p>
                      <p className="text-xs text-gray-400 mt-1">이 계약서는 안전한 것으로 보입니다</p>
                    </div>
                  ) : (
                    <>
                      {/* 정렬 옵션 */}
                      <div className="flex items-center gap-2 mb-3 pb-3 border-b border-gray-100">
                        <span className="text-[11px] text-gray-400 uppercase tracking-wider">정렬</span>
                        <div className="flex items-center gap-0.5 bg-gray-100/80 rounded-full p-0.5">
                          {[
                            { key: "default", label: "기본" },
                            { key: "risk", label: "위험도" },
                            { key: "clause", label: "조항" },
                          ].map((option) => (
                            <button
                              key={option.key}
                              onClick={() => setSortBy(option.key as typeof sortBy)}
                              className={cn(
                                "px-3 py-1 text-xs font-medium rounded-full transition-all duration-200",
                                sortBy === option.key
                                  ? "bg-white text-gray-900 shadow-sm"
                                  : "text-gray-500 hover:text-gray-700"
                              )}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      {/* 정렬된 위험 조항 목록 */}
                      {sortRiskClauses(riskClauses).map((clause, i) => (
                        <RiskClauseItem
                          key={i}
                          clause={clause}
                          index={i}
                        />
                      ))}
                    </>
                  )}
                </div>
              )}

              {activeTab === "text" && (
                <div className="card p-5 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap animate-fadeIn">
                  {contract.extracted_text || "추출된 텍스트가 없습니다"}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Chat Panel - Only render one based on screen size to prevent duplicate messages */}
      {showChat && isMobile && (
        <div className="fixed inset-0 bg-white z-50 animate-slideInUp">
          <ChatPanel
            contractId={parseInt(id)}
            initialQuestion={chatInitialQuestion}
            messages={chatMessages}
            setMessages={setChatMessages}
            onClose={() => {
              setShowChat(false);
              setChatInitialQuestion(undefined);
            }}
          />
        </div>
      )}

      {showChat && !isMobile && (
        <div
          className="fixed right-0 top-0 bottom-0 bg-white border-l border-gray-200 shadow-strong z-40"
          style={{ width: chatWidth }}
        >
          {/* Resize Handle */}
          <div
            onMouseDown={handleResizeStart}
            className={cn(
              "absolute -left-1 top-0 bottom-0 w-2 cursor-ew-resize group z-10",
              "flex items-center justify-center"
            )}
          >
            {/* Subtle line indicator */}
            <div className={cn(
              "w-0.5 h-8 rounded-full transition-all duration-200",
              "bg-gray-200 group-hover:bg-gray-400 group-hover:h-12",
              isResizing && "bg-blue-500 h-16"
            )} />
          </div>
          <ChatPanel
            contractId={parseInt(id)}
            initialQuestion={chatInitialQuestion}
            messages={chatMessages}
            setMessages={setChatMessages}
            onClose={() => {
              setShowChat(false);
              setChatInitialQuestion(undefined);
            }}
          />
        </div>
      )}

      {/* Text Selection Tooltip */}
      {selectedText && tooltipPosition && (
        <TextSelectionTooltip
          position={tooltipPosition}
          onAsk={handleAskAboutSelection}
          onClose={() => {
            setSelectedText(null);
            setTooltipPosition(null);
          }}
        />
      )}
    </div>
  );
}
