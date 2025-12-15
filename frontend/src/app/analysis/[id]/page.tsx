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
  IconClose,
  IconInfo,
  Logo,
} from "@/components/icons";
import { AIAvatar, AIAvatarSmall } from "@/components/ai-avatar";
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
}

function RiskClauseItem({ clause }: RiskClauseItemProps) {
  const [expanded, setExpanded] = useState(false);

  const getLevelIcon = (level: string) => {
    const l = level.toLowerCase();
    if (l === "high") return <IconDanger size={16} className="text-[#c94b45]" />;
    if (l === "medium") return <IconWarning size={16} className="text-[#d4a84d]" />;
    return <IconCheck size={16} className="text-[#4a9a5b]" />;
  };

  const getLevelColor = (level: string) => {
    const l = level.toLowerCase();
    if (l === "high") return "text-[#b54a45]";
    if (l === "medium") return "text-[#9a7b2d]";
    return "text-[#3d7a4a]";
  };

  const getLevelBg = (level: string) => {
    const l = level.toLowerCase();
    if (l === "high") return "bg-[#fdedec]";
    if (l === "medium") return "bg-[#fef7e0]";
    return "bg-[#e8f5ec]";
  };

  return (
    <div className="card-apple overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 text-left flex items-start gap-3 hover:bg-gray-50/50 transition-colors"
      >
        <div className={cn(
          "flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center",
          getLevelBg(clause.level)
        )}>
          {getLevelIcon(clause.level)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {clause.clauseNumber && (
              <span className="text-[11px] font-medium text-gray-400">
                {clause.clauseNumber}
              </span>
            )}
            <span className={cn("text-[11px] font-semibold", getLevelColor(clause.level))}>
              {clause.level.toUpperCase()}
            </span>
          </div>
          <p className="text-sm text-gray-800 leading-relaxed line-clamp-2">{clause.text}</p>
        </div>
        <span className={cn(
          "flex-shrink-0 mt-1 transition-transform duration-200",
          expanded && "rotate-90"
        )}>
          <IconChevronRight size={16} className="text-gray-300" />
        </span>
      </button>

      <div className={cn(
        "overflow-hidden transition-all duration-300",
        expanded ? "max-h-[800px] opacity-100" : "max-h-0 opacity-0"
      )}>
        <div className="px-4 pb-4 space-y-4">
          {/* 위험 사유 - 가장 먼저, 왜 문제인지 설명 */}
          {clause.explanation && (
            <div className="bg-[#f8f7f4] rounded-[14px] p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-[6px] bg-gray-200/80 flex items-center justify-center">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gray-500">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                    <path d="M12 16V12M12 8H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
                <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">위험 사유</p>
              </div>
              <p className="text-[13px] text-gray-700 leading-relaxed">{clause.explanation}</p>
            </div>
          )}

          {/* 수정 제안 - 액션 가능한 항목으로 강조 */}
          {clause.suggestion && (
            <div className="bg-[#e8f5ec]/80 border border-[#c8e6cf] rounded-[14px] p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-[6px] bg-[#c8e6cf] flex items-center justify-center">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-[#3d7a4a]">
                    <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
                <p className="text-[11px] font-semibold text-[#3d7a4a] uppercase tracking-wider">수정 제안</p>
              </div>
              <p className="text-[13px] text-gray-700 leading-relaxed">{clause.suggestion}</p>
            </div>
          )}

          {/* 법적 근거 - 참고 정보 */}
          {clause.legalBasis && (
            <div className="bg-blue-50/60 border border-blue-100/80 rounded-[14px] p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-[6px] bg-blue-100 flex items-center justify-center">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-blue-600">
                    <path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <p className="text-[11px] font-semibold text-blue-600 uppercase tracking-wider">법적 근거</p>
              </div>
              <p className="text-[13px] text-gray-700 leading-relaxed">{clause.legalBasis}</p>
            </div>
          )}

          {/* 참조 출처 - 부가 정보 */}
          {clause.sources && clause.sources.length > 0 && (
            <div className="pt-2">
              <p className="text-[10px] font-medium text-gray-400 uppercase tracking-wider mb-2">참조 출처</p>
              <div className="flex flex-wrap gap-1.5">
                {clause.sources.map((source, idx) => (
                  <span
                    key={idx}
                    className="text-[10px] bg-gray-100/80 px-2 py-0.5 rounded-[4px] text-gray-500"
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

// 스크롤 인디케이터 컴포넌트
interface ScrollableAreaProps {
  children: React.ReactNode;
  className?: string;
  onMouseUp?: () => void;
}

function ScrollableArea({ children, className, onMouseUp }: ScrollableAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showIndicator, setShowIndicator] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const checkScroll = () => {
      // 스크롤 가능한지 확인 (콘텐츠가 컨테이너보다 큰 경우만)
      const isScrollable = el.scrollHeight > el.clientHeight;
      // 아래에 더 스크롤할 내용이 있는지 확인
      const hasMoreBelow = el.scrollHeight - el.scrollTop - el.clientHeight > 20;
      setShowIndicator(isScrollable && hasMoreBelow);
    };

    // 초기 체크는 약간의 딜레이 후 (콘텐츠 로딩 대기)
    const timer = setTimeout(checkScroll, 100);
    el.addEventListener("scroll", checkScroll);
    window.addEventListener("resize", checkScroll);

    // MutationObserver로 콘텐츠 변경 감지
    const observer = new MutationObserver(checkScroll);
    observer.observe(el, { childList: true, subtree: true });

    return () => {
      clearTimeout(timer);
      el.removeEventListener("scroll", checkScroll);
      window.removeEventListener("resize", checkScroll);
      observer.disconnect();
    };
  }, []);

  return (
    <div className="relative flex-1 min-h-0">
      <div
        ref={scrollRef}
        className={cn("h-full overflow-auto scrollable-area", className)}
        onMouseUp={onMouseUp}
      >
        {children}
      </div>
      {/* Scroll indicator - subtle bottom fade with arrow */}
      <div
        className={cn(
          "absolute bottom-0 left-0 right-0 h-12 pointer-events-none transition-opacity duration-300",
          showIndicator ? "opacity-100" : "opacity-0"
        )}
      >
        <div className="absolute inset-0 bg-gradient-to-t from-[#fafafa] to-transparent" />
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex flex-col items-center gap-0.5">
          <div className="w-6 h-6 rounded-full bg-white/90 shadow-sm border border-gray-200/60 flex items-center justify-center">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gray-400">
              <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
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
  const [inputFocused, setInputFocused] = useState(false);
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

  // Quick prompt cards data
  const quickPrompts = [
    { icon: "shield", text: "이 계약서의 주요 위험 요소는?", color: "red" },
    { icon: "scale", text: "위약금 조항이 적법한가요?", color: "amber" },
    { icon: "document", text: "노동청에 신고하려면 어떻게 해야 하나요?", color: "blue" },
  ];

  return (
    <div className="flex flex-col h-full animate-slideInRight safe-area-inset bg-[#fafafa]">
      {/* Header */}
      <div className="relative bg-[#fafafa] pt-3 pb-4 px-4">
        <div className="flex items-center justify-between">
          {/* Back Button */}
          <button
            onClick={onClose}
            className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center text-gray-600 hover:bg-gray-50 hover:shadow-md active:scale-95 transition-all duration-200"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          {/* Center - Avatar + Name */}
          <div className="flex items-center gap-3 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-sm">
            <AIAvatar size={36} isThinking={isStreaming} isSpeaking={!!streamingContent} />
            <span className="text-base font-semibold text-gray-900 tracking-tight pr-1">AI Assistant</span>
          </div>

          {/* Menu Button */}
          <button
            className="w-12 h-12 bg-white rounded-full shadow-sm flex items-center justify-center text-gray-600 hover:bg-gray-50 hover:shadow-md active:scale-95 transition-all duration-200"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <circle cx="12" cy="6" r="1.5" />
              <circle cx="12" cy="12" r="1.5" />
              <circle cx="12" cy="18" r="1.5" />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 scrollable-area">
        {messages.length === 0 && !isStreaming && (
          <div className="animate-fadeIn">
            {/* Welcome Section */}
            <div className="text-center py-6">
              {/* AI Sphere */}
              <div className="relative inline-flex items-center justify-center mb-6 ai-sphere-container cursor-pointer">
                <img
                  src="/ai-sphere-transparent.svg"
                  alt="AI"
                  className="w-24 h-24 animate-ai-sphere"
                />
              </div>

              <h3 className="text-base font-semibold text-gray-900 tracking-tight mb-1">
                무엇이든 물어보세요
              </h3>
              <p className="text-sm text-gray-500 tracking-tight max-w-[240px] mx-auto">
                AI가 법령과 판례를 검색하여 계약서를 분석합니다
              </p>
            </div>

            {/* Quick Prompts - Card Style */}
            <div className="space-y-2.5 mt-6">
              <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider px-1">추천 질문</p>
              {quickPrompts.map((prompt, idx) => (
                <button
                  key={prompt.text}
                  onClick={() => sendMessage(prompt.text)}
                  className="group w-full text-left p-3.5 bg-white rounded-[14px] border border-gray-100 hover:border-gray-200 hover:shadow-md transition-all duration-300 transform hover:-translate-y-0.5"
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn(
                      "w-9 h-9 rounded-[10px] flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-110",
                      prompt.color === "red" && "bg-[#fdedec] text-[#c94b45]",
                      prompt.color === "amber" && "bg-[#fef7e0] text-[#d4a84d]",
                      prompt.color === "blue" && "bg-gray-100 text-gray-600"
                    )}>
                      {prompt.icon === "shield" && (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                      )}
                      {prompt.icon === "scale" && (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 3v18M3 7l3 3-3 3M21 7l-3 3 3 3M6 21h12"/>
                        </svg>
                      )}
                      {prompt.icon === "document" && (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                          <polyline points="14 2 14 8 20 8"/>
                          <line x1="16" y1="13" x2="8" y2="13"/>
                          <line x1="16" y1="17" x2="8" y2="17"/>
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-700 font-medium tracking-tight group-hover:text-gray-900 transition-colors">
                        {prompt.text}
                      </p>
                    </div>
                    <IconChevronRight size={16} className="text-gray-300 group-hover:text-gray-400 group-hover:translate-x-0.5 transition-all flex-shrink-0 mt-0.5" />
                  </div>
                </button>
              ))}
            </div>

            {/* Capabilities hint */}
            <div className="mt-6 p-4 bg-gray-50 rounded-[14px] border border-gray-100">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded-[6px] bg-[#3d5a47] flex items-center justify-center">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                  </svg>
                </div>
                <p className="text-xs font-semibold text-gray-700 tracking-tight">AI 분석 기능</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {["법령 검색", "판례 분석", "위험 평가", "수정 제안"].map((cap) => (
                  <div key={cap} className="flex items-center gap-1.5 text-[11px] text-gray-500">
                    <span className="w-1 h-1 rounded-full bg-[#3d5a47]" />
                    {cap}
                  </div>
                ))}
              </div>
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
              <div className="max-w-[85%] px-4 py-3 bg-[#3d5a47] text-white text-sm rounded-[16px] rounded-br-[4px] shadow-sm">
                {msg.content}
              </div>
            ) : (
              <div className="max-w-full flex gap-2.5">
                {/* AI Avatar */}
                <AIAvatarSmall size={28} className="mt-1" />
                <div className="flex-1 min-w-0">
                  <div className="prose prose-sm prose-gray max-w-none bg-white rounded-[16px] rounded-tl-[4px] px-4 py-3 shadow-sm border border-gray-100">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
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
              <div className="rounded-[14px] bg-gray-50 border border-gray-100 p-4 space-y-2.5">
                {currentStep && (
                  <div className="flex items-center gap-2.5 text-xs text-gray-700 font-medium">
                    <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                    {currentStep}
                  </div>
                )}
                {toolStatuses.map((tool, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "flex items-center gap-2.5 text-xs transition-all duration-500",
                      tool.status === "complete" ? "text-[#3d7a4a]" : "text-gray-600"
                    )}
                  >
                    <div className={cn(
                      "w-5 h-5 rounded-full flex items-center justify-center transition-all duration-300",
                      tool.status === "complete" ? "bg-[#e8f5ec]" : "bg-gray-200"
                    )}>
                      {tool.status === "searching" ? (
                        <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <IconCheck size={12} />
                      )}
                    </div>
                    <span className="font-medium">{tool.message}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Streaming content */}
            {streamingContent && (
              <div className="max-w-full flex gap-2.5">
                <AIAvatarSmall size={28} isThinking={true} className="mt-1" />
                <div className="flex-1 min-w-0">
                  <div className="prose prose-sm prose-gray max-w-none bg-white rounded-[16px] rounded-tl-[4px] px-4 py-3 shadow-sm border border-gray-100">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {streamingContent}
                    </ReactMarkdown>
                    <span className="inline-block w-2 h-4 bg-gray-400 rounded-[2px] animate-pulse ml-0.5" />
                  </div>
                </div>
              </div>
            )}

            {/* Typing indicator when no content yet */}
            {!streamingContent && !currentStep && toolStatuses.length === 0 && (
              <div className="flex gap-2.5">
                <AIAvatarSmall size={28} isThinking={true} />
                <div className="bg-white rounded-[16px] rounded-tl-[4px] px-4 py-3 shadow-sm border border-gray-100">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 sm:p-5 bg-[#fafafa] pb-safe">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="relative"
        >
          {/* Premium Input Card */}
          <div className={cn(
            "bg-white rounded-[20px] shadow-sm transition-all duration-300",
            inputFocused && "shadow-md ring-1 ring-gray-200"
          )}>
            {/* Input Area */}
            <div className="px-5 py-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder="무엇이든 물어보세요..."
                className="w-full bg-transparent text-base outline-none placeholder:text-gray-400 text-gray-800"
                disabled={isStreaming}
              />
            </div>

            {/* Bottom Bar */}
            <div className="flex items-center justify-between px-4 pb-3">
              {/* Left Icons */}
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-[10px] transition-colors"
                  title="법령 검색"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
                    <path d="M2 12h20"/>
                  </svg>
                </button>
                <button
                  type="button"
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-[10px] transition-colors"
                  title="판례 검색"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                </button>
                <button
                  type="button"
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-[10px] transition-colors"
                  title="음성 입력"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2 10v3a7 7 0 0 0 14 0v-3"/>
                    <line x1="9" y1="16" x2="9" y2="21"/>
                    <line x1="5" y1="21" x2="13" y2="21"/>
                    <rect x="5" y="3" width="8" height="13" rx="4"/>
                  </svg>
                </button>
              </div>

              {/* Send Button */}
              {isStreaming ? (
                <button
                  type="button"
                  onClick={stopGeneration}
                  className="w-11 h-11 bg-[#c94b45] text-white rounded-full shadow-lg hover:bg-[#b54a45] hover:scale-105 active:scale-95 transition-all duration-200 flex items-center justify-center flex-shrink-0"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className={cn(
                    "w-11 h-11 rounded-full shadow-lg transition-all duration-200 flex items-center justify-center flex-shrink-0",
                    input.trim()
                      ? "bg-[#3d5a47] text-white hover:bg-[#4a6b52] hover:scale-105 active:scale-95 hover:shadow-xl"
                      : "bg-[#3d5a47]/40 text-white/70 cursor-not-allowed"
                  )}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {/* Powered by hint */}
          <div className="flex items-center justify-center gap-1.5 mt-3">
            <span className="text-[10px] text-gray-500/70">Powered by</span>
            <Logo size={12} color="#6b7280" className="opacity-60" />
            <span className="text-[10px] text-gray-500/70">DocScanner AI</span>
          </div>
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
      <div className="min-h-[100dvh] flex items-center justify-center bg-[#fafafa]">
        <div className="flex flex-col items-center gap-3 animate-fadeIn">
          <IconLoading size={32} className="text-gray-400" />
          <p className="text-sm text-gray-500 tracking-tight">분석 결과를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !contract) {
    return (
      <div className="min-h-[100dvh] bg-[#fafafa]">
        <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200/80 sticky top-0 z-10">
          <div className="px-4 sm:px-5 h-14 flex items-center">
            <Link href="/" className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 transition-colors min-h-[44px]">
              <IconArrowLeft size={16} />
              <span className="tracking-tight">돌아가기</span>
            </Link>
          </div>
        </header>
        <main className="px-4 sm:px-8 py-12 sm:py-16 text-center animate-fadeIn">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-[#fdedec] rounded-2xl mb-4">
            <IconDanger size={28} className="text-[#c94b45]" />
          </div>
          <p className="text-[#b54a45] font-medium tracking-tight">{error || "계약서를 찾을 수 없습니다"}</p>
        </main>
      </div>
    );
  }

  const analysis = contract.analysis_result;
  const riskClauses = normalizeRiskClauses(analysis);
  const summary = analysis?.analysis_summary || analysis?.summary;

  return (
    <div className={cn("h-[100dvh] bg-[#fafafa] flex flex-col", isResizing && "select-none cursor-ew-resize")}>
      {/* Main Content */}
      <div
        className="flex-1 flex flex-col min-h-0 transition-all duration-300"
        style={{ marginRight: showChat && !isMobile ? chatWidth : 0 }}
      >
        <header className="bg-[#fafafa] flex-shrink-0 pt-3">
          <div className="px-4 sm:px-6 h-12 flex items-center justify-between gap-4">
            {/* Left: Back + Title */}
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <Link
                href="/"
                className="flex items-center justify-center w-9 h-9 bg-white hover:bg-gray-100 rounded-[10px] transition-all duration-200 flex-shrink-0 group shadow-sm border border-gray-200/60"
              >
                <IconArrowLeft size={16} className="text-gray-600 group-hover:text-gray-900 transition-colors" />
              </Link>
              <div className="min-w-0 flex-1">
                <h1 className="text-sm font-semibold text-gray-900 truncate tracking-tight">
                  {contract.title}
                </h1>
              </div>
            </div>

            {/* Right: Risk Badge */}
            <div className="flex items-center gap-3 flex-shrink-0">
              {analysis?.risk_level && <RiskLevelBadge level={analysis.risk_level} />}
            </div>
          </div>
        </header>

        {/* Mobile View Switcher */}
        <div className="md:hidden flex items-center justify-center p-3 bg-[#fafafa] flex-shrink-0">
          <div className="flex items-center gap-1 bg-white rounded-xl p-1 shadow-sm border border-gray-200/60">
            <button
              onClick={() => setMobileView("pdf")}
              className={cn(
                "px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 tracking-tight min-w-[100px]",
                mobileView === "pdf"
                  ? "bg-[#3d5a47] text-white shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              원본 문서
            </button>
            <button
              onClick={() => setMobileView("analysis")}
              className={cn(
                "px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 tracking-tight min-w-[100px]",
                mobileView === "analysis"
                  ? "bg-[#3d5a47] text-white shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              분석 결과
            </button>
          </div>
        </div>

        <div className="flex-1 flex animate-fadeIn min-h-0">
          {/* Left Panel - PDF Viewer (Desktop: always visible, Mobile: conditional) */}
          <div className={cn(
            "flex-col min-h-0",
            // Desktop: 50% width with subtle shadow separator
            "md:flex md:w-1/2 md:shadow-[1px_0_0_0_rgba(0,0,0,0.06)]",
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
            "flex-col bg-[#fafafa] min-h-0 relative",
            // Desktop: 50% width
            "md:flex md:w-1/2",
            // Mobile: full width or hidden
            mobileView === "analysis" ? "flex w-full" : "hidden"
          )}>
            <div className="bg-[#fafafa] h-12 flex items-center px-4 flex-shrink-0">
              <div className="flex items-center gap-1 bg-white/80 rounded-[10px] p-0.5 border border-gray-200/60">
                {[
                  { key: "overview", label: "개요" },
                  { key: "clauses", label: `위험 조항 (${riskClauses.length})` },
                  { key: "text", label: "텍스트" },
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key as typeof activeTab)}
                    className={cn(
                      "px-3 py-1.5 text-[11px] font-medium rounded-[8px] transition-all duration-200 whitespace-nowrap tracking-tight",
                      activeTab === tab.key
                        ? "bg-[#3d5a47] text-white"
                        : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <ScrollableArea className="p-4 sm:p-6" onMouseUp={handleRightPanelTextSelect}>
              {activeTab === "overview" && (
                <div className="space-y-6 animate-fadeIn">
                  {/* Risk Level Hero */}
                  <div className="card-apple p-5">
                    <div className="flex items-start gap-4">
                      <div className={cn(
                        "w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0",
                        analysis?.risk_level?.toLowerCase() === "high" ? "bg-[#fdedec]" :
                        analysis?.risk_level?.toLowerCase() === "medium" ? "bg-[#fef7e0]" : "bg-[#e8f5ec]"
                      )}>
                        {analysis?.risk_level?.toLowerCase() === "high" ? (
                          <IconDanger size={28} className="text-[#c94b45]" />
                        ) : analysis?.risk_level?.toLowerCase() === "medium" ? (
                          <IconWarning size={28} className="text-[#d4a84d]" />
                        ) : (
                          <IconCheck size={28} className="text-[#4a9a5b]" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={cn(
                            "text-xs font-semibold px-2 py-0.5 rounded-full",
                            analysis?.risk_level?.toLowerCase() === "high" ? "bg-[#fdedec] text-[#b54a45]" :
                            analysis?.risk_level?.toLowerCase() === "medium" ? "bg-[#fef7e0] text-[#9a7b2d]" : "bg-[#e8f5ec] text-[#3d7a4a]"
                          )}>
                            {analysis?.risk_level?.toUpperCase() || "N/A"} RISK
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 leading-relaxed">
                          {summary || "분석 결과를 확인하세요."}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Stats Row */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="card-static p-4">
                      <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-2">발견된 이슈</p>
                      <p className="text-2xl font-bold text-gray-900 tracking-tight">{riskClauses.length}</p>
                    </div>
                    <div className="card-static p-4">
                      <p className="text-[11px] font-medium text-[#b54a45] uppercase tracking-wider mb-2">고위험</p>
                      <p className="text-2xl font-bold text-[#c94b45] tracking-tight">
                        {riskClauses.filter((c: NormalizedClause) => c.level.toLowerCase() === "high").length}
                      </p>
                    </div>
                    <div className="card-static p-4">
                      <p className="text-[11px] font-medium text-[#9a7b2d] uppercase tracking-wider mb-2">주의</p>
                      <p className="text-2xl font-bold text-[#d4a84d] tracking-tight">
                        {riskClauses.filter((c: NormalizedClause) => c.level.toLowerCase() === "medium").length}
                      </p>
                    </div>
                  </div>

                  {/* 체불 예상액 - More prominent */}
                  {analysis?.stress_test && (analysis.stress_test.annual_underpayment || 0) > 0 && (
                    <div className="card-apple p-5 bg-gradient-to-br from-[#fdedec] to-white border-[#f5c6c4]">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-xl bg-[#fdedec] flex items-center justify-center">
                          <IconWarning size={18} className="text-[#c94b45]" />
                        </div>
                        <h3 className="text-sm font-semibold text-gray-900">예상 체불 금액</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-1">월간</p>
                          <p className="text-xl font-bold text-[#b54a45] tracking-tight">
                            {(analysis.stress_test.total_underpayment || 0).toLocaleString()}
                            <span className="text-sm font-medium ml-0.5">원</span>
                          </p>
                        </div>
                        <div>
                          <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider mb-1">연간</p>
                          <p className="text-xl font-bold text-[#b54a45] tracking-tight">
                            {(analysis.stress_test.annual_underpayment || 0).toLocaleString()}
                            <span className="text-sm font-medium ml-0.5">원</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 주요 이슈 */}
                  {riskClauses.length > 0 && (
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-semibold text-gray-900 tracking-tight">주요 이슈</h3>
                        {riskClauses.length > 3 && (
                          <button
                            onClick={() => setActiveTab("clauses")}
                            className="text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors flex items-center gap-0.5"
                          >
                            전체 보기
                            <IconChevronRight size={14} />
                          </button>
                        )}
                      </div>
                      <div className="space-y-3">
                        {riskClauses.slice(0, 3).map((clause, i) => (
                          <RiskClauseItem
                            key={i}
                            clause={clause}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 법률 참조 */}
                  {analysis?.crag_result?.final_answer && (
                    <div className="card-apple p-5">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">
                          <IconInfo size={18} className="text-blue-500" />
                        </div>
                        <h3 className="text-sm font-semibold text-gray-900">법률 참조</h3>
                      </div>
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {analysis.crag_result.final_answer}
                      </p>
                    </div>
                  )}

                </div>
              )}

              {activeTab === "clauses" && (
                <div className="animate-fadeIn">
                  {riskClauses.length === 0 ? (
                    <div className="text-center py-16">
                      <div className="inline-flex items-center justify-center w-16 h-16 bg-[#e8f5ec] rounded-2xl mb-4">
                        <IconCheck size={32} className="text-[#4a9a5b]" />
                      </div>
                      <p className="text-base font-semibold text-gray-900 tracking-tight">위험 조항 없음</p>
                      <p className="text-sm text-gray-500 mt-1">이 계약서는 안전한 것으로 보입니다</p>
                    </div>
                  ) : (
                    <>
                      {/* 정렬 옵션 */}
                      <div className="flex items-center justify-between mb-4">
                        <p className="text-sm text-gray-500">
                          총 <span className="font-semibold text-gray-900">{riskClauses.length}</span>개 이슈
                        </p>
                        <div className="flex items-center gap-0.5 bg-white rounded-full p-0.5 border border-gray-200">
                          {[
                            { key: "default", label: "기본" },
                            { key: "risk", label: "위험도" },
                            { key: "clause", label: "조항" },
                          ].map((option) => (
                            <button
                              key={option.key}
                              onClick={() => setSortBy(option.key as typeof sortBy)}
                              className={cn(
                                "px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200",
                                sortBy === option.key
                                  ? "bg-[#3d5a47] text-white"
                                  : "text-gray-500 hover:text-gray-700"
                              )}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      {/* 정렬된 위험 조항 목록 */}
                      <div className="space-y-3">
                        {sortRiskClauses(riskClauses).map((clause, i) => (
                          <RiskClauseItem
                            key={i}
                            clause={clause}
                          />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {activeTab === "text" && (
                <div className="card-apple p-5 animate-fadeIn">
                  <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {contract.extracted_text || "추출된 텍스트가 없습니다"}
                  </p>
                </div>
              )}
            </ScrollableArea>

            {/* Tab Indicator - Bottom Right */}
            <div className="absolute bottom-4 right-4 flex items-center gap-1.5 bg-white/90 backdrop-blur-sm rounded-full px-2 py-1.5 border border-gray-200/60 shadow-sm">
              {[
                { key: "overview", label: "개요" },
                { key: "clauses", label: "위험 조항" },
                { key: "text", label: "텍스트" },
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as typeof activeTab)}
                  className={cn(
                    "w-2 h-2 rounded-full transition-all duration-200",
                    activeTab === tab.key
                      ? "bg-[#3d5a47] scale-110"
                      : "bg-gray-300 hover:bg-gray-400"
                  )}
                  title={tab.label}
                />
              ))}
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

      {/* Floating AI Button */}
      {!showChat && (
        <button
          onClick={() => setShowChat(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-white border border-gray-200/80 rounded-full shadow-lg hover:shadow-xl hover:scale-105 active:scale-95 transition-all duration-200 z-30 flex items-center justify-center"
          title="AI에게 질문하기"
        >
          <AIAvatarSmall size={32} />
        </button>
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
