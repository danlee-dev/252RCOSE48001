"use client";

import { useState, useEffect, useCallback, use, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
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
  IconClose,
  IconInfo,
} from "@/components/icons";
import { AIAvatar, AIAvatarSmall } from "@/components/ai-avatar";
import { cn } from "@/lib/utils";

// PDF 뷰어는 클라이언트 사이드에서만 로드
const PDFViewer = dynamic(
  () => import("@/components/pdf-viewer").then((mod) => mod.PDFViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full bg-white">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-500">PDF 뷰어 로딩 중...</p>
        </div>
      </div>
    ),
  }
);

// HighlightClause 타입 import
import type { HighlightClause } from "@/components/pdf-viewer";

interface AnalysisPageProps {
  params: Promise<{
    id: string;
  }>;
}

function RiskLevelBadge({ level }: { level: string }) {
  const normalizedLevel = level.toLowerCase();

  if (normalizedLevel === "high" || normalizedLevel === "danger") {
    return (
      <span className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-[10px] bg-[#fdedec] text-[#b54a45] border border-[#f5c6c4]">
        <IconDanger size={16} />
        High Risk
      </span>
    );
  }
  if (normalizedLevel === "medium" || normalizedLevel === "warning") {
    return (
      <span className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-[10px] bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]">
        <IconWarning size={16} />
        Medium Risk
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-[10px] bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]">
      <IconCheck size={16} />
      Low Risk
    </span>
  );
}

// Normalized clause item that works with both old and new data structures
interface NormalizedClause {
  id: string;                 // 고유 식별자 (하이라이팅 연동용)
  text: string;
  level: string;
  explanation?: string;
  suggestion?: string;
  suggestedText?: string;     // 수정된 조항 텍스트 (대체용)
  clauseNumber?: string;      // V2: 조항 번호
  sources?: string[];         // V2: CRAG 검색 출처
  legalBasis?: string;        // V2: 법적 근거
  originalText?: string;      // 원본 계약서에서 매칭할 텍스트
  matchedText?: string;       // 하이라이팅할 실제 텍스트 (텍스트 기반 매칭용)
  startIndex?: number;        // 원본 텍스트에서 시작 위치
  endIndex?: number;          // 원본 텍스트에서 끝 위치
}

interface RiskClauseItemProps {
  clause: NormalizedClause;
  isActive?: boolean;
  onClauseClick?: (clause: NormalizedClause) => void;
}

function RiskClauseItem({ clause, isActive, onClauseClick }: RiskClauseItemProps) {
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

  const handleClick = () => {
    const willExpand = !expanded;
    setExpanded(willExpand);

    // 토글이 열릴 때만 왼쪽 문서를 해당 하이라이트로 스크롤
    if (willExpand) {
      // matchedText 또는 유효한 인덱스가 있는 경우에만 하이라이팅 지원
      const hasValidHighlight =
        (clause.startIndex !== undefined && clause.startIndex >= 0) ||
        (clause.matchedText && clause.matchedText.length >= 5);
      if (onClauseClick && hasValidHighlight) {
        onClauseClick(clause);
      }
    }
  };

  return (
    <div className={cn(
      "card-apple overflow-hidden transition-all duration-200",
      isActive && "ring-2 ring-[#3d5a47] ring-offset-2"
    )}>
      <button
        onClick={handleClick}
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
              <span className="text-sm font-medium text-gray-400">
                {clause.clauseNumber}
              </span>
            )}
            <span className={cn("text-sm font-semibold", getLevelColor(clause.level))}>
              {clause.level.toUpperCase()}
            </span>
          </div>
          <p className="text-lg text-gray-800 leading-relaxed line-clamp-2">{clause.text}</p>
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
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider">위험 사유</p>
              </div>
              <div className="text-base text-gray-700 leading-relaxed prose prose-sm max-w-none prose-strong:text-gray-900 prose-strong:font-semibold">
                <ReactMarkdown>{clause.explanation}</ReactMarkdown>
              </div>
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
                <p className="text-sm font-semibold text-[#3d7a4a] uppercase tracking-wider">수정 제안</p>
              </div>
              <div className="text-base text-gray-700 leading-relaxed prose prose-sm max-w-none prose-strong:text-[#2d5a3a] prose-strong:font-semibold">
                <ReactMarkdown>{clause.suggestion}</ReactMarkdown>
              </div>
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
                <p className="text-sm font-semibold text-blue-600 uppercase tracking-wider">법적 근거</p>
              </div>
              <p className="text-base text-gray-700 leading-relaxed">{clause.legalBasis}</p>
            </div>
          )}

          {/* 참조 출처 - 부가 정보 */}
          {clause.sources && clause.sources.length > 0 && (
            <div className="pt-2">
              <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">참조 출처</p>
              <div className="flex flex-wrap gap-1.5">
                {clause.sources.map((source, idx) => (
                  <span
                    key={idx}
                    className="text-sm bg-gray-100/80 px-2.5 py-1 rounded-[6px] text-gray-500"
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
    analysis.stress_test.violations.forEach((v: StressTestViolation, idx: number) => {
      const severity = v.severity?.toUpperCase();
      allClauses.push({
        id: `stress-${idx}-${v.clause_number || idx}`,
        text: v.type || "법적 기준 위반",
        level: severity === "CRITICAL" || severity === "HIGH" ? "High" : severity === "MEDIUM" ? "Medium" : "Low",
        explanation: v.description || `현재 값: ${v.current_value}, 법적 기준: ${v.legal_standard}`,
        suggestion: v.suggestion,
        suggestedText: v.suggested_text,
        clauseNumber: v.clause_number,
        sources: v.sources,
        legalBasis: v.legal_basis,
        // API에서 받은 원본 조항 텍스트 사용 (정확한 하이라이팅용)
        originalText: v.original_text || "",
        matchedText: v.matched_text || "",  // 텍스트 기반 하이라이팅용
        startIndex: v.start_index,
        endIndex: v.end_index,
      });
    });
  }

  // 2. Redlining changes - 계약서 조항의 불공정 분석
  // stress_test.violations가 있으면 redlining은 중복이므로 스킵
  // (pipeline.py에서 동일한 violations를 stress_test와 redlining 양쪽에 저장함)
  if (analysis.redlining?.changes && analysis.redlining.changes.length > 0 && allClauses.length === 0) {
    analysis.redlining.changes.forEach((change: RedliningChange, idx: number) => {
      allClauses.push({
        id: `redline-${idx}`,
        text: change.original || "",
        level: change.severity || "Medium",
        explanation: change.reason,
        suggestion: change.revised,
        originalText: change.original || "",
      });
    });
  }

  // 3. Fall back to legacy structure: risk_clauses
  if (allClauses.length === 0 && analysis.risk_clauses && analysis.risk_clauses.length > 0) {
    return analysis.risk_clauses.map((clause: RiskClause, idx: number) => ({
      id: `legacy-${idx}`,
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

// 슬라이딩 탭 컴포넌트
interface SlidingTabsProps<T extends string> {
  tabs: { key: T; label: string }[];
  activeTab: T;
  onTabChange: (tab: T) => void;
  size?: "default" | "small";
  className?: string;
}

function SlidingTabs<T extends string>({
  tabs,
  activeTab,
  onTabChange,
  size = "default",
  className,
}: SlidingTabsProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const activeIndex = tabs.findIndex((t) => t.key === activeTab);
    const buttons = container.querySelectorAll("button");
    const activeButton = buttons[activeIndex];

    if (activeButton) {
      const containerRect = container.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();
      setIndicatorStyle({
        left: buttonRect.left - containerRect.left,
        width: buttonRect.width,
      });
    }
  }, [activeTab, tabs]);

  const isSmall = size === "small";

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative flex items-center gap-0.5 bg-white/80 p-1 border border-gray-200/60",
        isSmall ? "rounded-full" : "rounded-[12px]",
        className
      )}
    >
      {/* Sliding indicator */}
      <div
        className={cn(
          "absolute bg-[#3d5a47] transition-all duration-300 ease-out",
          isSmall ? "rounded-full" : "rounded-[10px]"
        )}
        style={{
          left: indicatorStyle.left,
          width: indicatorStyle.width,
          top: 4,
          bottom: 4,
        }}
      />
      {/* Tab buttons */}
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onTabChange(tab.key)}
          className={cn(
            "relative z-10 font-medium transition-colors duration-200 whitespace-nowrap tracking-tight",
            isSmall ? "px-4 py-2 text-sm rounded-full" : "px-5 py-2.5 text-base rounded-[10px]",
            activeTab === tab.key
              ? "text-white"
              : "text-gray-500 hover:text-gray-700"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
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
        <div className="absolute inset-0 bg-gradient-to-t from-white/60 to-transparent" />
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
      className="fixed z-50"
      style={{
        left: position.x,
        top: position.y,
        transform: "translate(-50%, -100%)",
        marginTop: "-12px",
      }}
    >
      {/* Animated wrapper for scale effect */}
      <div className="animate-fadeScaleIn">
        {/* Liquid Glass Container */}
        <div className="relative px-2 py-2 rounded-[20px] overflow-hidden shadow-lg">
          {/* Glass background with blur - subtle green tint */}
          <div className="absolute inset-0 bg-[#e8f5ec]/85 backdrop-blur-xl" />
          {/* Gradient border effect */}
          <div className="absolute inset-0 rounded-[20px] border border-[#c8e6cf]/60" />
          {/* Top highlight for glass reflection */}
          <div className="absolute inset-x-0 top-0 h-1/2 bg-gradient-to-b from-white/40 to-transparent rounded-t-[20px]" />
          {/* Subtle inner shadow */}
          <div className="absolute inset-0 rounded-[20px] shadow-[inset_0_1px_2px_rgba(255,255,255,0.6),inset_0_-1px_2px_rgba(0,0,0,0.03)]" />
          {/* Outer glow */}
          <div className="absolute -inset-[1px] rounded-[21px] bg-gradient-to-b from-white/80 via-transparent to-black/5 -z-10 blur-[0.5px]" />

          {/* Content */}
          <div className="relative flex items-center gap-1">
            <button
              onClick={onAsk}
              className="flex items-center gap-2 px-3 py-2 rounded-[14px] hover:bg-white/50 transition-all duration-200 group"
            >
              <AIAvatarSmall size={24} />
              <span className="text-sm font-medium text-gray-700 group-hover:text-gray-900 tracking-tight">
                AI에게 질문하기
              </span>
            </button>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded-[10px] text-gray-400 hover:text-gray-600 hover:bg-white/50 transition-all duration-200"
            >
              <IconClose size={14} />
            </button>
          </div>
        </div>

        {/* Glass arrow with bridge */}
        <div className="absolute left-1/2 bottom-0 -translate-x-1/2 translate-y-[8px]">
          {/* Bridge to cover inner shadow */}
          <div className="absolute -top-[2px] left-1/2 -translate-x-1/2 w-5 h-[3px] bg-[#eaf5ed]" />
          {/* Arrow */}
          <div className="w-0 h-0 border-l-[10px] border-r-[10px] border-t-[10px] border-transparent border-t-[#eaf5ed]" />
        </div>
      </div>
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
  contractTitle: string;
  initialQuestion?: string;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  onClose: () => void;
}

function ChatPanel({ contractId, contractTitle, initialQuestion, messages, setMessages, onClose }: ChatPanelProps) {
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

  // Quick prompt data - simple text only
  const quickPrompts = [
    "이 계약서의 주요 위험 요소는 무엇인가",
    "위약금 조항이 적법한가요?",
    "노동청에 신고하려면 어떻게 해야 하나요?",
    "근로기준법 위반 사항이 있나요?",
    "계약서 수정이 필요한 부분은?",
  ];

  return (
    <div className="flex flex-col h-full animate-slideInRight safe-area-inset relative overflow-hidden">
      {/* Gradient Background */}
      <div className="absolute inset-0 bg-[#f8f9fa]" />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse 80% 60% at 5% 15%, rgba(220, 235, 224, 0.95) 0%, transparent 55%),
            radial-gradient(ellipse 60% 50% at 95% 85%, rgba(254, 243, 210, 0.7) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 60% 5%, rgba(220, 240, 226, 0.8) 0%, transparent 45%)
          `
        }}
      />
      {/* Grain Texture */}
      <div
        className="absolute inset-0 pointer-events-none opacity-30"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='2' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`
        }}
      />

      {/* Header */}
      <div className="relative z-10 pt-3 pb-4 px-4">
        <div className="flex items-center justify-between">
          {/* Back Button */}
          <button
            onClick={onClose}
            className="w-12 h-12 bg-white/80 backdrop-blur-sm rounded-full shadow-sm flex items-center justify-center text-gray-600 hover:bg-white hover:shadow-md active:scale-95 transition-all duration-200"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          {/* Center - Contract Title */}
          <div className="bg-white/80 backdrop-blur-sm px-5 py-2 rounded-full shadow-sm max-w-[220px]">
            <span className="text-sm font-medium text-gray-900 tracking-tight truncate block">{contractTitle}</span>
          </div>

          {/* Menu Button */}
          <button
            className="w-12 h-12 bg-white/80 backdrop-blur-sm rounded-full shadow-sm flex items-center justify-center text-gray-600 hover:bg-white hover:shadow-md active:scale-95 transition-all duration-200"
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
      <div className="relative z-10 flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 scrollable-area">
        {messages.length === 0 && !isStreaming && (
          <div className="animate-fadeIn">
            {/* Welcome Section with AIAvatar */}
            <div className="text-center py-8">
              <div className="inline-flex items-center justify-center mb-4">
                <AIAvatar size={64} isThinking={false} isSpeaking={false} />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 tracking-tight mb-1">
                무엇이든 물어보세요
              </h3>
              <p className="text-sm text-gray-500 tracking-tight">
                AI가 법령과 판례를 검색하여 분석합니다
              </p>
            </div>

            {/* Quick Prompts - Simple List Style */}
            <div className="space-y-0">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="group w-full text-left py-3.5 px-3 flex items-center gap-3 rounded-[14px] hover:bg-white/50 transition-colors"
                >
                  <span className="text-gray-400 flex-shrink-0">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="9 10 4 15 9 20" />
                      <path d="M20 4v7a4 4 0 0 1-4 4H4" />
                    </svg>
                  </span>
                  <span className="text-base text-gray-700 tracking-tight group-hover:text-gray-900 transition-colors">
                    {prompt}
                  </span>
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
      <div className="relative z-10 px-4 pb-3 pt-2 pb-safe">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="flex items-center gap-3"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-4 py-2.5 bg-white/40 backdrop-blur-xl rounded-full text-base outline-none ring-0 focus:outline-none focus:ring-0 placeholder:text-gray-400 text-gray-800 transition-all duration-300 border border-white/60 shadow-[inset_0_1px_2px_rgba(255,255,255,0.4),0_2px_8px_rgba(0,0,0,0.04)] focus:bg-white/60 focus:border-white/80 focus:shadow-[inset_0_1px_3px_rgba(255,255,255,0.5),0_4px_12px_rgba(0,0,0,0.06)] focus:scale-[1.02] origin-center"
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button
              type="button"
              onClick={stopGeneration}
              className="w-11 h-11 bg-[#c94b45] text-white rounded-full hover:bg-[#b54a45] active:scale-95 transition-all duration-200 flex items-center justify-center flex-shrink-0 shadow-sm"
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
                "w-11 h-11 rounded-full transition-all duration-200 flex items-center justify-center flex-shrink-0 shadow-sm",
                input.trim()
                  ? "bg-[#3d5a47] text-white hover:bg-[#4a6b52] active:scale-95"
                  : "bg-white/60 text-gray-400 cursor-not-allowed"
              )}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
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
  const searchParams = useSearchParams();
  const versionParam = searchParams.get("version");
  const [contract, setContract] = useState<ContractDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewingVersion, setViewingVersion] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "clauses" | "text">("overview");
  const [showChat, setShowChat] = useState(false);
  const [chatInitialQuestion, setChatInitialQuestion] = useState<string | undefined>();
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]); // Persistent chat messages
  const [chatWidth, setChatWidth] = useState(384); // Default: 24rem = 384px
  const [isResizing, setIsResizing] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(55); // 좌우 패널 비율 (%, 기본 55:45)
  const [isPanelResizing, setIsPanelResizing] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [sortBy, setSortBy] = useState<"default" | "risk" | "clause">("default");
  const [mobileView, setMobileView] = useState<"pdf" | "analysis">("analysis"); // Mobile view switcher
  const [isMobile, setIsMobile] = useState(false); // For SSR-safe mobile detection
  const [activeHighlightId, setActiveHighlightId] = useState<string | null>(null); // 현재 선택된 하이라이트
  const [documentText, setDocumentText] = useState<string>(""); // 현재 문서 텍스트 (수정 적용용)
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle"); // 저장 상태
  const saveStatusTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

      // 버전 정보 불러오기
      try {
        const versionData = await contractsApi.getVersions(parseInt(id));

        // URL에 version 파라미터가 있으면 해당 버전 로드, 없으면 현재 버전 로드
        const targetVersionNum = versionParam ? parseInt(versionParam) : null;
        const targetVersion = targetVersionNum
          ? versionData.versions.find(v => v.version_number === targetVersionNum)
          : versionData.versions.find(v => v.is_current);

        if (targetVersion && targetVersion.content) {
          setDocumentText(targetVersion.content);
          setViewingVersion(targetVersion.version_number);
          console.log(">>> [DEBUG] Loaded version:", targetVersion.version_number, targetVersion.is_current ? "(current)" : "(historical)");
        } else if (data.extracted_text) {
          setDocumentText(data.extracted_text);
          setViewingVersion(null);
          console.log(">>> [DEBUG] No version found, using extracted_text");
        }
      } catch {
        // 버전 조회 실패시 원본 텍스트 사용
        if (data.extracted_text) {
          setDocumentText(data.extracted_text);
          setViewingVersion(null);
        }
      }
      // DEBUG: 하이라이팅 디버그
      const violations = data.analysis_result?.stress_test?.violations || [];
      console.log(">>> [DEBUG] Total violations:", violations.length);
      violations.slice(0, 3).forEach((v: StressTestViolation, i: number) => {
        console.log(`>>> [DEBUG] Violation ${i + 1}:`, {
          type: v.type,
          start_index: v.start_index,
          end_index: v.end_index,
          original_text: v.original_text?.substring(0, 50),
        });
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "계약서를 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }

  // NormalizedClause를 HighlightClause로 변환
  const convertToHighlights = useCallback((clauses: NormalizedClause[]): HighlightClause[] => {
    console.log(">>> [DEBUG] Converting to highlights, total clauses:", clauses.length);

    // matchedText 또는 텍스트가 있는 조항만 필터링
    const filtered = clauses.filter(c => {
      const highlightText = c.matchedText || c.originalText || c.text;
      return highlightText && highlightText.length >= 5;
    });
    console.log(">>> [DEBUG] After filtering:", filtered.length, "clauses can be highlighted");

    return filtered.map(c => ({
      id: c.id,
      level: c.level,
      text: c.text,
      // Gemini에서 반환한 matchedText 우선, 없으면 originalText
      matchedText: c.matchedText || c.originalText || undefined,
      suggestedText: c.suggestedText,
      explanation: c.explanation,
      suggestion: c.suggestion,
      clauseNumber: c.clauseNumber,
    }));
  }, []);

  // 조항 클릭 핸들러 - 왼쪽 문서의 해당 하이라이트로 스크롤
  const handleClauseClick = useCallback((clause: NormalizedClause) => {
    setActiveHighlightId(clause.id);
    // 모바일에서는 PDF 뷰로 전환
    if (isMobile) {
      setMobileView("pdf");
    }
  }, [isMobile]);

  // 수정 적용 핸들러 (Gemini matchedText 기반)
  const handleApplyFix = useCallback(async (highlight: HighlightClause) => {
    if (!highlight.suggestedText || !documentText || !contract) return;

    // matchedText가 있으면 직접 사용 (Gemini에서 반환한 정확한 텍스트)
    const originalText = highlight.matchedText || highlight.text;
    if (!originalText) return;

    // 문서에서 해당 텍스트 찾기
    const startIndex = documentText.indexOf(originalText);

    if (startIndex === -1) {
      console.error(">>> Cannot find text to replace:", originalText.substring(0, 50));
      return;
    }

    const endIndex = startIndex + originalText.length;

    // 문서 텍스트에서 해당 부분을 수정안으로 교체
    const before = documentText.slice(0, startIndex);
    const after = documentText.slice(endIndex);
    const newText = before + highlight.suggestedText + after;

    // UI 먼저 업데이트 (낙관적 업데이트)
    setDocumentText(newText);
    setActiveHighlightId(null);
    setSaveStatus("saving");

    // 이전 타임아웃 정리
    if (saveStatusTimeoutRef.current) {
      clearTimeout(saveStatusTimeoutRef.current);
    }

    // 버전 관리 API 호출하여 수정 내용 저장
    try {
      const result = await contractsApi.createVersion(contract.id, {
        content: newText,
        changes: {
          clause_id: highlight.id,
          clause_number: highlight.clauseNumber,
          start_index: startIndex,
          end_index: endIndex,
          original_text: originalText,
          new_text: highlight.suggestedText,
          risk_level: highlight.level,
        },
        change_summary: `조항 수정: "${originalText.substring(0, 50)}${originalText.length > 50 ? '...' : ''}" -> "${highlight.suggestedText.substring(0, 50)}${highlight.suggestedText.length > 50 ? '...' : ''}"`,
        created_by: "user",
      });

      // 새 버전 번호로 즉시 업데이트
      if (result.version_number) {
        setViewingVersion(result.version_number);
      }

      setSaveStatus("saved");
      // 3초 후 저장됨 표시 숨기기
      saveStatusTimeoutRef.current = setTimeout(() => {
        setSaveStatus("idle");
      }, 3000);

      console.log(">>> Version saved successfully, new version:", result.version_number);
    } catch (err) {
      console.error(">>> Failed to save version:", err);
      setSaveStatus("idle");
      // API 실패해도 로컬 변경은 유지 (사용자 경험 우선)
    }
  }, [documentText, contract]);

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

  // 선택 해제 시 툴팁 자동 닫기
  useEffect(() => {
    if (!selectedText) return;

    const handleSelectionChange = () => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed || selection.toString().trim().length < 10) {
        setSelectedText(null);
        setTooltipPosition(null);
      }
    };

    document.addEventListener("selectionchange", handleSelectionChange);
    return () => document.removeEventListener("selectionchange", handleSelectionChange);
  }, [selectedText]);

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

  // 좌우 패널 리사이즈 핸들러
  const handlePanelResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsPanelResizing(true);

    const container = (e.target as HTMLElement).closest('.panel-container');
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const containerWidth = containerRect.width;

    const handleMouseMove = (e: MouseEvent) => {
      const relativeX = e.clientX - containerRect.left;
      const newLeftWidth = Math.min(Math.max((relativeX / containerWidth) * 100, 35), 70); // Min 35%, Max 70%
      setLeftPanelWidth(newLeftWidth);
    };

    const handleMouseUp = () => {
      setIsPanelResizing(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, []);

  if (loading) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center gradient-blob-bg">
        <div className="relative z-10 flex flex-col items-center gap-3 animate-fadeIn">
          <IconLoading size={32} className="text-gray-400" />
          <p className="text-sm text-gray-500 tracking-tight">분석 결과를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !contract) {
    return (
      <div className="min-h-[100dvh] gradient-blob-bg">
        <header className="relative z-10 bg-white/80 backdrop-blur-sm border-b border-gray-200/80 sticky top-0">
          <div className="px-4 sm:px-5 h-14 flex items-center">
            <Link href="/" className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 transition-colors min-h-[44px]">
              <IconArrowLeft size={16} />
              <span className="tracking-tight">돌아가기</span>
            </Link>
          </div>
        </header>
        <main className="relative z-10 px-4 sm:px-8 py-12 sm:py-16 text-center animate-fadeIn">
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
    <div className={cn("h-[100dvh] gradient-blob-bg flex flex-col", (isResizing || isPanelResizing) && "select-none cursor-ew-resize")}>
      {/* Main Content */}
      <div
        className="relative z-10 flex-1 flex flex-col min-h-0 transition-all duration-300"
        style={{ marginRight: showChat && !isMobile ? chatWidth : 0 }}
      >
        <header className="flex-shrink-0 pt-4">
          <div className="px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
            {/* Left: Back + Title */}
            <div className="flex items-center gap-4 min-w-0 flex-1">
              <button
                onClick={() => router.back()}
                className="flex items-center justify-center w-11 h-11 bg-white hover:bg-gray-100 rounded-[12px] transition-all duration-200 flex-shrink-0 group shadow-sm border border-gray-200/60"
              >
                <IconArrowLeft size={20} className="text-gray-600 group-hover:text-gray-900 transition-colors" />
              </button>
              <div className="min-w-0 flex-1">
                <h1 className="text-xl font-semibold text-gray-900 truncate tracking-tight">
                  {contract.title}
                </h1>
                {versionParam && viewingVersion && (
                  <div className="flex items-center gap-2 mt-1">
                    <span className="px-2 py-0.5 text-xs font-semibold rounded-[6px] bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]">
                      v{viewingVersion} 버전 보기
                    </span>
                    <Link
                      href={`/analysis/${id}`}
                      className="text-xs text-[#3d5a47] hover:underline font-medium"
                    >
                      최신 버전으로
                    </Link>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Risk Badge */}
            <div className="flex items-center gap-3 flex-shrink-0">
              {analysis?.risk_level && <RiskLevelBadge level={analysis.risk_level} />}
            </div>
          </div>
        </header>

        {/* Mobile View Switcher */}
        <div className="md:hidden flex items-center justify-center p-3 flex-shrink-0">
          <div className="flex items-center gap-1 bg-white rounded-[12px] p-1 shadow-sm border border-gray-200/60">
            <button
              onClick={() => setMobileView("pdf")}
              className={cn(
                "px-4 py-2 text-sm font-medium rounded-[10px] transition-all duration-200 tracking-tight min-w-[100px]",
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
                "px-4 py-2 text-sm font-medium rounded-[10px] transition-all duration-200 tracking-tight min-w-[100px]",
                mobileView === "analysis"
                  ? "bg-[#3d5a47] text-white shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              분석 결과
            </button>
          </div>
        </div>

        <div className="flex-1 flex animate-fadeIn min-h-0 panel-container">
          {/* Left Panel - PDF Viewer (Desktop: always visible, Mobile: conditional) */}
          <div
            className={cn(
              "flex-col min-h-0 transition-all duration-150",
              // Desktop: dynamic width with left padding
              "md:flex md:pl-6",
              // Mobile: full width or hidden
              mobileView === "pdf" ? "flex w-full" : "hidden"
            )}
            style={!isMobile ? { flexBasis: `${leftPanelWidth}%`, flexShrink: 0 } : undefined}
          >
            <PDFViewer
              fileUrl={contract.file_url}
              extractedText={documentText || contract.extracted_text}
              onTextSelect={(text, position) => {
                setSelectedText(text);
                setTooltipPosition(position);
              }}
              highlights={convertToHighlights(riskClauses)}
              activeHighlightId={activeHighlightId || undefined}
              onHighlightClick={(clause) => setActiveHighlightId(clause.id)}
              onApplyFix={handleApplyFix}
              viewingVersion={viewingVersion}
              saveStatus={saveStatus}
              className="flex-1"
            />
          </div>

          {/* Center Resize Handle (Desktop only) */}
          <div
            className="hidden md:flex items-center justify-center w-6 cursor-ew-resize group flex-shrink-0 relative"
            onMouseDown={handlePanelResizeStart}
          >
            {/* Resize handle bar */}
            <div className={cn(
              "w-1.5 h-16 rounded-full transition-all duration-200 relative",
              "bg-gray-300 group-hover:bg-[#3d5a47] group-hover:h-20",
              isPanelResizing && "bg-[#3d5a47] h-24"
            )}>
              {/* Grip dots */}
              <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1">
                <div className={cn("w-1 h-1 rounded-full bg-white/80 transition-opacity", isPanelResizing ? "opacity-100" : "opacity-0 group-hover:opacity-100")} />
                <div className={cn("w-1 h-1 rounded-full bg-white/80 transition-opacity", isPanelResizing ? "opacity-100" : "opacity-0 group-hover:opacity-100")} />
                <div className={cn("w-1 h-1 rounded-full bg-white/80 transition-opacity", isPanelResizing ? "opacity-100" : "opacity-0 group-hover:opacity-100")} />
              </div>
            </div>
          </div>

          {/* Right Panel - Analysis (Desktop: always visible, Mobile: conditional) */}
          <div
            className={cn(
              "flex-col min-h-0 relative flex-1",
              // Desktop: dynamic width with right padding
              "md:flex md:pr-6",
              // Mobile: full width or hidden
              mobileView === "analysis" ? "flex w-full" : "hidden"
            )}
          >
            <div className="h-14 flex items-center px-4 flex-shrink-0">
              <SlidingTabs
                tabs={[
                  { key: "overview" as const, label: "개요" },
                  { key: "clauses" as const, label: `위험 조항 (${riskClauses.length})` },
                  { key: "text" as const, label: "텍스트" },
                ]}
                activeTab={activeTab}
                onTabChange={(tab) => setActiveTab(tab)}
              />
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
                        <div className="text-lg text-gray-600 leading-relaxed prose prose-gray max-w-none prose-strong:text-gray-900 prose-strong:font-semibold">
                          <ReactMarkdown>
                            {summary || "분석 결과를 확인하세요."}
                          </ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Stats Row */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="card-static p-4">
                      <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">발견된 이슈</p>
                      <p className="text-3xl font-bold text-gray-900 tracking-tight">{riskClauses.length}</p>
                    </div>
                    <div className="card-static p-4">
                      <p className="text-sm font-medium text-[#b54a45] uppercase tracking-wider mb-2">고위험</p>
                      <p className="text-3xl font-bold text-[#c94b45] tracking-tight">
                        {riskClauses.filter((c: NormalizedClause) => c.level.toLowerCase() === "high").length}
                      </p>
                    </div>
                    <div className="card-static p-4">
                      <p className="text-sm font-medium text-[#9a7b2d] uppercase tracking-wider mb-2">주의</p>
                      <p className="text-3xl font-bold text-[#d4a84d] tracking-tight">
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
                        <h3 className="text-lg font-semibold text-gray-900">예상 체불 금액</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-1">월간</p>
                          <p className="text-2xl font-bold text-[#b54a45] tracking-tight">
                            {(analysis.stress_test.total_underpayment || 0).toLocaleString()}
                            <span className="text-base font-medium ml-0.5">원</span>
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-1">연간</p>
                          <p className="text-2xl font-bold text-[#b54a45] tracking-tight">
                            {(analysis.stress_test.annual_underpayment || 0).toLocaleString()}
                            <span className="text-base font-medium ml-0.5">원</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 분석 신뢰도 - LLM-as-a-Judge 결과 */}
                  {analysis?.judgment && (analysis.judgment.overall_score || analysis.judgment.confidence_level) && (
                    <div className="card-apple p-5">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-xl bg-[#e8f0ea] flex items-center justify-center">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-[#3d5a47]">
                            <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-gray-900 tracking-tight">분석 신뢰도</h3>
                        {/* 신뢰도 레벨 뱃지 */}
                        {analysis.judgment.confidence_level && (
                          <span className={cn(
                            "ml-auto text-xs font-semibold px-2.5 py-1 rounded-full",
                            analysis.judgment.confidence_level.toUpperCase() === "HIGH"
                              ? "bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]"
                              : analysis.judgment.confidence_level.toUpperCase() === "MEDIUM"
                              ? "bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]"
                              : "bg-[#fdedec] text-[#b54a45] border border-[#f5c6c4]"
                          )}>
                            {analysis.judgment.confidence_level.toUpperCase()}
                          </span>
                        )}
                      </div>

                      {/* 점수 표시 - 백엔드에서 0-1 스케일로 반환 */}
                      {analysis.judgment.overall_score !== undefined && (
                        (() => {
                          // 백엔드가 0-1 스케일이면 100을 곱해서 표시
                          const scorePercent = analysis.judgment.overall_score <= 1
                            ? analysis.judgment.overall_score * 100
                            : analysis.judgment.overall_score;
                          return (
                            <div className="mb-4">
                              <div className="flex items-baseline gap-2 mb-2">
                                <span className="text-3xl font-bold text-gray-900 tracking-tight">
                                  {scorePercent.toFixed(1)}
                                </span>
                                <span className="text-lg text-gray-400 font-medium">/ 100</span>
                              </div>
                              {/* 프로그레스 바 */}
                              <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                  className={cn(
                                    "h-full rounded-full transition-all duration-500",
                                    scorePercent >= 80
                                      ? "bg-gradient-to-r from-[#4a9a5b] to-[#3d7a4a]"
                                      : scorePercent >= 60
                                      ? "bg-gradient-to-r from-[#d4a84d] to-[#9a7b2d]"
                                      : "bg-gradient-to-r from-[#c94b45] to-[#b54a45]"
                                  )}
                                  style={{ width: `${Math.min(scorePercent, 100)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })()
                      )}

                      {/* 판정 결과 */}
                      {analysis.judgment.verdict && (
                        <p className="text-base text-gray-600 leading-relaxed mb-4">
                          {analysis.judgment.verdict}
                        </p>
                      )}

                      {/* 권장 사항 */}
                      {analysis.judgment.recommendations && analysis.judgment.recommendations.length > 0 && (
                        <div className="space-y-2">
                          {analysis.judgment.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start gap-2">
                              <div className={cn(
                                "w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5",
                                rec.startsWith("!") || rec.includes("주의") || rec.includes("권장")
                                  ? "bg-[#fef7e0]"
                                  : "bg-[#e8f5ec]"
                              )}>
                                {rec.startsWith("!") || rec.includes("주의") || rec.includes("권장") ? (
                                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-[#9a7b2d]">
                                    <path d="M12 9V13M12 17H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                  </svg>
                                ) : (
                                  <IconCheck size={12} className="text-[#3d7a4a]" />
                                )}
                              </div>
                              <span className="text-sm text-gray-600 leading-relaxed">
                                {rec.startsWith("!") ? rec.slice(1).trim() : rec}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* 주요 이슈 */}
                  {riskClauses.length > 0 && (
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900 tracking-tight">주요 이슈</h3>
                        {riskClauses.length > 3 && (
                          <button
                            onClick={() => setActiveTab("clauses")}
                            className="text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors flex items-center gap-0.5"
                          >
                            전체 보기
                            <IconChevronRight size={16} />
                          </button>
                        )}
                      </div>
                      <div className="space-y-3">
                        {riskClauses.slice(0, 3).map((clause) => (
                          <RiskClauseItem
                            key={clause.id}
                            clause={clause}
                            isActive={activeHighlightId === clause.id}
                            onClauseClick={handleClauseClick}
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
                        <h3 className="text-base font-semibold text-gray-900">법률 참조</h3>
                      </div>
                      <p className="text-base text-gray-600 leading-relaxed">
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
                      <p className="text-lg font-semibold text-gray-900 tracking-tight">위험 조항 없음</p>
                      <p className="text-base text-gray-500 mt-1">이 계약서는 안전한 것으로 보입니다</p>
                    </div>
                  ) : (
                    <>
                      {/* 정렬 옵션 */}
                      <div className="flex items-center justify-between mb-4">
                        <p className="text-base text-gray-500">
                          총 <span className="font-semibold text-gray-900">{riskClauses.length}</span>개 이슈
                        </p>
                        <SlidingTabs
                          tabs={[
                            { key: "default" as const, label: "기본" },
                            { key: "risk" as const, label: "위험도" },
                            { key: "clause" as const, label: "조항" },
                          ]}
                          activeTab={sortBy}
                          onTabChange={(tab) => setSortBy(tab)}
                          size="small"
                        />
                      </div>
                      {/* 정렬된 위험 조항 목록 */}
                      <div className="space-y-3">
                        {sortRiskClauses(riskClauses).map((clause) => (
                          <RiskClauseItem
                            key={clause.id}
                            clause={clause}
                            isActive={activeHighlightId === clause.id}
                            onClauseClick={handleClauseClick}
                          />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {activeTab === "text" && (
                <div className="card-apple p-5 animate-fadeIn">
                  <p className="text-base text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {documentText || contract.extracted_text || "추출된 텍스트가 없습니다"}
                  </p>
                </div>
              )}
            </ScrollableArea>
          </div>
        </div>
      </div>

      {/* Chat Panel - Only render one based on screen size to prevent duplicate messages */}
      {showChat && isMobile && (
        <div className="fixed inset-0 bg-transparent z-50 animate-slideInUp">
          <ChatPanel
            contractId={parseInt(id)}
            contractTitle={contract.title}
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
          className="fixed right-0 top-0 bottom-0 bg-transparent border-l border-gray-200 shadow-strong z-40"
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
            contractTitle={contract.title}
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
