"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  IconZoomIn,
  IconZoomOut,
  IconDownload,
  IconEdit,
  IconCheck,
  IconClose,
} from "@/components/icons";

// 위험 조항 하이라이트 정보
export interface HighlightClause {
  id: string;
  level: string; // "high" | "medium" | "low"
  text: string;
  matchedText?: string;   // Gemini에서 반환한 정확한 하이라이트 텍스트
  suggestedText?: string; // Gemini에서 반환한 수정안 텍스트
  explanation?: string;
  suggestion?: string;
  clauseNumber?: string;
  // Legacy (더 이상 사용하지 않음, 하위 호환성 유지)
  startIndex?: number;
  endIndex?: number;
}

interface DocumentViewerProps {
  fileUrl: string;
  extractedText?: string | null;
  onTextSelect?: (text: string, position: { x: number; y: number }) => void;
  className?: string;
  // 하이라이팅 관련 props
  highlights?: HighlightClause[];
  onHighlightClick?: (clause: HighlightClause) => void;
  onApplyFix?: (clause: HighlightClause) => void;
  activeHighlightId?: string; // 외부에서 선택된 조항 ID
  // 편집 관련 props
  onTextChange?: (newText: string) => void; // 텍스트 변경 콜백
  onSaveVersion?: (content: string, summary: string) => Promise<void>; // 버전 저장 콜백
  // 버전 표시 관련 props
  viewingVersion?: number | null; // 현재 보고 있는 버전 번호 (null이면 원본)
  // 저장 상태 표시
  saveStatus?: "idle" | "saving" | "saved";
}

// 하이라이트 툴팁 컴포넌트
interface HighlightTooltipProps {
  clause: HighlightClause;
  highlightRect: DOMRect | null; // 하이라이트 요소의 rect
  onApplyFix?: (clause: HighlightClause) => void;
  onClose: () => void;
}

function HighlightTooltip({ clause, highlightRect, onApplyFix, onClose }: HighlightTooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const closeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [placement, setPlacement] = useState<"above" | "below">("below");
  const [adjustedPosition, setAdjustedPosition] = useState({ x: 0, y: 0 });

  // 화면 공간에 따라 위/아래 배치 결정 및 위치 조정
  useEffect(() => {
    if (!highlightRect || !tooltipRef.current) return;

    const tooltipHeight = tooltipRef.current.offsetHeight;
    const tooltipWidth = tooltipRef.current.offsetWidth;
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const margin = 16;

    // 아래 공간 vs 위 공간 비교
    const spaceBelow = viewportHeight - highlightRect.bottom - margin;
    const spaceAbove = highlightRect.top - margin;

    // 아래에 공간이 부족하고 위에 더 많은 공간이 있으면 위에 배치
    const shouldPlaceAbove = spaceBelow < tooltipHeight && spaceAbove > spaceBelow;
    setPlacement(shouldPlaceAbove ? "above" : "below");

    // X 위치 조정 (화면 밖으로 나가지 않도록)
    let x = highlightRect.left;
    if (x + tooltipWidth > viewportWidth - 20) {
      x = viewportWidth - tooltipWidth - 20;
    }
    if (x < 20) {
      x = 20;
    }

    // Y 위치 계산
    const y = shouldPlaceAbove
      ? highlightRect.top - tooltipHeight - 8
      : highlightRect.bottom + 8;

    setAdjustedPosition({ x, y });
  }, [highlightRect]);

  // 마우스가 모달 영역을 벗어나면 닫기 (약간의 딜레이)
  const handleMouseLeave = useCallback(() => {
    closeTimeoutRef.current = setTimeout(() => {
      onClose();
    }, 300); // 300ms 딜레이
  }, [onClose]);

  const handleMouseEnter = useCallback(() => {
    // 마우스가 다시 들어오면 닫기 취소
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
  }, []);

  // 컴포넌트 언마운트 시 타이머 정리
  useEffect(() => {
    return () => {
      if (closeTimeoutRef.current) {
        clearTimeout(closeTimeoutRef.current);
      }
    };
  }, []);

  const getLevelBadge = (level: string) => {
    const l = level.toLowerCase();
    if (l === "high") {
      return (
        <span className="px-2 py-0.5 text-xs font-semibold bg-[#fdedec] text-[#b54a45] rounded-md">
          고위험
        </span>
      );
    }
    if (l === "medium") {
      return (
        <span className="px-2 py-0.5 text-xs font-semibold bg-[#fef7e0] text-[#9a7b2d] rounded-md">
          주의
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 text-xs font-semibold bg-[#e8f5ec] text-[#3d7a4a] rounded-md">
        저위험
      </span>
    );
  };

  if (!highlightRect) return null;

  return (
    /* 외부 래퍼: 마진 영역을 포함하여 호버 영역 확장 */
    <div
      className="fixed z-50"
      style={{
        left: adjustedPosition.x - 16,
        top: adjustedPosition.y - (placement === "above" ? 16 : 0),
        padding: "16px", // 마진 영역
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div
        ref={tooltipRef}
        className="bg-white rounded-[20px] shadow-xl border border-gray-200/60 p-5 max-w-sm animate-fadeIn"
        onClick={(e) => e.stopPropagation()}
      >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          {clause.clauseNumber && (
            <span className="text-sm font-medium text-gray-400">{clause.clauseNumber}</span>
          )}
          {getLevelBadge(clause.level)}
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-[8px] transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {/* 위험 사유 */}
      {(clause.explanation || clause.suggestion) && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">위험 사유</p>
          <div className="prose prose-sm prose-gray max-w-none text-sm text-gray-700 leading-relaxed [&_strong]:font-semibold [&_strong]:text-gray-900 [&_p]:m-0">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {clause.explanation || clause.suggestion || ""}
            </ReactMarkdown>
          </div>
        </div>
      )}

      {/* 수정 제안 텍스트 */}
      {clause.suggestedText && (
        <div className="mb-4">
          <p className="text-xs font-semibold text-[#3d7a4a] uppercase tracking-wider mb-1.5">수정안</p>
          <div className="p-3 bg-[#e8f5ec]/60 border border-[#c8e6cf] rounded-[12px]">
            <p className="text-sm text-gray-700 leading-relaxed line-clamp-4">
              {clause.suggestedText}
            </p>
          </div>
        </div>
      )}

      {/* 수정하기 버튼 */}
      {clause.suggestedText && onApplyFix && (
        <button
          onClick={() => onApplyFix(clause)}
          className="w-full py-3 px-4 bg-[#3d5a47] hover:bg-[#4a6b52] text-white text-sm font-medium rounded-[12px] transition-colors flex items-center justify-center gap-2"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          수정 적용하기
        </button>
      )}
      </div>
    </div>
  );
}

// 마크다운 + 하이라이트 렌더링 컴포넌트
interface HighlightedTextProps {
  text: string;
  highlights: HighlightClause[];
  activeHighlightId?: string;
  onHighlightClick?: (clause: HighlightClause, rect: DOMRect, element: HTMLElement) => void;
  onHighlightHover?: (clause: HighlightClause | null, rect: DOMRect | null, element: HTMLElement | null) => void;
}

// 텍스트 전처리: 마크다운 정리 및 서명 섹션 포맷팅
function preprocessMarkdown(text: string): string {
  // 1. <br> 태그를 줄바꿈으로 변환
  let processed = text.replace(/<br\s*\/?>/gi, '\n\n');

  // 2. 서명 섹션 포맷팅 (사업주/근로자 정보를 표로 변환)
  processed = formatSignatureSection(processed);

  // 3. 라인별 처리
  return processed.split('\n').map(line => {
    // "--- Page X ---" 같은 대시 구분선은 이스케이프
    if (/^\s*-{2,}/.test(line)) {
      return line.replace(/-/g, '\\-');
    }

    // "____" 같은 언더스코어 구분선도 이스케이프
    if (/^\s*_{2,}/.test(line)) {
      return line.replace(/_/g, '\\_');
    }

    // 나머지는 그대로 유지
    return line;
  }).join('\n');
}

// 서명 섹션을 표 형식으로 변환
function formatSignatureSection(text: string): string {
  // 사업주/근로자 정보 패턴 매칭
  // s 플래그 대신 [\s\S] 사용 (ES5 호환)
  const signaturePattern = /---\s*\(사업주\)\s*([\s\S]+?)\n\s*\(근로자\)\s*([\s\S]+?)(?=\n\n|\n$|$)/;
  const match = text.match(signaturePattern);

  if (match) {
    const employerInfo = match[1].trim();
    const employeeInfo = match[2].trim();

    // 정보 파싱
    const parseInfo = (info: string) => {
      const result: Record<string, string> = {};

      // 사업체명/사업주명
      const businessMatch = info.match(/사업체?명\s*:\s*([^(]+?)(?:\s*\(|$)/);
      if (businessMatch) result['사업체명'] = businessMatch[1].trim();

      // 전화
      const phoneMatch = info.match(/전화\s*:\s*([\d-]+)/);
      if (phoneMatch) result['전화'] = phoneMatch[1].trim();

      // 주소
      const addressMatch = info.match(/주\s*소\s*:\s*(.+?)(?=\s*대표자|\s*연\s*락|$)/);
      if (addressMatch) result['주소'] = addressMatch[1].trim();

      // 대표자
      const ceoMatch = info.match(/대표자\s*:\s*(.+?)(?:\s*\(서명\)|$)/);
      if (ceoMatch) result['대표자'] = ceoMatch[1].trim();

      // 연락처 (근로자용)
      const contactMatch = info.match(/연\s*락\s*처\s*:\s*([\d-]+)/);
      if (contactMatch) result['연락처'] = contactMatch[1].trim();

      // 성명 (근로자용)
      const nameMatch = info.match(/성\s*명\s*:\s*(.+?)(?:\s*\(서명\)|$)/);
      if (nameMatch) result['성명'] = nameMatch[1].trim();

      return result;
    };

    const employer = parseInfo(employerInfo);
    const employee = parseInfo(employeeInfo);

    // 마크다운 표 생성
    let table = '\n\n---\n\n';
    table += '| 구분 | 사업주 | 근로자 |\n';
    table += '|:---:|:---|:---|\n';

    if (employer['사업체명'] || employee['성명']) {
      table += `| 명칭/성명 | ${employer['사업체명'] || '-'} | ${employee['성명'] || '-'} |\n`;
    }
    if (employer['대표자']) {
      table += `| 대표자 | ${employer['대표자']} | - |\n`;
    }
    if (employer['주소'] || employee['주소']) {
      table += `| 주소 | ${employer['주소'] || '-'} | ${employee['주소'] || '-'} |\n`;
    }
    if (employer['전화'] || employee['연락처']) {
      table += `| 연락처 | ${employer['전화'] || '-'} | ${employee['연락처'] || '-'} |\n`;
    }

    table += '\n**(서명)**\n';

    // 원본 서명 섹션을 표로 대체
    return text.replace(signaturePattern, table);
  }

  return text;
}

function HighlightedText({ text, highlights, activeHighlightId, onHighlightClick, onHighlightHover }: HighlightedTextProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const appliedHighlightsRef = useRef<string>(""); // 이미 적용된 하이라이트 키 추적

  // 하이라이트 ID -> 데이터 맵
  const highlightMap = useMemo(() => {
    const map = new Map<string, HighlightClause>();
    highlights.forEach(h => map.set(h.id, h));
    return map;
  }, [highlights]);

  // 유효한 하이라이트만 필터링 (matchedText 우선)
  const validHighlights = useMemo(() => {
    return highlights
      .filter(h => {
        // matchedText가 있으면 유효 (Gemini에서 반환)
        if (h.matchedText && h.matchedText.length >= 5) {
          return true;
        }
        // matchedText가 없으면 text 필드로 폴백
        if (h.text && h.text.length >= 5) {
          return true;
        }
        return false;
      });
  }, [highlights]);

  // 하이라이트 키 생성 (ID + matchedText/text 기반으로 실제 변경 감지)
  const highlightsKey = useMemo(() => {
    return validHighlights.map(h => `${h.id}:${h.matchedText || h.text}`).join('|');
  }, [validHighlights]);

  // 마크다운 렌더링 후 DOM에서 하이라이트 적용 (텍스트 매칭 기반)
  useEffect(() => {
    if (!containerRef.current || validHighlights.length === 0) return;

    // 이미 동일한 하이라이트가 적용되어 있으면 스킵 (깜빡임 방지)
    if (appliedHighlightsRef.current === highlightsKey) {
      return;
    }

    // 기존 하이라이트 제거
    containerRef.current.querySelectorAll('.highlight-mark').forEach(el => {
      const parent = el.parentNode;
      if (parent) {
        while (el.firstChild) {
          parent.insertBefore(el.firstChild, el);
        }
        parent.removeChild(el);
      }
    });

    // 마크다운 마커 제거 함수 (DOM 렌더링 후 텍스트와 비교용)
    const stripMarkdownMarkers = (text: string) => {
      return text
        .replace(/^[\s]*[-*+]\s+/gm, '')  // 불릿 리스트 마커
        .replace(/^\d+\.\s+/gm, '')       // 숫자 리스트 마커
        .trim();
    };

    // 텍스트 노드를 찾아서 하이라이트 적용
    const applyHighlights = () => {
      const container = containerRef.current;
      if (!container) return;

      for (const h of validHighlights) {
        // matchedText 우선 사용 (Gemini에서 반환한 정확한 텍스트)
        const targetText = h.matchedText || h.text;
        if (!targetText || targetText.length < 5) continue;

        console.log(`>>> [PDF-VIEWER] Processing highlight ${h.id}`);

        // matchedText를 줄 단위로 분리하고 마크다운 마커 제거
        const searchLines = targetText
          .split('\n')
          .map(l => stripMarkdownMarkers(l))
          .filter(l => l.length >= 10);  // 최소 10자 이상인 줄만

        if (searchLines.length === 0) {
          // 짧은 텍스트는 전체를 하나의 검색어로
          const cleanText = stripMarkdownMarkers(targetText);
          if (cleanText.length >= 5) {
            searchLines.push(cleanText);
          }
        }

        console.log(`>>> [PDF-VIEWER] Search lines for ${h.id}:`, searchLines);

        let found = false;

        for (const searchLine of searchLines) {
          if (found) break;

          // TreeWalker로 모든 텍스트 노드 순회
          const walker = document.createTreeWalker(
            container,
            NodeFilter.SHOW_TEXT,
            null
          );

          let node: Text | null;
          while ((node = walker.nextNode() as Text | null) && !found) {
            const nodeText = node.textContent || '';
            const matchIndex = nodeText.indexOf(searchLine);

            if (matchIndex === -1) continue;

            // 이미 하이라이트된 노드 안에 있으면 해당 마크에 ID 추가 (중복 하이라이트)
            const existingMark = node.parentElement?.closest('.highlight-mark') as HTMLElement | null;
            if (existingMark) {
              const existingIds = existingMark.getAttribute('data-highlight-ids') || existingMark.getAttribute('data-highlight-id') || '';
              const idsArray = existingIds.split(',').filter(id => id.trim());
              if (!idsArray.includes(h.id)) {
                idsArray.push(h.id);
                existingMark.setAttribute('data-highlight-ids', idsArray.join(','));
              }
              found = true;
              console.log(`>>> [PDF-VIEWER] Added ${h.id} to existing highlight (duplicate)`);
              break;
            }

            // 매칭된 텍스트 하이라이트
            const highlightStart = matchIndex;
            const highlightEnd = matchIndex + searchLine.length;

            const beforeText = nodeText.slice(0, highlightStart);
            const highlightText = nodeText.slice(highlightStart, highlightEnd);
            const afterText = nodeText.slice(highlightEnd);

            const parent = node.parentNode;
            if (!parent) continue;

            const fragment = document.createDocumentFragment();

            if (beforeText) {
              fragment.appendChild(document.createTextNode(beforeText));
            }

            const mark = document.createElement('mark');
            const isActive = h.id === activeHighlightId;
            const level = h.level.toLowerCase();
            mark.className = cn(
              'highlight-mark',
              `highlight-mark-${level}`,
              isActive && 'highlight-mark-active'
            );
            mark.setAttribute('data-highlight-id', h.id);
            mark.setAttribute('data-highlight-ids', h.id);
            mark.textContent = highlightText;
            fragment.appendChild(mark);

            if (afterText) {
              fragment.appendChild(document.createTextNode(afterText));
            }

            parent.replaceChild(fragment, node);
            found = true;
            console.log(`>>> [PDF-VIEWER] Highlighted ${h.id}: "${searchLine.substring(0, 50)}..."`);
          }
        }

        if (!found) {
          console.log(`>>> [PDF-VIEWER] NOT FOUND: ${h.id}`);
        }
      }

      // 적용 완료 후 키 저장 (다음 렌더링 시 스킵용)
      appliedHighlightsRef.current = highlightsKey;
    };

    // 약간의 지연 후 적용 (마크다운 렌더링 완료 대기)
    const timer = setTimeout(applyHighlights, 100);
    return () => clearTimeout(timer);
    // activeHighlightId는 의존성에서 제외 - 별도 useEffect에서 처리
  }, [text, highlightsKey, validHighlights]);

  // 활성 하이라이트 상태 업데이트 (깜빡임 방지를 위해 별도 처리)
  useEffect(() => {
    if (!containerRef.current) return;

    // 기존 활성 상태 제거
    containerRef.current.querySelectorAll('.highlight-mark-active').forEach(m => {
      m.classList.remove('highlight-mark-active');
    });

    if (!activeHighlightId) return;

    // 새 활성 상태 적용 및 스크롤
    const updateActiveState = () => {
      if (!containerRef.current) return;

      // 먼저 data-highlight-id로 검색
      let el = containerRef.current.querySelector(`[data-highlight-id="${activeHighlightId}"]`) as HTMLElement | null;

      // 못 찾으면 data-highlight-ids에서 해당 ID 포함하는 요소 검색
      if (!el) {
        const allMarks = containerRef.current.querySelectorAll('[data-highlight-ids]');
        for (const mark of allMarks) {
          const ids = mark.getAttribute('data-highlight-ids')?.split(',') || [];
          if (ids.includes(activeHighlightId)) {
            el = mark as HTMLElement;
            break;
          }
        }
      }

      if (el) {
        // 기존 활성 상태 다시 제거 (타이밍 이슈 방지)
        containerRef.current?.querySelectorAll('.highlight-mark-active').forEach(m => {
          m.classList.remove('highlight-mark-active');
        });
        el.classList.add('highlight-mark-active');
        // 스크롤 - block: "center"로 화면 중앙에 위치
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        console.log(`>>> [PDF-VIEWER] Scrolled to highlight: ${activeHighlightId}`);
      } else {
        console.log(`>>> [PDF-VIEWER] Highlight element not found: ${activeHighlightId}`);
      }
    };

    // 하이라이트 적용 후 실행 (applyHighlights가 100ms 후 실행되므로 150ms 대기)
    const timer = setTimeout(updateActiveState, 150);
    return () => clearTimeout(timer);
  }, [activeHighlightId]);

  // 이벤트 위임을 통한 클릭/호버 처리
  const handleClick = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const mark = target.closest('.highlight-mark') as HTMLElement;
    if (!mark) return;

    // 첫 번째 ID 사용 (data-highlight-id 또는 data-highlight-ids의 첫 번째)
    let id = mark.getAttribute('data-highlight-id');
    if (!id) {
      const ids = mark.getAttribute('data-highlight-ids')?.split(',') || [];
      id = ids[0] || null;
    }
    if (!id) return;

    const clause = highlightMap.get(id);
    if (!clause) return;

    e.stopPropagation();
    const rect = mark.getBoundingClientRect();
    onHighlightClick?.(clause, rect, mark);
  }, [highlightMap, onHighlightClick]);

  const handleMouseOver = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const mark = target.closest('.highlight-mark') as HTMLElement;
    if (!mark) return;

    // 첫 번째 ID 사용
    let id = mark.getAttribute('data-highlight-id');
    if (!id) {
      const ids = mark.getAttribute('data-highlight-ids')?.split(',') || [];
      id = ids[0] || null;
    }
    if (!id) return;

    const clause = highlightMap.get(id);
    if (!clause) return;

    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }

    const rect = mark.getBoundingClientRect();
    hoverTimeoutRef.current = setTimeout(() => {
      onHighlightHover?.(clause, rect, mark);
    }, 300);
  }, [highlightMap, onHighlightHover]);

  const handleMouseOut = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className="prose prose-gray prose-sm max-w-none contract-markdown"
      onClick={handleClick}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {preprocessMarkdown(text)}
      </ReactMarkdown>
    </div>
  );
}

export function PDFViewer({
  fileUrl,
  extractedText,
  onTextSelect,
  className,
  highlights = [],
  onHighlightClick,
  onApplyFix,
  activeHighlightId,
  onTextChange,
  onSaveVersion,
  viewingVersion,
  saveStatus = "idle",
}: DocumentViewerProps) {
  const [scale, setScale] = useState<number>(100);
  const [showScrollIndicator, setShowScrollIndicator] = useState(false);
  const [tooltipClause, setTooltipClause] = useState<HighlightClause | null>(null);
  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);
  const highlightElementRef = useRef<HTMLElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // 편집 모드 상태
  const [isEditMode, setIsEditMode] = useState(false);
  const [editText, setEditText] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 스크롤 인디케이터 로직
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const checkScroll = () => {
      const isScrollable = el.scrollHeight > el.clientHeight;
      const hasMoreBelow = el.scrollHeight - el.scrollTop - el.clientHeight > 20;
      setShowScrollIndicator(isScrollable && hasMoreBelow);
    };

    const timer = setTimeout(checkScroll, 100);
    el.addEventListener("scroll", checkScroll);
    window.addEventListener("resize", checkScroll);

    const observer = new MutationObserver(checkScroll);
    observer.observe(el, { childList: true, subtree: true });

    return () => {
      clearTimeout(timer);
      el.removeEventListener("scroll", checkScroll);
      window.removeEventListener("resize", checkScroll);
      observer.disconnect();
    };
  }, [extractedText]);

  // 외부 클릭 시 툴팁 닫기
  useEffect(() => {
    const handleClickOutside = () => {
      setTooltipClause(null);
      setHighlightRect(null);
      highlightElementRef.current = null;
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  // 스크롤 시 툴팁 위치 업데이트
  useEffect(() => {
    if (!tooltipClause || !highlightElementRef.current) return;

    const updatePosition = () => {
      if (highlightElementRef.current) {
        setHighlightRect(highlightElementRef.current.getBoundingClientRect());
      }
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener("scroll", updatePosition);
    }
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);

    return () => {
      if (container) {
        container.removeEventListener("scroll", updatePosition);
      }
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [tooltipClause]);

  // 파일 URL 정규화
  const normalizedUrl = fileUrl.startsWith("http")
    ? fileUrl
    : `http://localhost:8000${fileUrl}`;

  // 줌 조절
  const handleZoomIn = () => setScale((prev) => Math.min(150, prev + 10));
  const handleZoomOut = () => setScale((prev) => Math.max(70, prev - 10));
  const handleZoomReset = () => setScale(100);

  // 편집 모드 핸들러
  const handleEditStart = useCallback(() => {
    setEditText(extractedText || "");
    setIsEditMode(true);
    // 편집 모드 진입 시 textarea에 포커스
    setTimeout(() => textareaRef.current?.focus(), 100);
  }, [extractedText]);

  const handleEditCancel = useCallback(() => {
    setIsEditMode(false);
    setEditText("");
  }, []);

  const handleEditSave = useCallback(async () => {
    if (!editText || editText === extractedText) {
      setIsEditMode(false);
      return;
    }

    setIsSaving(true);
    try {
      // 부모 컴포넌트에 텍스트 변경 알림
      onTextChange?.(editText);

      // 버전 저장 (콜백이 있는 경우)
      if (onSaveVersion) {
        await onSaveVersion(editText, "사용자 직접 수정");
      }

      setIsEditMode(false);
    } catch (err) {
      console.error(">>> Failed to save edit:", err);
    } finally {
      setIsSaving(false);
    }
  }, [editText, extractedText, onTextChange, onSaveVersion]);

  // 텍스트 선택 핸들링
  const handleMouseUp = useCallback(() => {
    if (!onTextSelect) return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const text = selection.toString().trim();
    if (text.length < 10) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    onTextSelect(text, {
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  }, [onTextSelect]);

  // 하이라이트 클릭 핸들러
  const handleHighlightClick = useCallback((clause: HighlightClause, rect: DOMRect, element: HTMLElement) => {
    setTooltipClause(clause);
    setHighlightRect(rect);
    highlightElementRef.current = element;
    onHighlightClick?.(clause);
  }, [onHighlightClick]);

  // 하이라이트 호버 핸들러
  const handleHighlightHover = useCallback((clause: HighlightClause | null, rect: DOMRect | null, element: HTMLElement | null) => {
    if (clause && rect && element) {
      setTooltipClause(clause);
      setHighlightRect(rect);
      highlightElementRef.current = element;
    }
    // 호버로 열린 툴팁은 마우스가 떠나도 닫지 않음 (클릭으로 닫기)
  }, []);

  // 수정 적용 핸들러
  const handleApplyFix = useCallback((clause: HighlightClause) => {
    onApplyFix?.(clause);
    setTooltipClause(null);
    setHighlightRect(null);
    highlightElementRef.current = null;
  }, [onApplyFix]);

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Toolbar */}
      <div>
        <div className="flex items-center justify-end px-4 sm:px-6 h-14">
          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Save Status - 확대 버튼 왼쪽 */}
            <div className={cn(
              "flex items-center gap-1.5 text-sm transition-all duration-300",
              saveStatus === "idle" ? "opacity-0" : "opacity-100"
            )}>
              {saveStatus === "saving" ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gray-400 animate-spin">
                    <path d="M12 2V6M12 18V22M6 12H2M22 12H18M19.07 4.93L16.24 7.76M7.76 16.24L4.93 19.07M19.07 19.07L16.24 16.24M7.76 7.76L4.93 4.93" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <span className="text-gray-400">저장 중...</span>
                </>
              ) : saveStatus === "saved" ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                    <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10z" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 12l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span className="text-gray-400">드라이브에 저장됨</span>
                </>
              ) : null}
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center gap-1 bg-white/80 rounded-[12px] p-1 border border-gray-200/60">
              <button
                type="button"
                onClick={handleZoomOut}
                className="w-10 h-10 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-[10px] transition-all duration-200"
                title="축소"
              >
                <IconZoomOut size={18} />
              </button>
              <button
                type="button"
                onClick={handleZoomReset}
                className="min-w-[56px] h-10 px-3 text-base font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-[10px] transition-all duration-200"
                title="원본 크기"
              >
                {scale}%
              </button>
              <button
                type="button"
                onClick={handleZoomIn}
                className="w-10 h-10 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-[10px] transition-all duration-200"
                title="확대"
              >
                <IconZoomIn size={18} />
              </button>
            </div>

            {/* Edit Button */}
            {!isEditMode ? (
              <button
                type="button"
                onClick={handleEditStart}
                className="flex items-center justify-center gap-2 h-11 px-5 text-base font-medium text-gray-600 bg-white border border-gray-200/60 hover:border-[#3d5a47] hover:text-[#3d5a47] hover:bg-[#e8f0ea] rounded-[12px] transition-all duration-200"
                title="문서 편집"
              >
                <IconEdit size={18} />
                <span className="hidden sm:inline">편집</span>
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleEditCancel}
                  className="flex items-center justify-center gap-2 h-11 px-4 text-base font-medium text-gray-600 bg-white border border-gray-200/60 hover:border-gray-300 hover:bg-gray-50 rounded-[12px] transition-all duration-200"
                  title="취소"
                >
                  <IconClose size={18} />
                  <span className="hidden sm:inline">취소</span>
                </button>
                <button
                  type="button"
                  onClick={handleEditSave}
                  disabled={isSaving}
                  className="flex items-center justify-center gap-2 h-11 px-5 text-base font-medium text-white bg-[#3d5a47] hover:bg-[#4a6b52] rounded-[12px] transition-all duration-200 disabled:opacity-50"
                  title="저장"
                >
                  {isSaving ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <IconCheck size={18} />
                  )}
                  <span className="hidden sm:inline">{isSaving ? "저장 중..." : "저장"}</span>
                </button>
              </div>
            )}

            {/* PDF Download */}
            <a
              href={normalizedUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 h-11 px-5 text-base font-medium text-gray-600 bg-white border border-gray-200/60 hover:border-gray-300 hover:bg-gray-50 rounded-[12px] transition-all duration-200"
              title="PDF 원본 보기"
            >
              <IconDownload size={18} />
              <span className="hidden sm:inline">원본</span>
            </a>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="relative flex-1 min-h-0">
        <div
          ref={containerRef}
          className="h-full overflow-auto scrollable-area"
          onMouseUp={handleMouseUp}
        >
          {/* 스케일에 따라 확장되는 래퍼 - 스크롤 영역 확보 */}
          <div
            className="p-4 sm:p-6 transition-all duration-200"
            style={{
              width: scale > 100 ? `${scale}%` : "100%",
              minWidth: scale > 100 ? `${scale}%` : "auto",
            }}
          >
            <article
              className={cn(
                "relative card-apple p-6 sm:p-8 origin-top-left transition-transform duration-200",
                // 편집 모드가 아닐 때만 최대 너비 제한
                !isEditMode && "max-w-3xl mx-auto"
              )}
              style={{ transform: `scale(${scale / 100})` }}
            >
              {/* 문서 버전 Badge - Top Right */}
              <span className={cn(
                "absolute top-4 right-4 px-3 py-1.5 text-xs font-semibold rounded-[8px] shadow-sm border",
                viewingVersion && viewingVersion > 1
                  ? "bg-[#e8f5ec] text-[#3d7a4a] border-[#c8e6cf]"
                  : "bg-white/90 backdrop-blur-sm text-gray-500 border-gray-200/60"
              )}>
                {viewingVersion && viewingVersion > 1 ? `v${viewingVersion} 수정본` : "원본 문서"}
              </span>
              {isEditMode ? (
                /* 편집 모드: 에디터 + 실시간 프리뷰 */
                <div className="flex flex-col lg:flex-row gap-4 h-full min-h-[400px]">
                  {/* 에디터 영역 */}
                  <div className="flex-1 flex flex-col">
                    <div className="flex items-center gap-2 mb-2 px-1">
                      <div className="w-2 h-2 rounded-full bg-[#3d5a47]" />
                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">편집</span>
                    </div>
                    <textarea
                      ref={textareaRef}
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      className="flex-1 w-full p-4 text-sm text-gray-800 leading-relaxed bg-white border border-gray-200 rounded-[16px] resize-none focus:outline-none focus:ring-2 focus:ring-[#3d5a47]/20 focus:border-[#3d5a47] font-mono"
                      placeholder="계약서 내용을 편집하세요..."
                      spellCheck={false}
                    />
                  </div>
                  {/* 실시간 프리뷰 영역 */}
                  <div className="flex-1 flex flex-col">
                    <div className="flex items-center gap-2 mb-2 px-1">
                      <div className="w-2 h-2 rounded-full bg-blue-500" />
                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">미리보기</span>
                    </div>
                    <div className="flex-1 p-4 bg-gray-50 border border-gray-200 rounded-[16px] overflow-auto">
                      <div className="prose prose-gray prose-sm max-w-none contract-markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {preprocessMarkdown(editText)}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                </div>
              ) : extractedText ? (
                highlights.length > 0 ? (
                  <HighlightedText
                    text={extractedText}
                    highlights={highlights}
                    activeHighlightId={activeHighlightId}
                    onHighlightClick={handleHighlightClick}
                    onHighlightHover={handleHighlightHover}
                  />
                ) : (
                  <div className="prose prose-gray prose-sm max-w-none contract-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {preprocessMarkdown(extractedText)}
                    </ReactMarkdown>
                  </div>
                )
              ) : (
                <div className="text-center py-16">
                  <div className="inline-flex items-center justify-center w-14 h-14 bg-gray-100 rounded-[16px] mb-4">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                      <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <p className="text-sm font-medium text-gray-500 tracking-tight">추출된 텍스트가 없습니다</p>
                </div>
              )}
            </article>
          </div>
        </div>
        {/* Scroll indicator */}
        <div
          className={cn(
            "absolute bottom-0 left-0 right-0 h-12 pointer-events-none transition-opacity duration-300",
            showScrollIndicator ? "opacity-100" : "opacity-0"
          )}
        >
          <div className="absolute inset-0 bg-gradient-to-t from-white/60 to-transparent" />
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2">
            <div className="w-6 h-6 rounded-full bg-white/90 shadow-sm border border-gray-200/60 flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Highlight Tooltip */}
      {tooltipClause && highlightRect && (
        <HighlightTooltip
          clause={tooltipClause}
          highlightRect={highlightRect}
          onApplyFix={handleApplyFix}
          onClose={() => {
            setTooltipClause(null);
            setHighlightRect(null);
            highlightElementRef.current = null;
          }}
        />
      )}
    </div>
  );
}

export default PDFViewer;
