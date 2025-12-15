"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import {
  IconZoomIn,
  IconZoomOut,
  IconDownload,
} from "@/components/icons";

interface DocumentViewerProps {
  fileUrl: string;
  extractedText?: string | null;
  onTextSelect?: (text: string, position: { x: number; y: number }) => void;
  className?: string;
}

export function PDFViewer({
  fileUrl,
  extractedText,
  onTextSelect,
  className,
}: DocumentViewerProps) {
  const [scale, setScale] = useState<number>(100);
  const [showScrollIndicator, setShowScrollIndicator] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

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

  // 파일 URL 정규화
  const normalizedUrl = fileUrl.startsWith("http")
    ? fileUrl
    : `http://localhost:8000${fileUrl}`;

  // 줌 조절
  const handleZoomIn = () => setScale((prev) => Math.min(150, prev + 10));
  const handleZoomOut = () => setScale((prev) => Math.max(70, prev - 10));
  const handleZoomReset = () => setScale(100);

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

  return (
    <div className={cn("flex flex-col h-full bg-[#fafafa]", className)}>
      {/* Toolbar */}
      <div className="bg-[#fafafa]">
        <div className="flex items-center justify-between px-4 sm:px-5 h-12">
          {/* Left: Title */}
          <p className="text-[11px] font-medium text-gray-500 uppercase tracking-wider">원본 문서</p>

          {/* Right: Controls */}
          <div className="flex items-center gap-2">
            {/* Zoom Controls */}
            <div className="flex items-center gap-0.5 bg-white/80 rounded-[10px] p-0.5 border border-gray-200/60">
              <button
                type="button"
                onClick={handleZoomOut}
                className="w-7 h-7 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-[8px] transition-all duration-200"
                title="축소"
              >
                <IconZoomOut size={14} />
              </button>
              <button
                type="button"
                onClick={handleZoomReset}
                className="min-w-[40px] h-7 px-1.5 text-[11px] font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-[8px] transition-all duration-200"
                title="원본 크기"
              >
                {scale}%
              </button>
              <button
                type="button"
                onClick={handleZoomIn}
                className="w-7 h-7 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-[8px] transition-all duration-200"
                title="확대"
              >
                <IconZoomIn size={16} />
              </button>
            </div>

            {/* PDF Download */}
            <a
              href={normalizedUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 h-8 px-3 text-[11px] font-medium text-gray-600 bg-white border border-gray-200/60 hover:border-gray-300 hover:bg-gray-50 rounded-[10px] transition-all duration-200"
              title="PDF 원본 보기"
            >
              <IconDownload size={12} />
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
              className="mx-auto max-w-3xl card-apple p-6 sm:p-8 origin-top-left transition-transform duration-200"
              style={{ transform: `scale(${scale / 100})` }}
            >
              {extractedText ? (
                <div className="prose prose-gray prose-sm max-w-none contract-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {extractedText}
                  </ReactMarkdown>
                </div>
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
          <div className="absolute inset-0 bg-gradient-to-t from-[#fafafa] to-transparent" />
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2">
            <div className="w-6 h-6 rounded-full bg-white/90 shadow-sm border border-gray-200/60 flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PDFViewer;
