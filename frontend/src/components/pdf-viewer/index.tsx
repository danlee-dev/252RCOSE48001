"use client";

import { useState, useRef, useCallback } from "react";
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
  const containerRef = useRef<HTMLDivElement>(null);

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
    <div className={cn("flex flex-col h-full bg-white", className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
            문서 내용
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* 줌 */}
          <button
            onClick={handleZoomOut}
            className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            title="축소"
          >
            <IconZoomOut size={16} />
          </button>
          <button
            onClick={handleZoomReset}
            className="text-xs font-medium text-gray-600 hover:text-gray-900 min-w-[45px] text-center"
            title="원본 크기"
          >
            {scale}%
          </button>
          <button
            onClick={handleZoomIn}
            className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            title="확대"
          >
            <IconZoomIn size={16} />
          </button>

          <div className="w-px h-4 bg-gray-200 mx-2" />

          {/* PDF 원본 보기 */}
          <a
            href={normalizedUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            title="PDF 원본 보기"
          >
            <IconDownload size={14} />
            <span>원본 PDF</span>
          </a>
        </div>
      </div>

      {/* Content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto"
        onMouseUp={handleMouseUp}
      >
        <div className="p-6">
          <article
            className="mx-auto max-w-3xl bg-white rounded-xl border border-gray-200 shadow-soft p-8"
            style={{ fontSize: `${scale}%` }}
          >
            {extractedText ? (
              <div className="prose prose-gray prose-sm max-w-none contract-markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {extractedText}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-gray-400 text-center py-12">
                추출된 텍스트가 없습니다
              </p>
            )}
          </article>
        </div>
      </div>
    </div>
  );
}

export default PDFViewer;
