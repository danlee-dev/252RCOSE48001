"use client";

import { useState, useEffect, useRef, use } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LegalNoticeSessionDetail, legalNoticeApi } from "@/lib/api";
import {
  IconArrowLeft,
  IconLoading,
  IconFileText,
  IconShield,
  IconDownload,
  IconDocument,
  IconLightbulb,
  IconClose,
} from "@/components/icons";
import { AIAvatarSmall } from "@/components/ai-avatar";
import { cn } from "@/lib/utils";

interface SessionDetailPageProps {
  params: Promise<{
    id: string;
  }>;
}

interface ChatMessage {
  role: "user" | "ai";
  content: string;
}

export default function SessionDetailPage({ params }: SessionDetailPageProps) {
  const { id: sessionId } = use(params);

  const [loading, setLoading] = useState(true);
  const [session, setSession] = useState<LegalNoticeSessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Chat continuation
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [isInfoComplete, setIsInfoComplete] = useState(false);

  // Mobile modals
  const [showMobileChatModal, setShowMobileChatModal] = useState(false);
  const [showMobileEvidenceModal, setShowMobileEvidenceModal] = useState(false);

  // Right section tab (대화 / 증거수집 전략)
  const [rightTab, setRightTab] = useState<"chat" | "evidence">("chat");

  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadSession();
  }, [sessionId]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  async function loadSession() {
    try {
      setLoading(true);
      const data = await legalNoticeApi.getSession(sessionId);
      setSession(data);

      // Initialize chat messages from session
      const messages: ChatMessage[] = data.messages.map((msg) => ({
        role: msg.role as "user" | "ai",
        content: msg.content,
      }));
      setChatMessages(messages);

      // Check if info collection is complete (has enough required fields)
      const collectedKeys = Object.keys(data.collected_info || {});
      const hasEnoughInfo = collectedKeys.length >= 5;
      setIsInfoComplete(hasEnoughInfo || data.status === "completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "세션을 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }

  async function handleSendMessage() {
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setChatLoading(true);

    try {
      const response = await legalNoticeApi.chat({
        session_id: sessionId,
        user_message: userMessage,
      });

      setChatMessages((prev) => [...prev, { role: "ai", content: response.ai_message }]);
      setIsInfoComplete(response.is_complete);

      // Reload session to get updated collected_info
      loadSession();
    } catch (error) {
      console.error("Chat error:", error);
      setChatMessages((prev) => [...prev, {
        role: "ai",
        content: "죄송합니다. 오류가 발생했습니다. 다시 말씀해 주시겠어요?"
      }]);
    } finally {
      setChatLoading(false);
    }
  }

  async function handleRegenerate() {
    setGenerating(true);
    try {
      await legalNoticeApi.generate({
        session_id: sessionId,
      });

      // Reload session to get new content
      loadSession();
    } catch (error) {
      console.error("Generation error:", error);
      alert("내용증명 재생성에 실패했습니다.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleDownloadPDF() {
    if (!session?.final_content) {
      alert("다운로드할 내용이 없습니다.");
      return;
    }

    // For now, create a simple text file download
    // TODO: Implement proper PDF generation
    const blob = new Blob([session.final_content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `내용증명_${session.title || sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4 animate-fadeIn">
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-xl shadow-[#3d5a47]/30">
              <IconLoading size={40} className="text-white" />
            </div>
            <div className="absolute -inset-2 rounded-3xl border-2 border-[#3d5a47]/20 animate-pulse" />
          </div>
          <p className="text-lg text-gray-600 font-medium tracking-tight">불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <p className="text-lg text-[#b54a45] mb-4">{error || "세션을 찾을 수 없습니다"}</p>
          <Link
            href="/certification"
            className="inline-flex items-center gap-2 px-6 py-3 bg-[#3d5a47] text-white font-semibold rounded-xl"
          >
            <IconArrowLeft size={20} />
            목록으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-10">
      {/* Header */}
      <section className="relative">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-[#3d5a47]/10 to-transparent rounded-full blur-2xl" />
        <div className="relative">
          <Link
            href="/certification"
            className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-[#3d5a47] mb-4 transition-colors"
          >
            <IconArrowLeft size={16} />
            목록으로
          </Link>
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                  <IconShield size={20} className="text-white" />
                </div>
                <span className={cn(
                  "px-3 py-1 text-sm font-semibold rounded-full",
                  session.status === "completed"
                    ? "bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]"
                    : "bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]"
                )}>
                  {session.status === "completed" ? "완료" : "작성중"}
                </span>
              </div>
              <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
                {session.title || "내용증명"}
              </h1>
              {session.contract_title && (
                <p className="text-base text-gray-500 mt-2 flex items-center gap-2">
                  <IconDocument size={16} />
                  연결된 계약서: {session.contract_title}
                </p>
              )}
            </div>
            {session.status === "completed" && (
              <button
                onClick={handleDownloadPDF}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 transition-all"
              >
                <IconDownload size={20} />
                다운로드
              </button>
            )}
          </div>
        </div>
      </section>

      {/* Mobile Action Buttons */}
      <div className="flex gap-2 sm:hidden">
        <button
          onClick={() => setShowMobileChatModal(true)}
          className="flex-1 py-3 text-base font-semibold rounded-xl transition-all bg-[#3d5a47] text-white"
        >
          대화하기
        </button>
        <button
          onClick={() => setShowMobileEvidenceModal(true)}
          className="flex-1 py-3 text-base font-semibold rounded-xl transition-all bg-gray-100 text-gray-600"
        >
          증거수집 전략
        </button>
      </div>

      {/* Split View Container */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Preview */}
        <div className="space-y-6">
          {/* Legal Notice Content */}
          <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden relative z-10">
            <div className="px-6 py-4 border-b border-gray-100 bg-gray-50 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <IconFileText size={20} className="text-[#3d5a47]" />
                <h2 className="text-lg font-bold text-gray-900 tracking-tight">내용증명</h2>
              </div>
              {session.status === "completed" && (
                <button
                  onClick={handleRegenerate}
                  disabled={generating}
                  className="text-sm text-[#3d5a47] hover:underline disabled:opacity-50"
                >
                  {generating ? "생성 중..." : "다시 생성"}
                </button>
              )}
            </div>
            <div className="p-6">
              {session.final_content ? (
                <article className="contract-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {session.final_content
                      .replace(/^```markdown\n?/i, '')
                      .replace(/\n?```$/i, '')
                      .trim()}
                  </ReactMarkdown>
                </article>
              ) : (
                <div className="text-center py-10 text-gray-500">
                  <IconFileText size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-base">아직 생성된 내용증명이 없습니다.</p>
                  <p className="text-sm mt-1">대화를 완료하면 내용증명이 생성됩니다.</p>
                </div>
              )}
            </div>
          </section>

        </div>

        {/* Right: Chat & Evidence - Hidden on mobile, visible on desktop */}
        <div className="space-y-4 hidden sm:block">
          {/* Summary Section */}
          {session.damage_summary && (
            <div className="bg-gradient-to-r from-[#e8f0ea] to-[#f0f5f1] rounded-2xl border border-[#c8e6cf] p-5">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-xl bg-[#3d5a47] flex items-center justify-center flex-shrink-0">
                  <IconDocument size={20} className="text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-bold text-[#3d5a47] tracking-tight mb-1">내용증명 요약</h3>
                  <p className="text-sm text-gray-700 leading-relaxed line-clamp-3">
                    {session.damage_summary
                      .replace(/^```markdown\n?/i, '')
                      .replace(/\n?```$/i, '')
                      .replace(/[#*`]/g, '')
                      .trim()
                      .slice(0, 200)}
                    {session.damage_summary.length > 200 ? "..." : ""}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Tab Buttons */}
          <div className="flex gap-2 bg-gray-100 p-1.5 rounded-2xl">
            <button
              onClick={() => setRightTab("chat")}
              className={cn(
                "flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all",
                rightTab === "chat"
                  ? "bg-white text-[#3d5a47] shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              대화
            </button>
            <button
              onClick={() => setRightTab("evidence")}
              className={cn(
                "flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-1.5",
                rightTab === "evidence"
                  ? "bg-white text-[#3d5a47] shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              <IconLightbulb size={16} />
              증거수집 전략
            </button>
          </div>

          {/* Tab Content */}
          {rightTab === "chat" ? (
            <section className="liquid-glass rounded-2xl border border-white/40 overflow-hidden h-[calc(100vh-520px)] sm:h-[calc(100vh-420px)] min-h-[300px] sm:min-h-[400px] flex flex-col">
              {/* Collected Info - Compact on mobile */}
              {session.collected_info && Object.keys(session.collected_info).length > 0 && (
                <div className="px-4 sm:px-6 py-2 sm:py-3 bg-[#e8f0ea]/50 border-b border-[#c8e6cf]/50 flex-shrink-0">
                  <p className="text-xs sm:text-sm font-semibold text-[#3d5a47] mb-1.5 sm:mb-2">수집된 정보</p>
                  <div className="flex flex-wrap gap-1.5 sm:gap-2 max-h-[80px] sm:max-h-none overflow-y-auto">
                    {Object.entries(session.collected_info).map(([key, value]) => {
                      if (!value) return null;
                      const displayValue = String(value);
                      return (
                        <span
                          key={key}
                          className="px-2 sm:px-3 py-0.5 sm:py-1 bg-white rounded-full text-xs sm:text-sm text-[#3d5a47] border border-[#c8e6cf]"
                        >
                          {key}: {displayValue.slice(0, 15)}{displayValue.length > 15 ? "..." : ""}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Chat Messages */}
              <div
                ref={chatContainerRef}
                className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50/30"
              >
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "flex",
                      msg.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    {msg.role === "user" ? (
                      <div className="max-w-[85%] px-4 py-3 bg-[#3d5a47] text-white rounded-2xl rounded-br-md">
                        <p className="text-base whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    ) : (
                      <div className="max-w-[85%] flex gap-2.5">
                        <AIAvatarSmall size={28} className="mt-1 flex-shrink-0" />
                        <div className="bg-white text-gray-900 border border-gray-200 rounded-2xl rounded-tl-md shadow-sm px-4 py-3">
                          <p className="text-base whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="flex gap-2.5">
                      <AIAvatarSmall size={28} isThinking={true} className="flex-shrink-0" />
                      <div className="bg-white text-gray-500 px-4 py-3 rounded-2xl border border-gray-200 rounded-tl-md">
                        <div className="flex items-center gap-1.5">
                          <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                          <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                          <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="p-4 border-t border-gray-100 bg-white flex-shrink-0">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                    placeholder={session.status === "completed" ? "수정사항이나 질문을 입력하세요..." : "메시지를 입력하세요..."}
                    disabled={chatLoading || generating}
                    className="flex-1 px-4 py-3 text-base bg-gray-50 border-2 border-gray-200 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all disabled:opacity-50"
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!chatInput.trim() || chatLoading || generating}
                    className="px-5 py-3 bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-[#3d5a47]/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    전송
                  </button>
                </div>

                {/* Generate Button */}
                {isInfoComplete && !session.final_content && (
                  <button
                    onClick={handleRegenerate}
                    disabled={generating}
                    className="w-full mt-3 flex items-center justify-center gap-2 px-6 py-4 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 transition-all disabled:opacity-50"
                  >
                    {generating ? (
                      <>
                        <IconLoading size={20} />
                        생성 중...
                      </>
                    ) : (
                      <>
                        <IconFileText size={20} />
                        내용증명 생성
                      </>
                    )}
                  </button>
                )}
              </div>
            </section>
          ) : (
            <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden h-[calc(100vh-520px)] sm:h-[calc(100vh-420px)] min-h-[300px] sm:min-h-[400px] flex flex-col">
              <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-[#c8e6cf] bg-[#e8f0ea] flex items-center gap-3 flex-shrink-0">
                <IconLightbulb size={20} className="text-[#3d5a47]" />
                <h2 className="text-base sm:text-lg font-bold text-[#3d5a47] tracking-tight">증거 수집 가이드</h2>
              </div>
              <div className="flex-1 overflow-y-auto p-4 sm:p-6">
                {session.evidence_guide ? (
                  <article className="contract-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {session.evidence_guide
                        .replace(/^```markdown\n?/i, '')
                        .replace(/\n?```$/i, '')
                        .trim()}
                    </ReactMarkdown>
                  </article>
                ) : (
                  <div className="text-center py-10 text-gray-500">
                    <IconLightbulb size={48} className="mx-auto mb-4 text-gray-300" />
                    <p className="text-base">아직 생성된 증거 수집 가이드가 없습니다.</p>
                    <p className="text-sm mt-1">내용증명 생성 후 확인할 수 있습니다.</p>
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      </div>

      {/* Mobile Chat Modal */}
      {showMobileChatModal && (
        <div className="fixed inset-0 z-50 sm:hidden">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-sm"
            onClick={() => setShowMobileChatModal(false)}
          />

          {/* Modal Content */}
          <div className="absolute inset-x-0 bottom-0 top-16 bg-[#f0f5f1] rounded-t-3xl shadow-2xl flex flex-col animate-slideUp">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200/50 bg-white/80 backdrop-blur-xl rounded-t-3xl">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-[10px] bg-[#3d5a47] flex items-center justify-center">
                  <IconFileText size={16} className="text-white" />
                </div>
                <span className="font-semibold text-gray-900">대화</span>
              </div>
              <button
                onClick={() => setShowMobileChatModal(false)}
                className="w-9 h-9 flex items-center justify-center rounded-[10px] hover:bg-gray-100 transition-colors"
              >
                <IconClose size={20} className="text-gray-500" />
              </button>
            </div>

            {/* Collected Info - Compact */}
            {session.collected_info && Object.keys(session.collected_info).length > 0 && (
              <div className="px-4 py-2 bg-[#e8f0ea]/50 border-b border-[#c8e6cf]/50">
                <p className="text-xs font-semibold text-[#3d5a47] mb-1.5">수집된 정보</p>
                <div className="flex flex-wrap gap-1.5 max-h-[60px] overflow-y-auto">
                  {Object.entries(session.collected_info).map(([key, value]) => {
                    if (!value) return null;
                    const displayValue = String(value);
                    return (
                      <span
                        key={key}
                        className="px-2 py-0.5 bg-white rounded-full text-xs text-[#3d5a47] border border-[#c8e6cf]"
                      >
                        {key}: {displayValue.slice(0, 12)}{displayValue.length > 12 ? "..." : ""}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Chat Messages */}
            <div
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto p-4 space-y-3"
            >
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={cn(
                    "flex",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "user" ? (
                    <div className="max-w-[85%] px-3 py-2.5 bg-[#3d5a47] text-white rounded-2xl rounded-br-md">
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  ) : (
                    <div className="max-w-[85%] flex gap-2">
                      <AIAvatarSmall size={24} className="mt-1 flex-shrink-0" />
                      <div className="bg-white text-gray-900 border border-gray-200 rounded-2xl rounded-tl-md shadow-sm px-3 py-2.5">
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="flex gap-2">
                    <AIAvatarSmall size={24} isThinking={true} className="flex-shrink-0" />
                    <div className="bg-white text-gray-500 px-3 py-2.5 rounded-2xl border border-gray-200 rounded-tl-md">
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Chat Input */}
            <div className="p-3 border-t border-gray-200 bg-white safe-area-bottom">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                  placeholder="메시지를 입력하세요..."
                  disabled={chatLoading || generating}
                  className="flex-1 px-3 py-2.5 text-sm bg-gray-50 border border-gray-200 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-2 focus:ring-[#3d5a47]/10 transition-all disabled:opacity-50"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!chatInput.trim() || chatLoading || generating}
                  className="px-4 py-2.5 bg-[#3d5a47] text-white rounded-xl font-medium text-sm hover:bg-[#4a6b52] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  전송
                </button>
              </div>

              {/* Generate Button */}
              {isInfoComplete && !session.final_content && (
                <button
                  onClick={() => {
                    handleRegenerate();
                    setShowMobileChatModal(false);
                  }}
                  disabled={generating}
                  className="w-full mt-2 flex items-center justify-center gap-2 px-4 py-3 text-sm font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl transition-all disabled:opacity-50"
                >
                  {generating ? (
                    <>
                      <IconLoading size={18} />
                      생성 중...
                    </>
                  ) : (
                    <>
                      <IconFileText size={18} />
                      내용증명 생성
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Mobile Evidence Modal */}
      {showMobileEvidenceModal && (
        <div className="fixed inset-0 z-50 sm:hidden">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-sm"
            onClick={() => setShowMobileEvidenceModal(false)}
          />

          {/* Modal Content */}
          <div className="absolute inset-x-0 bottom-0 top-16 bg-[#f0f5f1] rounded-t-3xl shadow-2xl flex flex-col animate-slideUp">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#c8e6cf] bg-[#e8f0ea] rounded-t-3xl">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-[10px] bg-[#3d5a47] flex items-center justify-center">
                  <IconLightbulb size={16} className="text-white" />
                </div>
                <span className="font-semibold text-[#3d5a47]">증거수집 전략</span>
              </div>
              <button
                onClick={() => setShowMobileEvidenceModal(false)}
                className="w-9 h-9 flex items-center justify-center rounded-[10px] hover:bg-[#d8e8da] transition-colors"
              >
                <IconClose size={20} className="text-[#3d5a47]" />
              </button>
            </div>

            {/* Evidence Content */}
            <div className="flex-1 overflow-y-auto p-4 bg-white">
              {session.evidence_guide ? (
                <article className="contract-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {session.evidence_guide
                      .replace(/^```markdown\n?/i, '')
                      .replace(/\n?```$/i, '')
                      .trim()}
                  </ReactMarkdown>
                </article>
              ) : (
                <div className="text-center py-10 text-gray-500">
                  <IconLightbulb size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-base">증거수집 전략이 아직 생성되지 않았습니다.</p>
                  <p className="text-sm mt-1">내용증명 생성 후 확인할 수 있습니다.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
