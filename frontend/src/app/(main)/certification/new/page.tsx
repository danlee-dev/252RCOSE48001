"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Contract, contractsApi, legalNoticeApi } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  IconLoading,
  IconDocument,
  IconArrowLeft,
  IconFileText,
  IconCheck,
  IconShield,
  IconUpload,
  IconLightbulb,
  IconClose,
} from "@/components/icons";
import { AIAvatarSmall } from "@/components/ai-avatar";

interface ChatMessage {
  role: "user" | "ai";
  content: string;
}

export default function NewCertificationPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [contracts, setContracts] = useState<Contract[]>([]);

  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [isInfoComplete, setIsInfoComplete] = useState(false);
  const [collectedInfo, setCollectedInfo] = useState<Record<string, unknown>>({});
  const [generating, setGenerating] = useState(false);

  // Contract selection state
  const [selectedContractId, setSelectedContractId] = useState<number | null>(null);
  const [evidenceGuide, setEvidenceGuide] = useState<string | null>(null);

  // Right section tab
  const [rightTab, setRightTab] = useState<"chat" | "evidence" | "contract">("chat");

  // Mobile chat modal
  const [showMobileChatModal, setShowMobileChatModal] = useState(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadContracts();
    startNewSession();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  async function loadContracts() {
    try {
      const data = await contractsApi.list(0, 100);
      setContracts(data.items.filter((c) => c.status === "COMPLETED"));
    } catch {
      // Error handling
    }
  }

  async function startNewSession() {
    setLoading(true);
    try {
      const response = await legalNoticeApi.startSession({});
      setSessionId(response.session_id);
      setChatMessages([{ role: "ai", content: response.ai_message }]);
    } catch (error) {
      console.error("Failed to start session:", error);
      setChatMessages([{
        role: "ai",
        content: "세션을 시작하는 중 오류가 발생했습니다. 페이지를 새로고침 해주세요."
      }]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSendMessage() {
    if (!chatInput.trim() || !sessionId || chatLoading) return;

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
      setCollectedInfo(response.collected_info);
      setIsInfoComplete(response.is_complete);
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

  async function handleSelectContract(contractId: number) {
    if (!sessionId) return;

    setSelectedContractId(contractId);

    try {
      const response = await legalNoticeApi.linkContract(sessionId, contractId);
      const contract = contracts.find(c => c.id === contractId);

      // 증거수집 전략이 생성되었으면 저장
      if (response.evidence_guide) {
        setEvidenceGuide(response.evidence_guide);
      }

      setChatMessages((prev) => [...prev, {
        role: "ai",
        content: `"${contract?.title}" 계약서가 연결되었습니다. 계약서 분석 내용을 참고하여 내용증명을 작성하겠습니다.`
      }]);
    } catch (error) {
      console.error("Failed to link contract:", error);
      setChatMessages((prev) => [...prev, {
        role: "ai",
        content: "계약서 연결 중 오류가 발생했습니다. 다시 시도해주세요."
      }]);
    }
  }

  async function handleGenerateLegalNotice() {
    if (!sessionId) return;

    setGenerating(true);
    try {
      await legalNoticeApi.generate({
        session_id: sessionId,
      });

      router.push(`/certification/${sessionId}`);
    } catch (error) {
      console.error("Generation error:", error);
      setChatMessages((prev) => [...prev, {
        role: "ai",
        content: "내용증명 생성에 실패했습니다. 다시 시도해주세요."
      }]);
    } finally {
      setGenerating(false);
    }
  }

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [chatInput, sessionId, chatLoading]);

  const selectedContract = contracts.find((c) => c.id === selectedContractId);

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
          <p className="text-lg text-gray-600 font-medium tracking-tight">세션 시작 중...</p>
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
                <span className="px-3 py-1 text-sm font-semibold rounded-full bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]">
                  작성중
                </span>
              </div>
              <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
                내용증명 작성
              </h1>
              {selectedContract && (
                <p className="text-base text-gray-500 mt-2 flex items-center gap-2">
                  <IconDocument size={16} />
                  연결된 계약서: {selectedContract.title}
                </p>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Mobile Action Button */}
      <div className="sm:hidden">
        <button
          onClick={() => setShowMobileChatModal(true)}
          className="w-full py-3 text-base font-semibold rounded-xl transition-all bg-[#3d5a47] text-white"
        >
          대화하기
        </button>
      </div>

      {/* Split View Container */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Info Panel - Always visible on mobile now */}
        <div className="space-y-6">
          {/* Instructions Card */}
          <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-100 bg-gray-50 flex items-center gap-3">
              <IconFileText size={20} className="text-[#3d5a47]" />
              <h2 className="text-lg font-bold text-gray-900 tracking-tight">작성 안내</h2>
            </div>
            <div className="p-6 space-y-4">
              <p className="text-base text-gray-700 leading-relaxed">
                AI 비서와 대화하며 내용증명 작성에 필요한 정보를 알려주세요.
                필요한 모든 정보가 수집되면 내용증명이 자동으로 생성됩니다.
              </p>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#e8f5ec] flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-[#3d5a47]">1</span>
                  </div>
                  <p className="text-sm text-gray-600">발신인/수신인 정보 입력</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#e8f5ec] flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-[#3d5a47]">2</span>
                  </div>
                  <p className="text-sm text-gray-600">근무 기간 및 피해 내용 설명</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#e8f5ec] flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-[#3d5a47]">3</span>
                  </div>
                  <p className="text-sm text-gray-600">요구사항 명시</p>
                </div>
              </div>
            </div>
          </section>

          {/* Collected Info */}
          {Object.keys(collectedInfo).length > 0 && (
            <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-[#c8e6cf] bg-[#e8f0ea] flex items-center gap-3">
                <IconCheck size={20} className="text-[#3d5a47]" />
                <h2 className="text-lg font-bold text-[#3d5a47] tracking-tight">수집된 정보</h2>
              </div>
              <div className="p-6">
                <div className="space-y-3">
                  {Object.entries(collectedInfo).map(([key, value]) => {
                    if (!value) return null;
                    const displayValue = String(value);
                    return (
                      <div key={key} className="flex items-start gap-3">
                        <span className="text-sm font-medium text-gray-500 min-w-[100px]">{key}</span>
                        <span className="text-sm text-gray-900">{displayValue}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          )}

          {/* Generate Button (Desktop) */}
          <div className="hidden sm:block">
            <button
              onClick={handleGenerateLegalNotice}
              disabled={generating || !isInfoComplete}
              className={cn(
                "w-full flex items-center justify-center gap-2 px-8 py-4 text-base font-semibold text-white rounded-xl transition-all duration-200",
                "bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5",
                "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none disabled:hover:translate-y-0"
              )}
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
            {!isInfoComplete && (
              <p className="text-center text-sm text-gray-500 mt-3">
                모든 필수 정보가 수집되면 버튼이 활성화됩니다.
              </p>
            )}
          </div>
        </div>

        {/* Right: Chat & Contract Selection - Hidden on mobile, shown on desktop */}
        <div className="space-y-4 hidden sm:block">
          {/* Tab Buttons */}
          <div className="flex gap-1 bg-gray-100 p-1.5 rounded-2xl">
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
                "flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-1",
                rightTab === "evidence"
                  ? "bg-white text-[#3d5a47] shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              <IconLightbulb size={14} />
              증거수집
            </button>
            <button
              onClick={() => setRightTab("contract")}
              className={cn(
                "flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-1",
                rightTab === "contract"
                  ? "bg-white text-[#3d5a47] shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              <IconDocument size={14} />
              계약서
            </button>
          </div>

          {/* Tab Content */}
          {rightTab === "chat" ? (
            <section className="liquid-glass rounded-2xl border border-white/40 overflow-hidden h-[calc(100vh-480px)] sm:h-[calc(100vh-380px)] min-h-[280px] sm:min-h-[400px] flex flex-col">
              {/* Chat Messages */}
              <div
                ref={chatContainerRef}
                className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 bg-gray-50/30"
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
                    onKeyDown={handleKeyDown}
                    placeholder="메시지를 입력하세요..."
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

                {/* Mobile Generate Button */}
                <div className="sm:hidden mt-3">
                  <button
                    onClick={handleGenerateLegalNotice}
                    disabled={generating || !isInfoComplete}
                    className={cn(
                      "w-full flex items-center justify-center gap-2 px-6 py-4 text-base font-semibold text-white rounded-xl transition-all",
                      "bg-gradient-to-r from-[#3d5a47] to-[#4a6b52]",
                      "disabled:opacity-50 disabled:cursor-not-allowed"
                    )}
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
                </div>
              </div>
            </section>
          ) : rightTab === "evidence" ? (
            <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden h-[calc(100vh-480px)] sm:h-[calc(100vh-380px)] min-h-[280px] sm:min-h-[400px] flex flex-col">
              <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-[#c8e6cf] bg-[#e8f0ea] flex items-center gap-3 flex-shrink-0">
                <IconLightbulb size={20} className="text-[#3d5a47]" />
                <h2 className="text-base sm:text-lg font-bold text-[#3d5a47] tracking-tight">증거수집 전략</h2>
              </div>
              {evidenceGuide ? (
                <div className="flex-1 overflow-y-auto p-4 sm:p-6">
                  <article className="contract-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {evidenceGuide
                        .replace(/^```markdown\n?/i, '')
                        .replace(/\n?```$/i, '')
                        .trim()}
                    </ReactMarkdown>
                  </article>
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center p-6">
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-[#e8f0ea] flex items-center justify-center">
                      <IconLightbulb size={32} className="text-[#3d5a47]" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      계약서 연결 후 확인 가능
                    </h3>
                    <p className="text-sm text-gray-500 max-w-[280px] mx-auto leading-relaxed">
                      계약서를 연결하면 위반 사항과 피해현황을 종합하여 맞춤형 증거수집 전략을 제공합니다.
                    </p>
                  </div>
                </div>
              )}
            </section>
          ) : (
            <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden h-[calc(100vh-480px)] sm:h-[calc(100vh-380px)] min-h-[280px] sm:min-h-[400px] flex flex-col">
              <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-100 bg-gray-50 flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-3">
                  <IconDocument size={20} className="text-[#3d5a47]" />
                  <h2 className="text-base sm:text-lg font-bold text-gray-900 tracking-tight">계약서 연결</h2>
                </div>
                {selectedContract && (
                  <span className="px-3 py-1 text-sm font-medium text-[#3d5a47] bg-[#e8f0ea] rounded-full border border-[#c8e6cf]">
                    연결됨
                  </span>
                )}
              </div>
              <div className="flex-1 overflow-y-auto p-6">
                <p className="text-sm text-gray-600 mb-4">
                  분석 완료된 계약서를 연결하면 위반 사항을 참고하여 더 정확한 내용증명을 작성할 수 있습니다.
                </p>
                {contracts.length === 0 ? (
                  <div className="text-center py-10">
                    <IconDocument size={48} className="mx-auto text-gray-300 mb-4" />
                    <p className="text-base text-gray-500 mb-4">분석 완료된 계약서가 없습니다</p>
                    <button
                      onClick={() => router.push("/")}
                      className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium text-[#3d5a47] border border-[#c8e6cf] rounded-xl hover:bg-[#e8f0ea] transition-all"
                    >
                      <IconUpload size={18} />
                      계약서 업로드하기
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {contracts.map((contract) => (
                      <button
                        key={contract.id}
                        onClick={() => handleSelectContract(contract.id)}
                        className={cn(
                          "w-full p-4 rounded-xl border-2 text-left transition-all hover:shadow-md",
                          selectedContractId === contract.id
                            ? "border-[#3d5a47] bg-[#e8f0ea]"
                            : "border-gray-100 bg-white hover:border-[#3d5a47]/30"
                        )}
                      >
                        <div className="flex items-start gap-3">
                          <div className={cn(
                            "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0",
                            selectedContractId === contract.id
                              ? "bg-[#3d5a47] text-white"
                              : "bg-gray-100 text-gray-400"
                          )}>
                            <IconDocument size={20} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-base font-semibold text-gray-900 truncate">
                              {contract.title}
                            </p>
                            <p className="text-sm text-gray-500 mt-1">
                              {new Date(contract.created_at).toLocaleDateString("ko-KR")}
                            </p>
                          </div>
                          {selectedContractId === contract.id && (
                            <IconCheck size={20} className="text-[#3d5a47] flex-shrink-0" />
                          )}
                        </div>
                      </button>
                    ))}
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
              {isInfoComplete && (
                <button
                  onClick={() => {
                    handleGenerateLegalNotice();
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
    </div>
  );
}
