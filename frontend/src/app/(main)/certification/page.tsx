"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Contract, contractsApi } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  IconLoading,
  IconDocument,
  IconChevronRight,
  IconChevronLeft,
  IconFileText,
  IconPlus,
  IconCheck,
  IconInfo,
  IconUpload,
  IconLightbulb,
  IconShield,
  IconDownload,
} from "@/components/icons";

// Step configuration
const STEPS = [
  { id: 1, label: "피해 현황", description: "상황 파악" },
  { id: 2, label: "계약서 선택", description: "관련 문서" },
  { id: 3, label: "증거 수집 전략", description: "맞춤 가이드" },
  { id: 4, label: "내용증명 작성", description: "문서 생성" },
  { id: 5, label: "완료", description: "다운로드" },
];

// Damage type options
const DAMAGE_TYPES = [
  { id: "payment", label: "대금 미지급", icon: "won" },
  { id: "defect", label: "하자/불량", icon: "warning" },
  { id: "delay", label: "이행 지연", icon: "clock" },
  { id: "breach", label: "계약 위반", icon: "x" },
  { id: "fraud", label: "사기/허위", icon: "alert" },
  { id: "other", label: "기타", icon: "more" },
];

// Situation questions
const SITUATION_QUESTIONS = [
  {
    id: "damage_types",
    type: "multiselect",
    question: "어떤 유형의 피해를 입으셨나요?",
    description: "해당하는 항목을 모두 선택해주세요.",
    options: DAMAGE_TYPES,
  },
  {
    id: "damage_date",
    type: "date",
    question: "피해가 발생한 날짜는 언제인가요?",
    description: "정확한 날짜를 모르시면 대략적인 날짜를 입력해주세요.",
  },
  {
    id: "damage_amount",
    type: "text",
    question: "피해 금액은 얼마인가요?",
    description: "숫자만 입력해주세요. (예: 5000000)",
    placeholder: "5000000",
  },
  {
    id: "damage_description",
    type: "textarea",
    question: "피해 상황을 상세히 설명해주세요.",
    description: "구체적으로 어떤 일이 있었는지 적어주시면 더 정확한 분석이 가능합니다.",
    placeholder: "계약 내용과 실제 상황의 차이, 상대방의 대응 등을 상세히 적어주세요...",
  },
  {
    id: "previous_contact",
    type: "select",
    question: "상대방에게 이미 연락을 취하셨나요?",
    description: "이전에 문제 해결을 위해 연락한 적이 있는지 알려주세요.",
    options: [
      { id: "none", label: "아니오, 연락한 적 없습니다" },
      { id: "verbal", label: "네, 구두(전화/대면)로 연락했습니다" },
      { id: "written", label: "네, 서면(문자/이메일)으로 연락했습니다" },
      { id: "both", label: "네, 구두와 서면 모두 연락했습니다" },
    ],
  },
];

// Tip cards for waiting
const TIPS = [
  {
    title: "내용증명이란?",
    content: "내용증명은 우체국을 통해 발송하는 공식 문서로, 발송 사실과 내용을 법적으로 증명할 수 있습니다.",
    color: "blue",
  },
  {
    title: "증거 수집의 중요성",
    content: "계약서, 대화 기록, 송금 내역 등 관련 증거를 미리 정리해두면 분쟁 해결에 큰 도움이 됩니다.",
    color: "green",
  },
  {
    title: "법적 효력",
    content: "내용증명 자체가 법적 강제력을 갖지는 않지만, 추후 소송 시 중요한 증거로 활용될 수 있습니다.",
    color: "amber",
  },
];

interface SituationData {
  damage_types: string[];
  damage_date: string;
  damage_amount: string;
  damage_description: string;
  previous_contact: string;
}

interface CertificationForm {
  contractId: number | null;
  senderName: string;
  senderAddress: string;
  senderPhone: string;
  receiverName: string;
  receiverAddress: string;
  receiverPhone: string;
  demandItems: string[];
  deadline: string;
}

interface EvidenceStrategy {
  category: string;
  items: { title: string; description: string; priority: "high" | "medium" | "low" }[];
}

export default function CertificationPage() {
  const [loading, setLoading] = useState(true);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [step, setStep] = useState(1);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  const [situationData, setSituationData] = useState<SituationData>({
    damage_types: [],
    damage_date: "",
    damage_amount: "",
    damage_description: "",
    previous_contact: "",
  });

  const [form, setForm] = useState<CertificationForm>({
    contractId: null,
    senderName: "",
    senderAddress: "",
    senderPhone: "",
    receiverName: "",
    receiverAddress: "",
    receiverPhone: "",
    demandItems: [""],
    deadline: "",
  });

  const [evidenceStrategies, setEvidenceStrategies] = useState<EvidenceStrategy[]>([]);
  const [generatedContent, setGeneratedContent] = useState("");

  useEffect(() => {
    loadContracts();
  }, []);

  useEffect(() => {
    if (analyzing) {
      const interval = setInterval(() => {
        setCurrentTipIndex((prev) => (prev + 1) % TIPS.length);
      }, 4000);
      return () => clearInterval(interval);
    }
  }, [analyzing]);

  async function loadContracts() {
    try {
      setLoading(true);
      const data = await contractsApi.list(0, 100);
      setContracts(data.items.filter((c) => c.status === "COMPLETED"));
    } catch {
      // Error handling
    } finally {
      setLoading(false);
    }
  }

  const generateEvidenceStrategy = useCallback(() => {
    const strategies: EvidenceStrategy[] = [];

    if (situationData.damage_types.includes("payment")) {
      strategies.push({
        category: "대금 미지급 관련 증거",
        items: [
          { title: "계약서 원본", description: "계약 금액과 지급 조건이 명시된 계약서", priority: "high" },
          { title: "세금계산서/영수증", description: "거래 사실을 증명하는 세금계산서 또는 영수증", priority: "high" },
          { title: "거래 명세서", description: "납품 또는 서비스 제공 내역을 증명하는 문서", priority: "medium" },
          { title: "입금 내역", description: "일부 입금된 금액이 있다면 해당 내역", priority: "medium" },
        ],
      });
    }

    if (situationData.damage_types.includes("defect")) {
      strategies.push({
        category: "하자/불량 관련 증거",
        items: [
          { title: "하자 사진/영상", description: "문제가 되는 부분의 사진 또는 영상 자료", priority: "high" },
          { title: "전문가 감정서", description: "하자에 대한 전문가의 소견서", priority: "high" },
          { title: "수리 견적서", description: "하자 수리에 소요되는 비용 견적", priority: "medium" },
        ],
      });
    }

    if (situationData.damage_types.includes("delay")) {
      strategies.push({
        category: "이행 지연 관련 증거",
        items: [
          { title: "계약서 이행 기한", description: "계약서에 명시된 이행 기한 조항", priority: "high" },
          { title: "독촉 기록", description: "이행을 독촉한 문자, 이메일, 내용증명 등", priority: "high" },
          { title: "피해 산정 자료", description: "지연으로 인한 피해 금액 산정 근거", priority: "medium" },
        ],
      });
    }

    if (situationData.damage_types.includes("breach")) {
      strategies.push({
        category: "계약 위반 관련 증거",
        items: [
          { title: "계약서 위반 조항", description: "상대방이 위반한 계약서 조항", priority: "high" },
          { title: "위반 사실 증거", description: "계약 위반 사실을 증명하는 자료", priority: "high" },
        ],
      });
    }

    if (situationData.damage_types.includes("fraud")) {
      strategies.push({
        category: "사기/허위 관련 증거",
        items: [
          { title: "허위 진술 증거", description: "상대방의 허위 진술을 증명하는 자료", priority: "high" },
          { title: "피해 금액 증빙", description: "사기로 인한 피해 금액 증빙 자료", priority: "high" },
          { title: "대화 기록", description: "상대방과의 대화 기록", priority: "medium" },
        ],
      });
    }

    strategies.push({
      category: "공통 증거 자료",
      items: [
        { title: "신분증 사본", description: "본인 확인을 위한 신분증 사본", priority: "medium" },
        { title: "대화 기록", description: "상대방과의 모든 대화 기록 보관", priority: "medium" },
        { title: "증인 확보", description: "거래 또는 피해 사실을 아는 증인", priority: "low" },
      ],
    });

    return strategies;
  }, [situationData.damage_types]);

  function handleSituationAnswer(questionId: string, value: string | string[]) {
    setSituationData((prev) => ({ ...prev, [questionId]: value }));
  }

  function handleNextQuestion() {
    if (currentQuestion < SITUATION_QUESTIONS.length - 1) {
      setCurrentQuestion((prev) => prev + 1);
    } else {
      setStep(2);
    }
  }

  function handlePrevQuestion() {
    if (currentQuestion > 0) {
      setCurrentQuestion((prev) => prev - 1);
    }
  }

  function handleSelectContract(id: number) {
    setForm((prev) => ({ ...prev, contractId: id }));
    setAnalyzing(true);
    setTimeout(() => {
      const strategies = generateEvidenceStrategy();
      setEvidenceStrategies(strategies);
      setAnalyzing(false);
      setStep(3);
    }, 3000);
  }

  function handleSkipContract() {
    setAnalyzing(true);
    setTimeout(() => {
      const strategies = generateEvidenceStrategy();
      setEvidenceStrategies(strategies);
      setAnalyzing(false);
      setStep(3);
    }, 2000);
  }

  function handleAddDemandItem() {
    setForm((prev) => ({ ...prev, demandItems: [...prev.demandItems, ""] }));
  }

  function handleRemoveDemandItem(index: number) {
    setForm((prev) => ({ ...prev, demandItems: prev.demandItems.filter((_, i) => i !== index) }));
  }

  function handleDemandItemChange(index: number, value: string) {
    setForm((prev) => ({ ...prev, demandItems: prev.demandItems.map((item, i) => (i === index ? value : item)) }));
  }

  async function handleGenerate() {
    setGenerating(true);
    await new Promise((resolve) => setTimeout(resolve, 3000));

    const selectedContract = contracts.find((c) => c.id === form.contractId);
    const content = `
내 용 증 명

발신인: ${form.senderName}
주  소: ${form.senderAddress}
연락처: ${form.senderPhone}

수신인: ${form.receiverName}
주  소: ${form.receiverAddress}
연락처: ${form.receiverPhone}

제  목: ${selectedContract ? `${selectedContract.title} 관련 ` : ""}계약 이행 요구의 건

본인(${form.senderName})은 귀하(${form.receiverName})에게 다음과 같이 통지합니다.

1. 피해 현황
본인은 ${situationData.damage_date ? new Date(situationData.damage_date).toLocaleDateString("ko-KR") : "최근"}경 귀하와의 거래에서 피해를 입었습니다.
${situationData.damage_amount ? `피해 금액은 ${Number(situationData.damage_amount).toLocaleString()}원입니다.` : ""}

${situationData.damage_description}

2. 요구 사항
${form.demandItems.filter((item) => item.trim()).map((item, idx) => `  ${idx + 1}) ${item}`).join("\n")}

3. 이행 기한
상기 사항에 대하여 ${form.deadline ? new Date(form.deadline).toLocaleDateString("ko-KR") : "본 내용증명 수령일로부터 7일 이내"}까지 이행하여 주시기 바랍니다.

4. 경고
귀하가 위 기한 내에 이행하지 않을 경우, 본인은 민형사상 법적 조치를 취할 것임을 알려드립니다.

${new Date().toLocaleDateString("ko-KR")}

발신인: ${form.senderName} (인)
    `.trim();

    setGeneratedContent(content);
    setGenerating(false);
    setStep(5);
  }

  const selectedContract = contracts.find((c) => c.id === form.contractId);
  const currentQ = SITUATION_QUESTIONS[currentQuestion];

  const isCurrentQuestionAnswered = () => {
    const value = situationData[currentQ.id as keyof SituationData];
    if (currentQ.type === "multiselect") {
      return Array.isArray(value) && value.length > 0;
    }
    return value && value.toString().trim() !== "";
  };

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

  if (analyzing) {
    const tip = TIPS[currentTipIndex];
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] animate-fadeIn">
        {/* Animated loader */}
        <div className="relative mb-10">
          <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-xl shadow-[#3d5a47]/30">
            <IconLoading size={44} className="text-white" />
          </div>
          <div className="absolute -inset-2 rounded-3xl border-2 border-[#3d5a47]/20 animate-pulse" />
        </div>

        <h2 className="text-3xl font-bold text-gray-900 tracking-tight mb-3">계약서 분석 중...</h2>
        <p className="text-base text-gray-500 mb-10">맞춤형 증거 수집 전략을 생성하고 있습니다</p>

        {/* Tip card */}
        <div className={cn(
          "max-w-md w-full p-6 rounded-2xl border-2 transition-all duration-500 shadow-lg",
          tip.color === "blue" && "bg-gradient-to-br from-blue-50 to-white border-blue-200/50",
          tip.color === "green" && "bg-gradient-to-br from-[#e8f5ec] to-white border-[#c8e6cf]",
          tip.color === "amber" && "bg-gradient-to-br from-[#fef7e0] to-white border-[#f5e6b8]"
        )}>
          <div className="flex items-start gap-4">
            <div className={cn(
              "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0",
              tip.color === "blue" && "bg-blue-500",
              tip.color === "green" && "bg-[#4a9a5b]",
              tip.color === "amber" && "bg-[#d4a84d]"
            )}>
              <IconInfo size={20} className="text-white" />
            </div>
            <div>
              <p className={cn(
                "text-base font-bold mb-1.5",
                tip.color === "blue" && "text-blue-700",
                tip.color === "green" && "text-[#3d7a4a]",
                tip.color === "amber" && "text-[#9a7b2d]"
              )}>{tip.title}</p>
              <p className="text-base text-gray-600 leading-relaxed">{tip.content}</p>
            </div>
          </div>
        </div>

        {/* Progress dots */}
        <div className="flex gap-2 mt-8">
          {TIPS.map((_, idx) => (
            <div
              key={idx}
              className={cn(
                "h-2 rounded-full transition-all duration-300",
                idx === currentTipIndex
                  ? "bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] w-8"
                  : "bg-gray-200 w-2"
              )}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-10">
      {/* Header with gradient accent */}
      <section className="relative">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-[#3d5a47]/10 to-transparent rounded-full blur-2xl" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
              <IconShield size={20} className="text-white" />
            </div>
            <span className="text-sm font-medium text-[#3d5a47] tracking-tight">Legal Document</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">내용증명 및 증거 수집</h1>
          <p className="text-base text-gray-500 mt-2">AI가 맞춤형 내용증명 문서를 생성해드립니다</p>
        </div>
      </section>

      {/* Enhanced Step Indicator */}
      <section className="liquid-glass rounded-2xl p-6 border border-white/40">
        <div className="flex items-center justify-between">
          {STEPS.map((s, idx) => (
            <div key={s.id} className="flex items-center flex-1">
              <div className="flex flex-col items-center group">
                <div className={cn(
                  "w-14 h-14 rounded-2xl flex items-center justify-center text-lg font-bold transition-all duration-300 shadow-lg",
                  step > s.id
                    ? "bg-gradient-to-br from-[#4a9a5b] to-[#3d7a4a] text-white shadow-[#4a9a5b]/30"
                    : step === s.id
                      ? "bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] text-white shadow-[#3d5a47]/30 scale-110"
                      : "bg-white/80 text-gray-400 shadow-gray-200/50 border border-gray-100"
                )}>
                  {step > s.id ? <IconCheck size={24} /> : s.id}
                </div>
                <div className="mt-4 text-center">
                  <p className={cn(
                    "text-base font-semibold tracking-tight transition-colors",
                    step >= s.id ? "text-gray-900" : "text-gray-400"
                  )}>{s.label}</p>
                  <p className={cn(
                    "text-sm mt-1 hidden sm:block transition-colors",
                    step === s.id ? "text-[#3d5a47]" : "text-gray-400"
                  )}>{s.description}</p>
                </div>
              </div>
              {idx < STEPS.length - 1 && (
                <div className="flex-1 h-1.5 mx-4 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      step > s.id ? "bg-gradient-to-r from-[#4a9a5b] to-[#3d7a4a] w-full" : "w-0"
                    )}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {step === 1 && (
        <section className="animate-fadeIn">
          <div className="liquid-glass rounded-2xl p-8 border border-white/40">
            {/* Progress header */}
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center">
                  <span className="text-white font-bold text-sm">{currentQuestion + 1}</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-900">질문 {currentQuestion + 1}</span>
                  <span className="text-sm text-gray-400"> / {SITUATION_QUESTIONS.length}</span>
                </div>
              </div>
              <div className="flex-1 max-w-xs ml-6">
                <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${((currentQuestion + 1) / SITUATION_QUESTIONS.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Question */}
            <div className="mb-10">
              <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-3">{currentQ.question}</h2>
              <p className="text-base text-gray-500">{currentQ.description}</p>
            </div>

            {/* Answer options */}
            <div className="mb-10">
              {currentQ.type === "multiselect" && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {DAMAGE_TYPES.map((type) => {
                    const isSelected = situationData.damage_types.includes(type.id);
                    return (
                      <button
                        key={type.id}
                        onClick={() => {
                          const current = situationData.damage_types;
                          const updated = current.includes(type.id) ? current.filter((t) => t !== type.id) : [...current, type.id];
                          handleSituationAnswer("damage_types", updated);
                        }}
                        className={cn(
                          "p-5 rounded-2xl border-2 text-left transition-all duration-200 group relative overflow-hidden",
                          isSelected
                            ? "border-[#3d5a47] bg-gradient-to-br from-[#e8f0ea] to-[#d8e8dc] shadow-lg shadow-[#3d5a47]/10"
                            : "border-gray-100 bg-white hover:border-[#3d5a47]/30 hover:shadow-md"
                        )}
                      >
                        {isSelected && (
                          <div className="absolute top-3 right-3">
                            <div className="w-5 h-5 rounded-full bg-[#3d5a47] flex items-center justify-center">
                              <IconCheck size={12} className="text-white" />
                            </div>
                          </div>
                        )}
                        <div className={cn(
                          "w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-all duration-200",
                          isSelected
                            ? "bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] text-white shadow-lg shadow-[#3d5a47]/30"
                            : "bg-gray-100 text-gray-400 group-hover:bg-gray-200"
                        )}>
                          {type.icon === "won" && <span className="text-xl font-bold">W</span>}
                          {type.icon === "warning" && <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>}
                          {type.icon === "clock" && <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><circle cx="12" cy="12" r="10" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2" /></svg>}
                          {type.icon === "x" && <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>}
                          {type.icon === "alert" && <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
                          {type.icon === "more" && <svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" /></svg>}
                        </div>
                        <p className={cn(
                          "text-base font-semibold tracking-tight transition-colors",
                          isSelected ? "text-[#3d5a47]" : "text-gray-900"
                        )}>{type.label}</p>
                      </button>
                    );
                  })}
                </div>
              )}
              {currentQ.type === "date" && (
                <input
                  type="date"
                  value={situationData.damage_date}
                  onChange={(e) => handleSituationAnswer("damage_date", e.target.value)}
                  className="w-full max-w-sm px-5 py-4 text-lg bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all"
                />
              )}
              {currentQ.type === "text" && (
                <input
                  type="text"
                  value={situationData[currentQ.id as keyof SituationData] as string}
                  onChange={(e) => handleSituationAnswer(currentQ.id, e.target.value)}
                  placeholder={currentQ.placeholder}
                  className="w-full max-w-md px-5 py-4 text-lg bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all"
                />
              )}
              {currentQ.type === "textarea" && (
                <textarea
                  value={situationData[currentQ.id as keyof SituationData] as string}
                  onChange={(e) => handleSituationAnswer(currentQ.id, e.target.value)}
                  placeholder={currentQ.placeholder}
                  rows={5}
                  className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all resize-none"
                />
              )}
              {currentQ.type === "select" && currentQ.options && (
                <div className="space-y-3">
                  {currentQ.options.map((option) => {
                    const isSelected = situationData[currentQ.id as keyof SituationData] === option.id;
                    return (
                      <button
                        key={option.id}
                        onClick={() => handleSituationAnswer(currentQ.id, option.id)}
                        className={cn(
                          "w-full p-5 rounded-xl border-2 text-left transition-all duration-200 flex items-center gap-4",
                          isSelected
                            ? "border-[#3d5a47] bg-gradient-to-r from-[#e8f0ea] to-[#d8e8dc] shadow-lg shadow-[#3d5a47]/10"
                            : "border-gray-100 bg-white hover:border-[#3d5a47]/30 hover:shadow-md"
                        )}
                      >
                        <div className={cn(
                          "w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all",
                          isSelected ? "border-[#3d5a47] bg-[#3d5a47]" : "border-gray-300"
                        )}>
                          {isSelected && <IconCheck size={14} className="text-white" />}
                        </div>
                        <span className={cn(
                          "text-base font-medium transition-colors",
                          isSelected ? "text-[#3d5a47]" : "text-gray-900"
                        )}>{option.label}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-between items-center pt-4 border-t border-gray-100">
              <button
                onClick={handlePrevQuestion}
                disabled={currentQuestion === 0}
                className="flex items-center gap-2 px-5 py-3 text-base font-medium text-gray-600 hover:text-gray-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors rounded-xl hover:bg-gray-50"
              >
                <IconChevronLeft size={20} />이전
              </button>
              <button
                onClick={handleNextQuestion}
                disabled={!isCurrentQuestionAnswered()}
                className={cn(
                  "flex items-center gap-2 px-8 py-3.5 text-base font-semibold text-white rounded-xl transition-all duration-200",
                  "bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5",
                  "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none disabled:hover:translate-y-0"
                )}
              >
                {currentQuestion === SITUATION_QUESTIONS.length - 1 ? "다음 단계" : "다음"}
                <IconChevronRight size={20} />
              </button>
            </div>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="animate-fadeIn">
          <div className="liquid-glass rounded-2xl p-8 border border-white/40">
            <div className="flex items-start gap-4 mb-8">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                <IconDocument size={24} className="text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-2">관련 계약서 선택</h2>
                <p className="text-base text-gray-500">피해와 관련된 계약서를 선택하면 더 정확한 분석이 가능합니다.</p>
              </div>
            </div>

            {contracts.length === 0 ? (
              <div className="text-center py-16 mb-8 bg-gradient-to-b from-gray-50/50 to-transparent rounded-2xl border border-dashed border-gray-200">
                <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#e8f0ea] to-[#d8e8dc] rounded-2xl mb-6">
                  <IconDocument size={36} className="text-[#3d5a47]" />
                </div>
                <p className="text-xl text-gray-800 mb-2 font-semibold tracking-tight">분석 완료된 계약서가 없습니다</p>
                <p className="text-base text-gray-500">계약서 없이 진행하거나, 새 계약서를 업로드하세요</p>
              </div>
            ) : (
              <div className="space-y-3 mb-8">
                {contracts.map((contract) => (
                  <button
                    key={contract.id}
                    onClick={() => handleSelectContract(contract.id)}
                    className="w-full p-5 bg-white/80 border-2 border-gray-100 rounded-xl text-left group hover:border-[#3d5a47] hover:shadow-lg hover:shadow-[#3d5a47]/10 transition-all duration-200"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-4 flex-1 min-w-0">
                        <div className="flex-shrink-0 w-14 h-14 rounded-xl flex items-center justify-center bg-gradient-to-br from-[#e8f0ea] to-[#d8e8dc] text-[#3d5a47] group-hover:from-[#3d5a47] group-hover:to-[#4a6b52] group-hover:text-white transition-all duration-200">
                          <IconDocument size={26} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-lg font-semibold text-gray-900 truncate tracking-tight group-hover:text-[#3d5a47] transition-colors">{contract.title}</p>
                          <p className="text-base text-gray-500 mt-1">{new Date(contract.created_at).toLocaleDateString("ko-KR")}</p>
                        </div>
                      </div>
                      <div className="w-10 h-10 rounded-full bg-gray-50 group-hover:bg-[#3d5a47] flex items-center justify-center transition-all duration-200">
                        <IconChevronRight size={20} className="text-gray-400 group-hover:text-white transition-colors" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t border-gray-100">
              <button
                onClick={() => setStep(1)}
                className="flex items-center justify-center gap-2 px-6 py-4 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
              >
                <IconChevronLeft size={20} />이전 단계
              </button>
              <Link
                href="/"
                className="flex items-center justify-center gap-2 px-6 py-4 text-base font-semibold text-[#3d5a47] border-2 border-[#3d5a47] hover:bg-[#e8f0ea] rounded-xl transition-all"
              >
                <IconUpload size={20} />새 계약서 업로드
              </Link>
              <button
                onClick={handleSkipContract}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-4 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 rounded-xl transition-all duration-200"
              >
                계약서 없이 진행<IconChevronRight size={20} />
              </button>
            </div>
          </div>
        </section>
      )}

      {step === 3 && (
        <section className="animate-fadeIn space-y-6">
          <div className="liquid-glass rounded-2xl p-8 border border-white/40">
            <div className="flex items-start justify-between gap-4 mb-8">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                  <IconLightbulb size={24} className="text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-2">맞춤형 증거 수집 전략</h2>
                  <p className="text-base text-gray-500">선택하신 피해 유형에 따라 수집이 필요한 증거 목록입니다.</p>
                </div>
              </div>
              {selectedContract && (
                <div className="hidden sm:flex items-center gap-3 px-4 py-2.5 bg-gradient-to-r from-[#e8f0ea] to-[#d8e8dc] rounded-xl border border-[#c8e6cf]">
                  <IconDocument size={18} className="text-[#3d5a47]" />
                  <span className="text-base font-semibold text-[#3d5a47]">{selectedContract.title}</span>
                </div>
              )}
            </div>

            <div className="space-y-6 mb-8">
              {evidenceStrategies.map((strategy, idx) => (
                <div key={idx} className="bg-white/60 border border-gray-100 rounded-2xl overflow-hidden">
                  <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-100">
                    <h3 className="text-lg font-bold text-gray-900 tracking-tight">{strategy.category}</h3>
                  </div>
                  <div className="p-5 space-y-3">
                    {strategy.items.map((item, itemIdx) => (
                      <div
                        key={itemIdx}
                        className={cn(
                          "flex items-start gap-4 p-4 rounded-xl border-2 transition-all hover:shadow-md",
                          item.priority === "high" && "bg-gradient-to-r from-[#fdedec]/50 to-white border-[#f5c6c4]/50",
                          item.priority === "medium" && "bg-gradient-to-r from-[#fef7e0]/50 to-white border-[#f5e6b8]/50",
                          item.priority === "low" && "bg-gradient-to-r from-gray-50/50 to-white border-gray-100"
                        )}
                      >
                        <div className={cn(
                          "w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm",
                          item.priority === "high" && "bg-gradient-to-br from-[#c94b45] to-[#b54a45] text-white",
                          item.priority === "medium" && "bg-gradient-to-br from-[#d4a84d] to-[#9a7b2d] text-white",
                          item.priority === "low" && "bg-gradient-to-br from-gray-400 to-gray-500 text-white"
                        )}>
                          {item.priority === "high" && <span className="text-base font-bold">!</span>}
                          {item.priority === "medium" && <span className="text-base font-bold">-</span>}
                          {item.priority === "low" && <span className="text-base font-bold">+</span>}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-1.5">
                            <p className="text-base font-semibold text-gray-900 tracking-tight">{item.title}</p>
                            <span className={cn(
                              "text-sm font-bold px-2.5 py-1 rounded-full",
                              item.priority === "high" && "bg-[#fdedec] text-[#b54a45] border border-[#f5c6c4]",
                              item.priority === "medium" && "bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]",
                              item.priority === "low" && "bg-gray-100 text-gray-500 border border-gray-200"
                            )}>
                              {item.priority === "high" && "필수"}
                              {item.priority === "medium" && "권장"}
                              {item.priority === "low" && "선택"}
                            </span>
                          </div>
                          <p className="text-base text-gray-600">{item.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t border-gray-100">
              <button
                onClick={() => setStep(2)}
                className="flex items-center justify-center gap-2 px-6 py-4 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
              >
                <IconChevronLeft size={20} />이전 단계
              </button>
              <button
                onClick={() => setStep(4)}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-4 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 rounded-xl transition-all duration-200"
              >
                <IconFileText size={20} />내용증명 작성하기
              </button>
            </div>
          </div>
        </section>
      )}

      {step === 4 && (
        <section className="animate-fadeIn">
          <div className="liquid-glass rounded-2xl p-8 border border-white/40">
            <div className="flex items-start gap-4 mb-8">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                <IconFileText size={24} className="text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-2">내용증명 정보 입력</h2>
                <p className="text-base text-gray-500">내용증명에 필요한 정보를 입력해주세요.</p>
              </div>
            </div>

            <div className="space-y-6">
              {/* 발신인 정보 */}
              <div className="bg-white/60 border border-gray-100 rounded-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-[#e8f0ea] to-[#d8e8dc] px-6 py-4 border-b border-[#c8e6cf]">
                  <h3 className="text-lg font-bold text-[#3d5a47] tracking-tight">발신인 정보</h3>
                </div>
                <div className="p-6 grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div>
                    <label className="block text-base font-semibold text-gray-700 mb-2">성명/상호</label>
                    <input type="text" value={form.senderName} onChange={(e) => setForm((prev) => ({ ...prev, senderName: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="홍길동" />
                  </div>
                  <div>
                    <label className="block text-base font-semibold text-gray-700 mb-2">연락처</label>
                    <input type="text" value={form.senderPhone} onChange={(e) => setForm((prev) => ({ ...prev, senderPhone: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="010-1234-5678" />
                  </div>
                  <div className="sm:col-span-2">
                    <label className="block text-base font-semibold text-gray-700 mb-2">주소</label>
                    <input type="text" value={form.senderAddress} onChange={(e) => setForm((prev) => ({ ...prev, senderAddress: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="서울시 강남구 테헤란로 123" />
                  </div>
                </div>
              </div>

              {/* 수신인 정보 */}
              <div className="bg-white/60 border border-gray-100 rounded-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-[#fef7e0] to-[#fef3c7] px-6 py-4 border-b border-[#f5e6b8]">
                  <h3 className="text-lg font-bold text-[#9a7b2d] tracking-tight">수신인 정보</h3>
                </div>
                <div className="p-6 grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div>
                    <label className="block text-base font-semibold text-gray-700 mb-2">성명/상호</label>
                    <input type="text" value={form.receiverName} onChange={(e) => setForm((prev) => ({ ...prev, receiverName: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="(주)OO회사" />
                  </div>
                  <div>
                    <label className="block text-base font-semibold text-gray-700 mb-2">연락처</label>
                    <input type="text" value={form.receiverPhone} onChange={(e) => setForm((prev) => ({ ...prev, receiverPhone: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="02-1234-5678" />
                  </div>
                  <div className="sm:col-span-2">
                    <label className="block text-base font-semibold text-gray-700 mb-2">주소</label>
                    <input type="text" value={form.receiverAddress} onChange={(e) => setForm((prev) => ({ ...prev, receiverAddress: e.target.value }))} className="w-full px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="서울시 서초구 강남대로 456" />
                  </div>
                </div>
              </div>

              {/* 요구 사항 */}
              <div className="bg-white/60 border border-gray-100 rounded-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-100 flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-900 tracking-tight">요구 사항</h3>
                  <button onClick={handleAddDemandItem} className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-[#3d5a47] bg-[#e8f0ea] hover:bg-[#d8e8dc] rounded-lg transition-colors">
                    <IconPlus size={16} />항목 추가
                  </button>
                </div>
                <div className="p-6 space-y-4">
                  {form.demandItems.map((item, index) => (
                    <div key={index} className="flex items-center gap-3 group">
                      <span className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center text-sm font-bold text-white flex-shrink-0 shadow-sm">{index + 1}</span>
                      <input type="text" value={item} onChange={(e) => handleDemandItemChange(index, e.target.value)} className="flex-1 px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" placeholder="요구 사항을 입력하세요" />
                      {form.demandItems.length > 1 && (
                        <button onClick={() => handleRemoveDemandItem(index)} className="w-10 h-10 rounded-xl bg-gray-100 hover:bg-[#fdedec] text-gray-400 hover:text-[#b54a45] flex items-center justify-center transition-all opacity-0 group-hover:opacity-100">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* 이행 기한 */}
              <div className="bg-white/60 border border-gray-100 rounded-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-100">
                  <h3 className="text-lg font-bold text-gray-900 tracking-tight">이행 기한</h3>
                </div>
                <div className="p-6">
                  <input type="date" value={form.deadline} onChange={(e) => setForm((prev) => ({ ...prev, deadline: e.target.value }))} className="w-full max-w-sm px-5 py-4 text-base bg-white/80 border-2 border-gray-100 rounded-xl outline-none focus:border-[#3d5a47] focus:ring-4 focus:ring-[#3d5a47]/10 transition-all" />
                </div>
              </div>

              {/* 안내 메시지 */}
              <div className="flex items-start gap-4 p-5 bg-gradient-to-r from-[#e8f0ea] to-[#d8e8dc] rounded-xl border border-[#c8e6cf]">
                <div className="w-12 h-12 rounded-xl bg-[#3d5a47] flex items-center justify-center flex-shrink-0">
                  <IconInfo size={22} className="text-white" />
                </div>
                <div>
                  <p className="text-base font-semibold text-[#3d5a47] mb-1">AI 자동 생성</p>
                  <p className="text-base text-[#3d7a4a] leading-relaxed">AI가 입력하신 정보와 피해 현황을 바탕으로 내용증명 초안을 작성합니다.</p>
                </div>
              </div>

              {/* 버튼 */}
              <div className="flex flex-col-reverse sm:flex-row gap-4 pt-4 border-t border-gray-100">
                <button onClick={() => setStep(3)} className="flex items-center justify-center gap-2 px-6 py-4 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">
                  <IconChevronLeft size={20} />이전 단계
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={generating || !form.senderName || !form.receiverName}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 px-8 py-4 text-base font-semibold text-white rounded-xl transition-all duration-200",
                    "bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5",
                    "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none disabled:hover:translate-y-0"
                  )}
                >
                  {generating ? <><IconLoading size={20} />생성 중...</> : <><IconFileText size={20} />내용증명 생성</>}
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {step === 5 && (
        <section className="animate-fadeIn space-y-6">
          {/* Success Header */}
          <div className="liquid-glass rounded-2xl p-10 text-center border border-white/40 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-[#4a9a5b]/5 to-transparent" />
            <div className="relative">
              <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-[#4a9a5b] to-[#3d7a4a] rounded-3xl mb-6 shadow-lg shadow-[#4a9a5b]/30">
                <IconCheck size={48} className="text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 tracking-tight mb-3">내용증명 생성 완료</h2>
              <p className="text-base text-gray-500">아래에서 생성된 내용증명을 확인하고 다운로드할 수 있습니다.</p>
            </div>
          </div>

          {/* Preview */}
          <div className="liquid-glass rounded-2xl p-8 border border-white/40">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center">
                  <IconFileText size={20} className="text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 tracking-tight">미리보기</h3>
              </div>
              <button onClick={() => setStep(4)} className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-[#3d5a47] bg-[#e8f0ea] hover:bg-[#d8e8dc] rounded-lg transition-colors">
                수정하기
              </button>
            </div>
            <div className="p-8 bg-white border-2 border-gray-100 rounded-xl font-mono text-base whitespace-pre-wrap leading-relaxed shadow-inner">
              {generatedContent}
            </div>
          </div>

          {/* Actions */}
          <div className="liquid-glass rounded-2xl p-6 border border-white/40">
            <div className="flex flex-col sm:flex-row gap-4">
              <button onClick={() => setStep(4)} className="flex items-center justify-center gap-2 px-6 py-4 text-base font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">
                수정하기
              </button>
              <button
                onClick={() => alert("PDF 다운로드 기능은 백엔드 연결 후 구현 예정입니다.")}
                className="flex-1 flex items-center justify-center gap-2 px-8 py-4 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 rounded-xl transition-all duration-200"
              >
                <IconDownload size={20} />PDF 다운로드
              </button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
