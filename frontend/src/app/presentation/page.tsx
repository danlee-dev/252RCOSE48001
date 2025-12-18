"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  Maximize2,
  Minimize2,
  Home,
} from "lucide-react";

// Simple icon components
const DatabaseIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
  </svg>
);

const GitBranchIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="18" cy="18" r="3" />
    <circle cx="6" cy="6" r="3" />
    <path d="M6 21V9a9 9 0 0 0 9 9" />
  </svg>
);

// Slide data structure
interface Slide {
  id: number;
  type: "title" | "content" | "split" | "diagram" | "summary";
  title?: string;
  subtitle?: string;
  content?: React.ReactNode;
  leftContent?: React.ReactNode;
  rightContent?: React.ReactNode;
  background?: string;
}

export default function PresentationPage() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isAutoPlay, setIsAutoPlay] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [showDemoIframe, setShowDemoIframe] = useState(false);
  const [isKorean, setIsKorean] = useState(false);

  // Translation helper
  const t = (en: string, ko: string) => isKorean ? ko : en;

  // Helper to determine if current slide has light background
  const isLightBackground = (bg: string) => {
    return bg.includes("bg-white") || bg.includes("bg-slate-50") || bg.includes("bg-slate-100");
  };

  const slides: Slide[] = [
    // Slide 1: Title
    {
      id: 1,
      type: "title",
      background: "bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900",
    },
    // Slide 2: Problem Statement
    {
      id: 2,
      type: "content",
      title: "Problem Statement",
      subtitle: "Why Traditional Contract Review Fails",
      background: "bg-white",
    },
    // Slide 3: Solution Overview
    {
      id: 3,
      type: "diagram",
      title: "DocScanner AI Solution",
      subtitle: "AI-Powered Legal Document Analysis Pipeline",
      background: "bg-slate-50",
    },
    // Slide 4: Hybrid DB Architecture
    {
      id: 4,
      type: "split",
      title: "Hybrid DB Architecture",
      subtitle: "Vector DB + Graph DB for Comprehensive Legal Retrieval",
      background: "bg-white",
    },
    // Slide 5: MUVERA Embedding
    {
      id: 5,
      type: "split",
      title: "Perception I: MUVERA Embedding",
      subtitle:
        "Multi-Vector Retrieval via Fixed Dimensional Encodings (Google Research, NeurIPS 2024)",
      background: "bg-slate-50",
    },
    // Slide 6: RAPTOR
    {
      id: 6,
      type: "split",
      title: "Perception II: RAPTOR Indexing",
      subtitle: "Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024)",
      background: "bg-white",
    },
    // Slide 7: HyDE & CRAG & Knowledge Graph
    {
      id: 7,
      type: "split",
      title: "Perception III: HyDE & CRAG & Knowledge Graph",
      subtitle: "Advanced Retrieval Enhancement for Legal Context",
      background: "bg-slate-50",
    },
    // Slide 8: Neuro-Symbolic AI & Constitutional AI (Combined)
    {
      id: 8,
      type: "split",
      title: "Reasoning & Action: Neuro-Symbolic + Constitutional AI",
      subtitle: "Accurate Calculation meets Ethical Self-Correction",
      background: "bg-white",
    },
    // Slide 9: System Architecture
    {
      id: 9,
      type: "diagram",
      title: "System Architecture",
      subtitle: "Full-Stack Implementation Overview",
      background: "bg-slate-50",
    },
    // Slide 10: Data & Results
    {
      id: 10,
      type: "content",
      title: "Data & Implementation",
      subtitle: "Built Knowledge Base Statistics",
      background: "bg-white",
    },
    // Slide 11: Live Demo
    {
      id: 11,
      type: "diagram",
      title: "Live Demo",
      subtitle: "Real-time Contract Analysis System",
      background: "bg-slate-900",
    },
    // Slide 12: Discussion - Future Improvements
    {
      id: 12,
      type: "split",
      title: "Discussion: Future Improvements",
      subtitle: "Advanced Reranking for Legal Authority",
      background: "bg-slate-50",
    },
    // Slide 13: Evaluation & Cost
    {
      id: 13,
      type: "split",
      title: "Evaluation & Cost",
      subtitle: "System Performance Analysis and Economic Efficiency",
      background: "bg-white",
    },
    // Slide 14: Pipeline Flow Diagram (Example Scenario)
    {
      id: 14,
      type: "diagram",
      title: "Analysis Pipeline Flow",
      subtitle: "Step-by-Step Contract Analysis Example",
      background: "bg-white",
    },
    // Slide 15: MUVERA Technical Diagram
    {
      id: 15,
      type: "diagram",
      title: "MUVERA Embedding",
      subtitle: "Multi-Vector Retrieval via Fixed Dimensional Encodings",
      background: "bg-white",
    },
    // Slide 16: Full System Architecture
    {
      id: 16,
      type: "diagram",
      title: "System Architecture",
      subtitle: "DocScanner.ai Full-Stack Implementation",
      background: "bg-white",
    },
    // Slide 17: Hybrid RAG Architecture Diagram
    {
      id: 17,
      type: "diagram",
      title: "Hybrid RAG Architecture",
      subtitle: "Vector DB + Knowledge Graph Synergy",
      background: "bg-white",
    },
    // Slide 18: Conclusion
    {
      id: 18,
      type: "title",
      background: "bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900",
    },
  ];

  const goToSlide = useCallback(
    (index: number) => {
      if (index >= 0 && index < slides.length) {
        setCurrentSlide(index);
      }
    },
    [slides.length]
  );

  const nextSlide = useCallback(() => {
    goToSlide(currentSlide + 1);
  }, [currentSlide, goToSlide]);

  const prevSlide = useCallback(() => {
    goToSlide(currentSlide - 1);
  }, [currentSlide, goToSlide]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        nextSlide();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        prevSlide();
      } else if (e.key === "f" || e.key === "F") {
        toggleFullscreen();
      } else if (e.key === "Escape") {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [nextSlide, prevSlide]);

  // Auto-play
  useEffect(() => {
    if (isAutoPlay) {
      const timer = setInterval(() => {
        if (currentSlide < slides.length - 1) {
          nextSlide();
        } else {
          setIsAutoPlay(false);
        }
      }, 30000); // 30 seconds per slide for 10-min presentation
      return () => clearInterval(timer);
    }
  }, [isAutoPlay, currentSlide, nextSlide, slides.length]);

  // Fullscreen handling
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Hide controls after inactivity
  useEffect(() => {
    let timeout: NodeJS.Timeout;
    const handleMouseMove = () => {
      setShowControls(true);
      clearTimeout(timeout);
      timeout = setTimeout(() => setShowControls(false), 3000);
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      clearTimeout(timeout);
    };
  }, []);

  const renderSlideContent = (slide: Slide, index: number) => {
    switch (index) {
      case 0: // Title Slide
        return (
          <div className="flex flex-col items-center justify-center h-full text-white px-8 relative overflow-hidden">
            {/* Subtle background elements */}
            <div className="absolute inset-0 pointer-events-none">
              {/* Gradient orb */}
              <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-teal-500/10 via-cyan-500/5 to-transparent rounded-full blur-3xl" />
              {/* Grid pattern */}
              <div className="absolute inset-0 opacity-[0.03]" style={{
                backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                                  linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
                backgroundSize: '60px 60px'
              }} />
              {/* Corner accents */}
              <div className="absolute top-12 left-12 w-24 h-[1px] bg-gradient-to-r from-teal-500/50 to-transparent" />
              <div className="absolute top-12 left-12 w-[1px] h-24 bg-gradient-to-b from-teal-500/50 to-transparent" />
              <div className="absolute bottom-12 right-12 w-24 h-[1px] bg-gradient-to-l from-teal-500/50 to-transparent" />
              <div className="absolute bottom-12 right-12 w-[1px] h-24 bg-gradient-to-t from-teal-500/50 to-transparent" />
            </div>

            {/* Content */}
            <div className="relative z-10">
              <p
                className="text-sm tracking-[0.3em] text-slate-500 uppercase mb-8 text-center animate-fadeInUp"
                style={{ animationDelay: "0.05s" }}
              >
                {t("Korea University / COSE361", "고려대학교 / COSE361 인공지능")}
              </p>
              <h1
                className="text-5xl md:text-7xl font-bold mb-6 text-center animate-fadeInUp"
                style={{ animationDelay: "0.1s" }}
              >
                AI Legal Guardian
              </h1>
              <div className="w-24 h-[2px] bg-gradient-to-r from-transparent via-teal-400 to-transparent mx-auto mb-8 animate-fadeInUp" style={{ animationDelay: "0.15s" }} />
              <p
                className="text-xl md:text-2xl text-slate-400 mb-12 text-center max-w-3xl animate-fadeInUp"
                style={{ animationDelay: "0.2s" }}
              >
                {t("An Agent System for Contract Assistance for the Vulnerable", "사회적 약자를 위한 계약서 검토 에이전트 시스템")}
              </p>
              <p
                className="text-lg text-teal-400 animate-fadeInUp"
                style={{ animationDelay: "0.3s" }}
              >
                {t("2023320132, Seongmin Lee", "2023320132 이성민")}
              </p>
            </div>
          </div>
        );

      case 1: // Problem Statement
        return (
          <div className="flex flex-col justify-center h-full px-16 py-8">
            <div className="mb-8">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("The Challenge", "문제 정의")}
              </p>
              <h2 className="text-4xl font-bold text-slate-900 mb-3">
                {t("Problem Statement", "문제 상황")}
              </h2>
              <p className="text-xl text-slate-500">{t("Why Traditional Contract Review Fails", "기존 계약서 검토 방식의 한계")}</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="bg-red-50 rounded-2xl p-8 border border-red-100">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mb-6">
                  <span className="text-2xl">1</span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-4">
                  {t("Information Asymmetry", "정보 비대칭")}
                </h3>
                <ul className="space-y-3 text-slate-600">
                  <li className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">-</span>
                    {t("Workers lack legal expertise to identify unfair clauses", "근로자는 불공정 조항을 식별할 법률 전문 지식이 부족함")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">-</span>
                    {t("Legal consultation is expensive and time-consuming", "법률 상담은 비용이 많이 들고 시간이 오래 걸림")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">-</span>
                    {t("Critical issues discovered too late", "중요한 문제가 너무 늦게 발견됨")}
                  </li>
                </ul>
              </div>
              <div className="bg-amber-50 rounded-2xl p-8 border border-amber-100">
                <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center mb-6">
                  <span className="text-2xl">2</span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-4">
                  {t("Complex Legal Standards", "복잡한 법적 기준")}
                </h3>
                <ul className="space-y-3 text-slate-600">
                  <li className="flex items-start gap-2">
                    <span className="text-amber-500 mt-1">-</span>
                    {t("11+ quantitative checkpoints in Korean Labor Law", "근로기준법의 11개 이상 정량적 검증 항목")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-amber-500 mt-1">-</span>
                    {t("Annual updates (e.g., 2025 minimum wage: 10,030 KRW)", "연간 업데이트 (예: 2025 최저임금 10,030원)")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-amber-500 mt-1">-</span>
                    {t("Cross-clause dependencies require holistic analysis", "조항 간 의존성으로 인해 전체적 분석 필요")}
                  </li>
                </ul>
              </div>
              <div className="bg-blue-50 rounded-2xl p-8 border border-blue-100">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-6">
                  <span className="text-2xl">3</span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-4">
                  {t("Traditional AI Limitations", "기존 AI의 한계")}
                </h3>
                <ul className="space-y-3 text-slate-600">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-500 mt-1">-</span>
                    {t("Regex-based systems miss semantic violations", "정규식 기반 시스템은 의미적 위반을 놓침")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-500 mt-1">-</span>
                    {t("Pure LLMs hallucinate legal citations", "순수 LLM은 법률 인용을 환각함")}
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-500 mt-1">-</span>
                    {t("No verifiable reasoning trace", "검증 가능한 추론 과정 없음")}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        );

      case 2: // Solution Overview
        return (
          <div className="flex h-full">
            {/* Left: Circular Agent Diagram */}
            <div className="w-2/5 bg-slate-900 p-4 flex items-center justify-center overflow-hidden">
              <div className="relative w-[400px] h-[400px]">
                {/* Circular arrows (SVG) - aligned with node edges */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 400">
                  {/* Top (Perception) to Right (Reasoning) */}
                  <path
                    d="M 200 50 A 150 150 0 0 1 350 200"
                    fill="none"
                    stroke="rgba(16, 185, 129, 0.5)"
                    strokeWidth="2"
                  />
                  {/* Right (Reasoning) to Bottom (Action) */}
                  <path
                    d="M 350 200 A 150 150 0 0 1 200 350"
                    fill="none"
                    stroke="rgba(245, 158, 11, 0.5)"
                    strokeWidth="2"
                  />
                  {/* Bottom (Action) to Left (Environment) */}
                  <path
                    d="M 200 350 A 150 150 0 0 1 50 200"
                    fill="none"
                    stroke="rgba(244, 63, 94, 0.5)"
                    strokeWidth="2"
                  />
                  {/* Left (Environment) to Top (Perception) */}
                  <path
                    d="M 50 200 A 150 150 0 0 1 200 50"
                    fill="none"
                    stroke="rgba(99, 102, 241, 0.5)"
                    strokeWidth="2"
                  />
                </svg>

                {/* Center: Goal - solid bg + colored overlay */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full w-28 h-28 flex flex-col items-center justify-center text-white text-center p-2">
                  <div className="absolute inset-0 bg-slate-900 rounded-full"></div>
                  <div className="absolute inset-0 bg-blue-500/40 border border-blue-400/60 rounded-full"></div>
                  <p className="relative text-xs text-blue-300">{t("Goal", "목표")}</p>
                  <p className="relative text-sm font-bold leading-tight">{t("Legal Safety", "법적 안전")}</p>
                </div>

                {/* Perception - Top - centered on line */}
                <div className="absolute top-[50px] left-1/2 -translate-x-1/2 -translate-y-1/2 px-4 py-2 rounded-2xl text-center">
                  <div className="absolute inset-0 bg-slate-900 rounded-2xl"></div>
                  <div className="absolute inset-0 bg-emerald-500/30 border border-emerald-500/50 rounded-2xl"></div>
                  <p className="relative font-bold text-emerald-400 text-sm">{t("Perception", "인식")}</p>
                  <p className="relative text-[10px] text-emerald-300/80">MUVERA, HyDE</p>
                </div>

                {/* Reasoning - Right - centered on line */}
                <div className="absolute top-1/2 right-[50px] -translate-y-1/2 translate-x-1/2 px-4 py-2 rounded-2xl text-center">
                  <div className="absolute inset-0 bg-slate-900 rounded-2xl"></div>
                  <div className="absolute inset-0 bg-amber-500/30 border border-amber-500/50 rounded-2xl"></div>
                  <p className="relative font-bold text-amber-400 text-sm">{t("Reasoning", "추론")}</p>
                  <p className="relative text-[10px] text-amber-300/80">Neuro-Symbolic</p>
                </div>

                {/* Action - Bottom - centered on line */}
                <div className="absolute bottom-[50px] left-1/2 -translate-x-1/2 translate-y-1/2 px-4 py-2 rounded-2xl text-center">
                  <div className="absolute inset-0 bg-slate-900 rounded-2xl"></div>
                  <div className="absolute inset-0 bg-rose-500/30 border border-rose-500/50 rounded-2xl"></div>
                  <p className="relative font-bold text-rose-400 text-sm">{t("Action", "행동")}</p>
                  <p className="relative text-[10px] text-rose-300/80">Constitutional AI</p>
                </div>

                {/* Environment - Left - centered on line */}
                <div className="absolute top-1/2 left-[50px] -translate-y-1/2 -translate-x-1/2 px-4 py-2 rounded-2xl text-center">
                  <div className="absolute inset-0 bg-slate-900 rounded-2xl"></div>
                  <div className="absolute inset-0 bg-slate-500/30 border border-slate-400/50 rounded-2xl"></div>
                  <p className="relative font-bold text-slate-300 text-sm">{t("Environment", "환경")}</p>
                  <p className="relative text-[10px] text-slate-400">{t("Contract", "계약서")}</p>
                </div>
              </div>
            </div>

            {/* Right: Pipeline & Technologies */}
            <div className="w-3/5 p-10 flex flex-col justify-center">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Our Approach", "우리의 접근법")}
              </p>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">
                {t("DocScanner AI Solution", "DocScanner AI 솔루션")}
              </h2>
              <p className="text-lg text-slate-500 mb-6">{t("AI-Powered Legal Document Analysis Pipeline", "AI 기반 법률 문서 분석 파이프라인")}</p>

              {/* Simplified Pipeline */}
              <div className="flex items-center gap-2 mb-8">
                {[
                  { icon: "DOC", title: t("Upload", "업로드"), color: "bg-slate-100" },
                  { icon: "OCR", title: t("Parse", "파싱"), color: "bg-blue-100" },
                  { icon: "AI", title: t("Analyze", "분석"), color: "bg-emerald-100" },
                  { icon: "LAW", title: t("Verify", "검증"), color: "bg-amber-100" },
                  { icon: "OUT", title: t("Report", "보고서"), color: "bg-green-100" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center">
                    <div className={`${step.color} rounded-xl p-3 text-center w-24`}>
                      <span className="text-sm font-bold text-slate-600">{step.icon}</span>
                      <p className="text-xs font-medium text-slate-900 mt-1">{step.title}</p>
                    </div>
                    {i < 4 && <ChevronRight className="w-4 h-4 text-slate-300 mx-1" />}
                  </div>
                ))}
              </div>

              {/* Key Technologies */}
              <div className="grid grid-cols-2 gap-4">
                {[
                  { title: "MUVERA FDE", desc: t("Sentence-level embedding", "문장 단위 임베딩"), icon: "01" },
                  { title: "HyDE + CRAG", desc: t("Context-aware retrieval", "컨텍스트 인식 검색"), icon: "02" },
                  { title: t("Neuro-Symbolic", "신경-기호"), desc: t("LLM + Python calculation", "LLM + Python 계산"), icon: "03" },
                  { title: "Constitutional AI", desc: t("Self-correcting ethics", "자기 수정 윤리"), icon: "04" },
                ].map((tech, i) => (
                  <div key={i} className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
                    <span className="text-sm font-bold text-blue-600">{tech.icon}</span>
                    <h4 className="font-bold text-slate-900 mt-2 text-sm">{tech.title}</h4>
                    <p className="text-xs text-slate-500">{tech.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 3: // Hybrid DB Architecture
        return (
          <div className="flex h-full">
            {/* Left: Vertical Pipeline Diagram */}
            <div className="w-3/5 bg-slate-900 p-8 flex items-center justify-center">
              <div className="w-full max-w-lg">
                <div className="text-white space-y-4">
                  {/* Input: Extracted Clause */}
                  <div className="bg-slate-800 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-3 py-1.5 bg-amber-500/30 rounded text-base text-amber-300">{t("Input", "입력")}</span>
                      <p className="text-slate-400 text-base">{t("Extracted Clause", "추출된 조항")}</p>
                    </div>
                    <div className="bg-slate-700/50 rounded p-4 text-base font-mono text-slate-300">
                      <span className="text-amber-400">{t("Article 5", "제5조")}</span>: {t("Hourly wage 9,500 KRW", "시급 9,500원")}<br/>
                      {t("8 hours/day, 5 days/week", "1일 8시간, 주 5일")}
                    </div>
                  </div>

                  {/* Arrow down */}
                  <div className="flex justify-center">
                    <ChevronRight className="w-8 h-8 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 1: Vector DB Search */}
                  <div className="bg-slate-800 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-3 py-1.5 bg-blue-500/30 rounded text-base text-blue-300">Step 1</span>
                      <p className="text-slate-400 text-base">{t("Vector DB Search", "Vector DB 검색")}</p>
                      <span className="ml-auto text-sm text-blue-400">Elasticsearch</span>
                    </div>
                    <div className="bg-slate-700/30 rounded p-4 space-y-2">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm text-slate-500">{t("Query:", "쿼리:")}</span>
                        <span className="text-sm text-slate-300 font-mono">{t("\"hourly wage\"", "\"시급 임금\"")}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-6 h-6 bg-green-500/40 rounded text-sm flex items-center justify-center">1</span>
                        <span className="text-sm text-slate-300">{t("Precedent: Min wage violation", "판례: 최저임금 미달")}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-6 h-6 bg-blue-500/40 rounded text-sm flex items-center justify-center">2</span>
                        <span className="text-sm text-slate-300">{t("Labor Act Art.6", "근로기준법 제6조")}</span>
                      </div>
                      <div className="text-sm text-slate-500 pl-8">+3 {t("more", "건")}...</div>
                    </div>
                  </div>

                  {/* Arrow down */}
                  <div className="flex justify-center">
                    <ChevronRight className="w-8 h-8 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 2: Graph DB Expansion */}
                  <div className="bg-slate-800 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-3 py-1.5 bg-emerald-500/30 rounded text-base text-emerald-300">Step 2</span>
                      <p className="text-slate-400 text-base">{t("Graph DB Expansion", "Graph DB 확장")}</p>
                      <span className="ml-auto text-sm text-emerald-400">Neo4j</span>
                    </div>
                    <div className="bg-slate-700/30 rounded p-4">
                      {/* Mini Graph - SVG Tree */}
                      <div className="relative h-24">
                        {/* Top node - Precedent */}
                        <div className="absolute top-0 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-green-500/40 rounded text-sm text-green-300 border border-green-500/30 z-10">
                          {t("Precedent", "판례")}
                        </div>
                        {/* Bottom left node - Law */}
                        <div className="absolute bottom-0 left-4 px-4 py-1.5 bg-blue-500/40 rounded text-sm text-blue-300 border border-blue-500/30 z-10">
                          {t("Law", "법령")}
                        </div>
                        {/* Bottom right node - RiskPattern */}
                        <div className="absolute bottom-0 right-4 px-4 py-1.5 bg-amber-500/40 rounded text-sm text-amber-300 border border-amber-500/30 z-10">
                          {t("Risk", "위험")}
                        </div>
                        {/* SVG Lines connecting nodes */}
                        <svg className="absolute inset-0 w-full h-full overflow-visible">
                          <line x1="50%" y1="32" x2="20%" y2="60" stroke="rgba(52,211,153,0.7)" strokeWidth="2" />
                          <line x1="50%" y1="32" x2="80%" y2="60" stroke="rgba(52,211,153,0.7)" strokeWidth="2" />
                          <text x="32%" y="50" fill="rgba(148,163,184,0.9)" fontSize="11" textAnchor="middle">CITES</text>
                          <text x="68%" y="50" fill="rgba(148,163,184,0.9)" fontSize="11" textAnchor="middle">CITES</text>
                        </svg>
                      </div>
                      <div className="flex items-center justify-center gap-4 mt-3 text-sm text-slate-400">
                        <span className="px-3 py-1.5 bg-slate-600/50 rounded">+3 {t("precedents", "판례")}</span>
                        <span className="px-3 py-1.5 bg-slate-600/50 rounded">+2 {t("interpretations", "해석례")}</span>
                      </div>
                    </div>
                  </div>

                  {/* Arrow down */}
                  <div className="flex justify-center">
                    <ChevronRight className="w-8 h-8 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 3: LLM Analysis */}
                  <div className="bg-gradient-to-r from-blue-600 to-emerald-600 rounded-xl p-5">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="px-3 py-1.5 bg-white/20 rounded text-base text-white">Step 3</span>
                      <p className="text-white/90 text-lg font-medium">{t("LLM Analysis", "LLM 분석")}</p>
                    </div>
                    <div className="flex items-center justify-center gap-6 mb-3">
                      <div className="text-center">
                        <p className="text-white/70 text-sm">{t("Vector", "Vector")}</p>
                        <p className="text-white font-bold text-xl">5</p>
                      </div>
                      <span className="text-white/50 text-2xl">+</span>
                      <div className="text-center">
                        <p className="text-white/70 text-sm">{t("Graph", "Graph")}</p>
                        <p className="text-white font-bold text-xl">6</p>
                      </div>
                      <span className="text-white/50 text-2xl">=</span>
                      <div className="text-center">
                        <p className="text-white/70 text-sm">{t("Total", "합계")}</p>
                        <p className="text-white font-bold text-2xl">11 {t("docs", "문서")}</p>
                      </div>
                    </div>
                    <div className="bg-white/10 rounded-lg px-4 py-3">
                      <p className="text-sm text-white/60 mb-1">{t("Result:", "결과:")}</p>
                      <p className="text-base text-white">{t("\"9,500 KRW < 2025 min wage 10,030 KRW\"", "\"시급 9,500원 < 최저임금 10,030원\"")}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: DB Architecture Info */}
            <div className="w-2/5 p-8 flex flex-col justify-center">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Perception", "지각")}
              </p>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">
                {t("Hybrid DB Architecture", "하이브리드 DB 아키텍처")}
              </h2>
              <p className="text-lg text-slate-500 mb-8">
                {t("Vector DB + Graph DB for Complete Context", "완전한 컨텍스트를 위한 Vector + Graph DB")}
              </p>

              <div className="space-y-6">
                {/* Vector DB */}
                <div className="bg-blue-50 rounded-xl p-5 border border-blue-200">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                      <DatabaseIcon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-lg">Vector DB</h3>
                      <p className="text-xs text-slate-500">Elasticsearch</p>
                    </div>
                  </div>
                  <div className="flex gap-6 mb-3">
                    <div>
                      <p className="text-2xl font-bold text-blue-600">15,223</p>
                      <p className="text-xs text-slate-500">{t("Chunks", "청크")}</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-blue-600">1,024</p>
                      <p className="text-xs text-slate-500">{t("Dimensions", "차원")}</p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">MUVERA FDE</span>
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">Hybrid Search</span>
                  </div>
                </div>

                {/* Graph DB */}
                <div className="bg-emerald-50 rounded-xl p-5 border border-emerald-200">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center">
                      <GitBranchIcon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-lg">Graph DB</h3>
                      <p className="text-xs text-slate-500">Neo4j</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <p className="text-xs text-slate-500 mb-1">{t("Nodes:", "노드:")}</p>
                      <div className="flex flex-wrap gap-2">
                        <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded">Precedent</span>
                        <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded">Law</span>
                        <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded">RiskPattern</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 mb-1">{t("Relations:", "관계:")}</p>
                      <div className="flex flex-wrap gap-2">
                        <span className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded font-mono">CITES</span>
                        <span className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded font-mono">HAS_CASE</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Why Hybrid */}
                <div className="bg-slate-100 rounded-xl p-5">
                  <p className="font-semibold text-slate-800 mb-3 text-base">{t("Why Hybrid?", "왜 하이브리드?")}</p>
                  <ul className="space-y-2 text-sm text-slate-600">
                    <li><strong>Vector</strong>: {t("Semantic similarity search", "의미적 유사도 검색")}</li>
                    <li><strong>Graph</strong>: {t("Relation-based expansion", "관계 기반 확장")}</li>
                    <li className="text-blue-600 font-medium">{t("= Complete legal context", "= 완전한 법률 컨텍스트")}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );

      case 4: // MUVERA Embedding
        return (
          <div className="flex h-full">
            {/* Left: Diagram */}
            <div className="w-1/2 bg-slate-900 p-8 flex items-center justify-center">
              <div className="w-full max-w-lg">
                <div className="text-white">
                  {/* Step 1: Document Chunking */}
                  <div className="bg-slate-800 rounded-xl p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 bg-amber-500/30 rounded text-xs text-amber-300">Step 1</span>
                      <p className="text-slate-400 text-xs">{t("Article-based Chunking", "조항 단위 청킹")}</p>
                    </div>
                    <div className="bg-slate-700/50 rounded p-2 text-xs font-mono text-slate-300">
                      <span className="text-amber-400">{t("Article 5", "제5조")}</span> {t("(Working Hours)", "(근로시간)")}<br/>
                      {t("\"8 hours/day, 40 hours/week...\"", "\"1일 8시간, 1주 40시간...\"")}
                    </div>
                  </div>

                  <div className="flex justify-center my-2">
                    <ChevronRight className="w-5 h-5 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 2: Sentence Split + KURE-v1 Embedding */}
                  <div className="bg-slate-800 rounded-xl p-3 mb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 bg-blue-500/30 rounded text-xs text-blue-300">Step 2</span>
                      <p className="text-slate-400 text-xs">{t("Sentence Split + KURE-v1 Embedding", "문장 분리 + KURE-v1 임베딩")}</p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-700/50 rounded p-1.5 text-xs text-slate-300">
                          {t("\"8 hours/day, 40 hours/week\"", "\"1일 8시간, 1주 40시간\"")}
                        </div>
                        <ChevronRight className="w-4 h-4 text-blue-400" />
                        <div className="w-20 h-6 bg-gradient-to-r from-blue-600 to-blue-500 rounded flex items-center justify-center text-xs font-mono">
                          v1 [1024]
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-700/50 rounded p-1.5 text-xs text-slate-300">
                          {t("\"Overtime limited to 12 hours\"", "\"연장근로 12시간 제한\"")}
                        </div>
                        <ChevronRight className="w-4 h-4 text-emerald-400" />
                        <div className="w-20 h-6 bg-gradient-to-r from-emerald-600 to-emerald-500 rounded flex items-center justify-center text-xs font-mono">
                          v2 [1024]
                        </div>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-2 text-right">KURE-v1 (Korean Legal Model)</p>
                  </div>

                  <div className="flex justify-center my-2">
                    <ChevronRight className="w-5 h-5 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 3: SimHash Partitioning - Detailed */}
                  <div className="bg-slate-800 rounded-xl p-3 mb-3">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="px-2 py-0.5 bg-purple-500/30 rounded text-xs text-purple-300">Step 3</span>
                      <p className="text-slate-400 text-xs">{t("SimHash: Locality-Sensitive Hashing", "SimHash: 지역 민감 해싱")}</p>
                    </div>

                    {/* SimHash Process Visualization */}
                    <div className="bg-slate-700/30 rounded-lg p-2.5 mb-2.5">
                      <div className="flex items-center gap-2 text-[10px] text-slate-400 mb-2">
                        <span className="text-purple-300">{t("Random Hyperplane Projection", "랜덤 초평면 투영")}</span>
                      </div>

                      {/* Vector to Binary Hash */}
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-14 h-5 bg-blue-500/40 rounded text-[8px] flex items-center justify-center font-mono">
                          v1[1024]
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-slate-500 text-[10px]">x</span>
                          <div className="w-8 h-5 bg-purple-500/30 rounded text-[8px] flex items-center justify-center">R</div>
                        </div>
                        <ChevronRight className="w-3 h-3 text-purple-400" />
                        <div className="flex gap-0.5">
                          {['1','0','1','1'].map((bit, i) => (
                            <div key={i} className={`w-4 h-5 rounded text-[9px] flex items-center justify-center font-mono ${bit === '1' ? 'bg-blue-500/60 text-blue-200' : 'bg-slate-600/60 text-slate-400'}`}>
                              {bit}
                            </div>
                          ))}
                        </div>
                        <span className="text-[9px] text-slate-500">=</span>
                        <div className="px-1.5 py-0.5 bg-blue-500/30 rounded text-[9px] font-mono text-blue-300">
                          11 (0xB)
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <div className="w-14 h-5 bg-emerald-500/40 rounded text-[8px] flex items-center justify-center font-mono">
                          v2[1024]
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-slate-500 text-[10px]">x</span>
                          <div className="w-8 h-5 bg-purple-500/30 rounded text-[8px] flex items-center justify-center">R</div>
                        </div>
                        <ChevronRight className="w-3 h-3 text-purple-400" />
                        <div className="flex gap-0.5">
                          {['0','1','1','1'].map((bit, i) => (
                            <div key={i} className={`w-4 h-5 rounded text-[9px] flex items-center justify-center font-mono ${bit === '1' ? 'bg-emerald-500/60 text-emerald-200' : 'bg-slate-600/60 text-slate-400'}`}>
                              {bit}
                            </div>
                          ))}
                        </div>
                        <span className="text-[9px] text-slate-500">=</span>
                        <div className="px-1.5 py-0.5 bg-emerald-500/30 rounded text-[9px] font-mono text-emerald-300">
                          7 (0x7)
                        </div>
                      </div>

                      <p className="text-[9px] text-slate-500 mt-2 italic">
                        {t("sign(v . R) > 0 ? 1 : 0  (R: random projection matrix)", "sign(v . R) > 0 ? 1 : 0  (R: 랜덤 투영 행렬)")}
                      </p>
                    </div>

                    {/* 16 Partitions - Compact */}
                    <div className="flex items-center gap-1.5 mt-2">
                      <span className="text-[9px] text-slate-500 shrink-0">{t("Buckets:", "버킷:")}</span>
                      <div className="flex items-center gap-0.5">
                        <span className="w-5 h-5 bg-slate-700/50 rounded text-[8px] flex items-center justify-center text-slate-500">0</span>
                        <span className="text-slate-600 text-[10px]">..</span>
                        <span className="w-7 h-5 bg-emerald-500/60 rounded text-[8px] flex items-center justify-center font-mono ring-1 ring-emerald-400">v2</span>
                        <span className="text-slate-600 text-[10px]">..</span>
                        <span className="w-7 h-5 bg-blue-500/60 rounded text-[8px] flex items-center justify-center font-mono ring-1 ring-blue-400">v1</span>
                        <span className="text-slate-600 text-[10px]">..</span>
                        <span className="w-5 h-5 bg-slate-700/50 rounded text-[8px] flex items-center justify-center text-slate-500">15</span>
                      </div>
                      <span className="text-[9px] text-slate-500">(2^4=16)</span>
                    </div>
                    <p className="text-[9px] text-slate-500 mt-1.5">{t("Similar vectors -> Same bucket (locality-sensitive)", "유사 벡터 -> 같은 버킷 (지역 민감)")}</p>
                  </div>

                  <div className="flex justify-center my-2">
                    <ChevronRight className="w-5 h-5 text-slate-500 rotate-90" />
                  </div>

                  {/* Step 4: FDE Compression */}
                  <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 bg-white/20 rounded text-xs text-white">Step 4</span>
                      <p className="text-white/80 text-xs">{t("FDE Compression (Sum Aggregation)", "FDE 압축 (Sum Aggregation)")}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-5 bg-white/20 rounded overflow-hidden flex">
                        {[...Array(16)].map((_, i) => (
                          <div
                            key={i}
                            className="flex-1 border-r border-white/10 last:border-0"
                            style={{
                              background: `linear-gradient(to bottom, rgba(59,130,246,${0.2 + (i % 4) * 0.15}), rgba(16,185,129,${0.2 + ((i + 2) % 4) * 0.15}))`,
                            }}
                          />
                        ))}
                      </div>
                      <span className="text-xs font-mono font-bold">1024-dim</span>
                    </div>
                    <p className="text-xs text-white/60 mt-1.5">
                      {t("Multi-vector -> Single FDE vector (preserves semantics)", "Multi-vector -> Single FDE 벡터 (의미 보존)")}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Explanation */}
            <div className="w-1/2 p-12 flex flex-col justify-center">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Perception I", "인지 I")}
              </p>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">
                {t("Perception I: MUVERA Embedding", "인지 I: MUVERA 임베딩")}
              </h2>
              <p className="text-lg text-slate-500 mb-8">{t("Multi-Vector Retrieval via Fixed Dimensional Encodings (Google Research, NeurIPS 2024)", "고정 차원 인코딩을 통한 다중 벡터 검색 (Google Research, NeurIPS 2024)")}</p>

              <div className="space-y-6">
                <div>
                  <h3 className="font-bold text-slate-900 mb-3 flex items-center gap-2">
                    <span className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center text-sm">
                      !
                    </span>
                    {t("The Challenge", "문제점")}
                  </h3>
                  <ul className="space-y-2 text-slate-600 ml-8">
                    <li>
                      - {t("Single-vector embedding", "단일 벡터 임베딩은")} <strong>{t("loses details", "세부 정보를 손실")}</strong>{" "}
                      {t("in long documents", "긴 문서에서")}
                    </li>
                    <li>
                      - {t("ColBERT multi-vector is accurate but", "ColBERT 다중 벡터는 정확하지만")}{" "}
                      <strong>{t("computationally expensive", "계산 비용이 높음")}</strong>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="font-bold text-slate-900 mb-3 flex items-center gap-2">
                    <span className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold text-green-600">
                      OK
                    </span>
                    {t("MUVERA Solution", "MUVERA 솔루션")}
                  </h3>
                  <ul className="space-y-2 text-slate-600 ml-8">
                    <li>
                      1. {t("Chunk by articles", "조항 단위로 청킹")} <strong>{t("(Article N)", "(제N조)")}</strong>
                    </li>
                    <li>
                      2. {t("Split into sentences, embed with", "문장 분리 후")} <strong>KURE-v1</strong> {t("embedding", "임베딩")}
                    </li>
                    <li>
                      3. <strong>SimHash</strong> {t("assigns vectors to", "로 벡터를")} <strong>{t("16 partitions", "16개 파티션에 분배")}</strong>
                    </li>
                    <li>
                      4. <strong>FDE</strong> {t("compresses to single", "로")} <strong>{t("1024-dim vector", "1024차원 벡터")}</strong>{t(" via sum aggregation", "로 압축")}
                    </li>
                  </ul>
                </div>

                <div className="bg-green-50 rounded-xl p-4 border border-green-200">
                  <p className="font-semibold text-green-800 mb-2">
                    {t("Performance (per paper)", "성능 (논문 기준)")}
                  </p>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-green-700">
                        5-20x
                      </p>
                      <p className="text-xs text-green-600">{t("Less candidates", "후보 감소")}</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-700">90%</p>
                      <p className="text-xs text-green-600">{t("Latency reduction", "지연 시간 감소")}</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-700">+10%</p>
                      <p className="text-xs text-green-600">{t("Recall improvement", "재현율 향상")}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 5: // RAPTOR Indexing
        return (
          <div className="flex h-full">
            {/* Left: Tree Diagram */}
            <div className="w-1/2 bg-slate-900 p-8 flex items-center justify-center">
              <div className="w-full max-w-lg">
                <div className="text-white">
                  {/* Tree Visualization */}
                  <div className="relative">
                    {/* Level 2: Root Summary */}
                    <div className="flex justify-center mb-4">
                      <div className="bg-gradient-to-r from-slate-600 to-slate-700 rounded-xl px-5 py-3 text-center shadow-lg border border-slate-500">
                        <p className="text-[10px] text-slate-300 mb-1">Level 2 - Root</p>
                        <p className="text-sm font-medium">{t("Contract Overview", "계약서 전체 요약")}</p>
                        <p className="text-[10px] text-slate-300 mt-1">{t("Comprehensive Summary", "종합 분석 및 위험 평가")}</p>
                      </div>
                    </div>

                    {/* Connection Lines Level 2 -> 1 */}
                    <div className="flex justify-center mb-2">
                      <svg className="w-80 h-8" viewBox="0 0 320 32">
                        <line x1="160" y1="0" x2="60" y2="32" stroke="rgba(100,116,139,0.6)" strokeWidth="2" />
                        <line x1="160" y1="0" x2="160" y2="32" stroke="rgba(100,116,139,0.6)" strokeWidth="2" />
                        <line x1="160" y1="0" x2="260" y2="32" stroke="rgba(100,116,139,0.6)" strokeWidth="2" />
                      </svg>
                    </div>

                    {/* Level 1: Category Summaries */}
                    <div className="flex justify-center gap-3 mb-4">
                      <div className="bg-rose-500/30 border border-rose-500/50 rounded-lg px-3 py-2 text-center flex-1 max-w-[100px]">
                        <p className="text-[9px] text-rose-300">Level 1</p>
                        <p className="text-xs font-medium text-rose-200">{t("Wage", "임금")}</p>
                      </div>
                      <div className="bg-emerald-500/30 border border-emerald-500/50 rounded-lg px-3 py-2 text-center flex-1 max-w-[100px]">
                        <p className="text-[9px] text-emerald-300">Level 1</p>
                        <p className="text-xs font-medium text-emerald-200">{t("Work Hours", "근로시간")}</p>
                      </div>
                      <div className="bg-blue-500/30 border border-blue-500/50 rounded-lg px-3 py-2 text-center flex-1 max-w-[100px]">
                        <p className="text-[9px] text-blue-300">Level 1</p>
                        <p className="text-xs font-medium text-blue-200">{t("Leave", "휴가")}</p>
                      </div>
                    </div>

                    {/* Connection Lines Level 1 -> 0 */}
                    <div className="flex justify-center mb-2">
                      <svg className="w-80 h-6" viewBox="0 0 320 24">
                        {/* From Wage */}
                        <line x1="53" y1="0" x2="30" y2="24" stroke="rgba(244,63,94,0.4)" strokeWidth="1.5" />
                        <line x1="53" y1="0" x2="70" y2="24" stroke="rgba(244,63,94,0.4)" strokeWidth="1.5" />
                        {/* From Work Hours */}
                        <line x1="160" y1="0" x2="130" y2="24" stroke="rgba(52,211,153,0.4)" strokeWidth="1.5" />
                        <line x1="160" y1="0" x2="160" y2="24" stroke="rgba(52,211,153,0.4)" strokeWidth="1.5" />
                        <line x1="160" y1="0" x2="190" y2="24" stroke="rgba(52,211,153,0.4)" strokeWidth="1.5" />
                        {/* From Leave */}
                        <line x1="267" y1="0" x2="250" y2="24" stroke="rgba(59,130,246,0.4)" strokeWidth="1.5" />
                        <line x1="267" y1="0" x2="290" y2="24" stroke="rgba(59,130,246,0.4)" strokeWidth="1.5" />
                      </svg>
                    </div>

                    {/* Level 0: Original Chunks */}
                    <div className="flex justify-center gap-1">
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.5", "제5조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.6", "제6조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.7", "제7조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.8", "제8조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.9", "제9조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.10", "제10조")}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded px-2 py-1.5 text-center">
                        <p className="text-[8px] text-slate-400">L0</p>
                        <p className="text-[10px]">{t("Art.11", "제11조")}</p>
                      </div>
                    </div>

                    {/* Level Labels */}
                    <div className="absolute -left-4 top-0 bottom-0 flex flex-col justify-between text-[9px] text-slate-500">
                      <span className="py-3">{t("Summary", "요약")}</span>
                      <span>{t("Chunks", "청크")}</span>
                    </div>
                  </div>

                  {/* Process Flow */}
                  <div className="mt-8 bg-slate-800 rounded-xl p-4">
                    <p className="text-xs text-slate-400 mb-3">{t("Build Process", "구축 과정")}</p>
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-center flex-1">
                        <div className="w-10 h-10 mx-auto bg-amber-500/30 rounded-lg flex items-center justify-center mb-1">
                          <span className="text-amber-300 text-sm">1</span>
                        </div>
                        <p className="text-[10px] text-slate-300">{t("Embed", "임베딩")}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-slate-600" />
                      <div className="text-center flex-1">
                        <div className="w-10 h-10 mx-auto bg-blue-500/30 rounded-lg flex items-center justify-center mb-1">
                          <span className="text-blue-300 text-sm">2</span>
                        </div>
                        <p className="text-[10px] text-slate-300">{t("Cluster", "클러스터링")}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-slate-600" />
                      <div className="text-center flex-1">
                        <div className="w-10 h-10 mx-auto bg-emerald-500/30 rounded-lg flex items-center justify-center mb-1">
                          <span className="text-emerald-300 text-sm">3</span>
                        </div>
                        <p className="text-[10px] text-slate-300">{t("Summarize", "요약")}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-slate-600" />
                      <div className="text-center flex-1">
                        <div className="w-10 h-10 mx-auto bg-slate-500/30 rounded-lg flex items-center justify-center mb-1">
                          <span className="text-slate-300 text-sm">4</span>
                        </div>
                        <p className="text-[10px] text-slate-300">{t("Repeat", "반복")}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Info */}
            <div className="w-1/2 p-12 flex flex-col justify-center">
              <p className="text-teal-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Hierarchical Indexing", "계층적 인덱싱")}
              </p>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">
                {t("RAPTOR: Tree-Organized Retrieval", "RAPTOR: 트리 기반 검색")}
              </h2>
              <p className="text-lg text-slate-500 mb-8">
                {t("Multi-level abstraction for comprehensive understanding", "포괄적 이해를 위한 다중 레벨 추상화")}
              </p>

              <div className="space-y-5">
                {/* Clustering Methods */}
                <div className="bg-teal-50 rounded-xl p-4 border border-teal-200">
                  <h3 className="font-bold text-slate-900 mb-3">{t("Dynamic Clustering", "동적 클러스터링")}</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-teal-500 rounded-full"></span>
                      <span className="text-sm text-slate-700">Agglomerative</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-teal-500 rounded-full"></span>
                      <span className="text-sm text-slate-700">DBSCAN</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-teal-500 rounded-full"></span>
                      <span className="text-sm text-slate-700">K-Means</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-teal-500 rounded-full"></span>
                      <span className="text-sm text-slate-700">Semantic</span>
                    </div>
                  </div>
                </div>

                {/* Legal Categories */}
                <div className="bg-slate-100 rounded-xl p-4">
                  <h3 className="font-bold text-slate-900 mb-3">{t("Legal Category Classification", "법률 카테고리 분류")}</h3>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 bg-rose-100 text-rose-700 text-xs rounded">{t("Wage", "임금")}</span>
                    <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded">{t("Work Hours", "근로시간")}</span>
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">{t("Leave", "휴일휴가")}</span>
                    <span className="px-2 py-1 bg-amber-100 text-amber-700 text-xs rounded">{t("Termination", "해고퇴직")}</span>
                    <span className="px-2 py-1 bg-slate-200 text-slate-700 text-xs rounded">{t("Contract Period", "계약기간")}</span>
                    <span className="px-2 py-1 bg-cyan-100 text-cyan-700 text-xs rounded">{t("Benefits", "복리후생")}</span>
                  </div>
                </div>

                {/* Search Strategies */}
                <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                  <h3 className="font-bold text-slate-900 mb-3">{t("Adaptive Search", "적응형 검색")}</h3>
                  <div className="space-y-2 text-sm text-slate-600">
                    <div className="flex items-center gap-2">
                      <ChevronRight className="w-4 h-4 text-blue-500" />
                      <span><strong>Top-Down</strong>: {t("Root to leaf traversal", "루트에서 리프로 탐색")}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <ChevronRight className="w-4 h-4 text-blue-500" />
                      <span><strong>Bottom-Up</strong>: {t("Leaf aggregation", "리프에서 상향 집계")}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <ChevronRight className="w-4 h-4 text-blue-500" />
                      <span><strong>Adaptive</strong>: {t("Query-based selection", "질문 유형에 따라 선택")}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 6: // HyDE & CRAG & Knowledge Graph
        return (
          <div className="flex h-full">
            {/* Left: Three Diagrams */}
            <div className="w-1/2 bg-slate-900 p-6 flex items-center justify-center">
              <div className="w-full max-w-lg space-y-4">
                {/* 1. HyDE Section */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-1 bg-blue-500/30 rounded text-xs text-blue-300 font-bold">1</span>
                    <p className="text-blue-400 font-semibold text-sm">HyDE</p>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-700 rounded-lg p-2">
                        <p className="text-[10px] text-slate-400">{t("Query", "질의")}</p>
                        <p className="text-xs text-white">{t("\"Overtime pay?\"", "\"연장근로 수당?\"")}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-blue-400 flex-shrink-0" />
                      <div className="flex-1 bg-blue-900/50 rounded-lg p-2 border border-blue-500/30">
                        <p className="text-[10px] text-blue-400">LLM</p>
                        <p className="text-xs text-slate-300">{t("Generate hypo", "가상 답변 생성")}</p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-blue-400 flex-shrink-0" />
                      <div className="flex-1 bg-green-900/50 rounded-lg p-2 border border-green-500/30">
                        <p className="text-[10px] text-green-400">{t("Hypo Doc", "가상 문서")}</p>
                        <p className="text-xs text-slate-300">{t("\"Art.56 50%...\"", "\"제56조 50%...\"")}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. CRAG Section */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-1 bg-amber-500/30 rounded text-xs text-amber-300 font-bold">2</span>
                    <p className="text-amber-400 font-semibold text-sm">CRAG</p>
                    <span className="text-slate-500 text-xs ml-auto">{t("Corrective RAG", "교정적 RAG")}</span>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-3">
                    <div className="flex items-center gap-2">
                      {/* Retrieved Docs */}
                      <div className="flex-shrink-0 bg-slate-700 rounded-lg p-2">
                        <p className="text-[10px] text-slate-400">{t("Retrieved", "검색됨")}</p>
                        <div className="flex gap-1 mt-1">
                          <span className="w-4 h-4 bg-slate-600 rounded text-[8px] flex items-center justify-center">1</span>
                          <span className="w-4 h-4 bg-slate-600 rounded text-[8px] flex items-center justify-center">2</span>
                          <span className="w-4 h-4 bg-slate-600 rounded text-[8px] flex items-center justify-center">3</span>
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-amber-400 flex-shrink-0" />
                      {/* Quality Grading */}
                      <div className="flex-1 bg-amber-900/50 rounded-lg p-2 border border-amber-500/30">
                        <p className="text-[10px] text-amber-400 mb-1">{t("8-Level Grading", "8단계 품질 등급")}</p>
                        <div className="flex gap-0.5">
                          <span className="px-1 py-0.5 bg-green-500/50 rounded text-[8px] text-green-200">HIGH</span>
                          <span className="px-1 py-0.5 bg-yellow-500/50 rounded text-[8px] text-yellow-200">MED</span>
                          <span className="px-1 py-0.5 bg-red-500/50 rounded text-[8px] text-red-200">LOW</span>
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-amber-400 flex-shrink-0" />
                      {/* Correction */}
                      <div className="flex-1 bg-blue-900/50 rounded-lg p-2 border border-blue-500/30">
                        <p className="text-[10px] text-blue-400 mb-1">{t("7 Corrections", "7가지 교정")}</p>
                        <div className="flex flex-wrap gap-0.5">
                          <span className="px-1 py-0.5 bg-slate-600 rounded text-[8px]">REFINE</span>
                          <span className="px-1 py-0.5 bg-slate-600 rounded text-[8px]">AUGMENT</span>
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-green-400 flex-shrink-0" />
                      {/* Output */}
                      <div className="flex-shrink-0 bg-green-900/50 rounded-lg p-2 border border-green-500/30">
                        <p className="text-[10px] text-green-400">{t("Quality", "고품질")}</p>
                        <p className="text-[10px] text-green-300">{t("Context", "컨텍스트")}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 3. Knowledge Graph Section */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-1 bg-emerald-500/30 rounded text-xs text-emerald-300 font-bold">3</span>
                    <p className="text-emerald-400 font-semibold text-sm">{t("Knowledge Graph", "지식 그래프")}</p>
                    <span className="text-slate-500 text-xs ml-auto">Neo4j</span>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-3">
                    <div className="flex items-center justify-center">
                      <div className="relative w-full h-32">
                        {/* Connections */}
                        <svg className="absolute inset-0 w-full h-full">
                          <line x1="50%" y1="22" x2="10%" y2="50%" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
                          <line x1="50%" y1="22" x2="94%" y2="50%" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
                          <line x1="10%" y1="50%" x2="50%" y2="106" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
                          <line x1="94%" y1="50%" x2="50%" y2="106" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
                        </svg>
                        {/* Nodes with solid background behind transparent overlay */}
                        <div className="absolute top-0 left-1/2 -translate-x-1/2 z-10">
                          <div className="bg-slate-800 rounded">
                            <div className="bg-amber-500/40 px-2 py-1 rounded text-[10px] text-amber-200">
                              {t("RiskPattern", "위험패턴")}
                            </div>
                          </div>
                        </div>
                        <div className="absolute top-1/2 -translate-y-1/2 left-0 z-10">
                          <div className="bg-slate-800 rounded">
                            <div className="bg-blue-500/40 px-2 py-1 rounded text-[10px] text-blue-200">
                              {t("ClauseType", "조항유형")}
                            </div>
                          </div>
                        </div>
                        <div className="absolute top-1/2 -translate-y-1/2 right-0 z-10">
                          <div className="bg-slate-800 rounded">
                            <div className="bg-green-500/40 px-2 py-1 rounded text-[10px] text-green-200">
                              {t("Precedent", "판례")}
                            </div>
                          </div>
                        </div>
                        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 z-10">
                          <div className="bg-slate-800 rounded">
                            <div className="bg-cyan-500/40 px-2 py-1 rounded text-[10px] text-cyan-200">
                              {t("Law", "법령")}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-500 text-center mt-1">{t("Multi-hop: Clause → Risk → Precedent → Law", "다중 홉: 조항 → 위험 → 판례 → 법령")}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Explanation */}
            <div className="w-1/2 p-8 flex flex-col justify-center">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Perception III", "인지 III")}
              </p>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">
                {t("HyDE & CRAG & Knowledge Graph", "HyDE & CRAG & 지식 그래프")}
              </h2>
              <p className="text-base text-slate-500 mb-6">{t("Advanced Retrieval Enhancement for Legal Context", "법률 컨텍스트를 위한 고급 검색 향상")}</p>

              <div className="space-y-4">
                <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                  <h3 className="font-bold text-blue-800 mb-2 flex items-center gap-2">
                    <span className="w-5 h-5 bg-blue-500 rounded text-white text-xs flex items-center justify-center">1</span>
                    HyDE
                  </h3>
                  <ul className="space-y-1 text-sm text-slate-600">
                    <li>- {t("Generate hypothetical answer before search", "검색 전 가상 답변 생성")}</li>
                    <li>- {t("Match semantic intent, not keywords", "키워드가 아닌 의미적 의도 매칭")}</li>
                    <li>- {t("Strategies: SINGLE / ENSEMBLE / MULTI", "전략: SINGLE / ENSEMBLE / MULTI")}</li>
                  </ul>
                </div>

                <div className="bg-amber-50 rounded-xl p-4 border border-amber-200">
                  <h3 className="font-bold text-amber-800 mb-2 flex items-center gap-2">
                    <span className="w-5 h-5 bg-amber-500 rounded text-white text-xs flex items-center justify-center">2</span>
                    CRAG
                  </h3>
                  <ul className="space-y-1 text-sm text-slate-600">
                    <li>- <strong>{t("8-level grading", "8단계 등급")}</strong>: EXCELLENT → IRRELEVANT</li>
                    <li>- <strong>{t("7 corrections", "7가지 교정")}</strong>: REFINE, AUGMENT, REWRITE...</li>
                    <li>- {t("Only high-quality context to LLM", "LLM에 고품질 컨텍스트만 전달")}</li>
                  </ul>
                </div>

                <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-200">
                  <h3 className="font-bold text-emerald-800 mb-2 flex items-center gap-2">
                    <span className="w-5 h-5 bg-emerald-500 rounded text-white text-xs flex items-center justify-center">3</span>
                    {t("Knowledge Graph", "지식 그래프")}
                  </h3>
                  <ul className="space-y-1 text-sm text-slate-600">
                    <li>- <strong>{t("8 node types", "8가지 노드")}</strong>: Document, Precedent, Law...</li>
                    <li>- <strong>{t("Multi-hop traversal", "다중 홉 탐색")}</strong> {t("for complete context", "완전한 컨텍스트")}</li>
                    <li>- {t("LLM-extracted citations via GPT", "GPT로 LLM 인용 추출")}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );

      case 7: // Neuro-Symbolic AI + Constitutional AI (Combined)
        return (
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="bg-white px-12 py-4 border-b border-slate-200">
              <p className="text-blue-600 font-semibold mb-1 text-sm tracking-wide uppercase">
                {t("Reasoning & Action", "추론 및 행동")}
              </p>
              <h2 className="text-2xl font-bold text-slate-900">
                {t("Neuro-Symbolic AI + Constitutional AI", "신경-기호 AI + Constitutional AI")}
              </h2>
              <p className="text-sm text-slate-500">{t("Accurate Calculation meets Ethical Self-Correction", "정확한 계산과 윤리적 자기 수정의 만남")}</p>
            </div>

            {/* Two Diagrams Side by Side */}
            <div className="flex-1 flex">
              {/* Left: Neuro-Symbolic AI - Original Style */}
              <div className="w-1/2 bg-slate-900 p-6 flex flex-col">
                <div className="flex items-center gap-2 mb-4">
                  <span className="px-3 py-1.5 bg-teal-500/30 rounded text-sm text-teal-300 font-bold">1</span>
                  <span className="text-white font-semibold">{t("Neuro-Symbolic AI", "신경-기호 AI")}</span>
                </div>

                <div className="flex-1 flex flex-col justify-center">
                  <div className="text-white space-y-4">
                    {/* Input */}
                    <div className="bg-slate-800 rounded-xl p-4">
                      <p className="text-slate-400 text-sm mb-2">{t("Contract Text", "계약서 텍스트")}</p>
                      <p className="text-sm">
                        {t(
                          "\"Monthly salary: 3,000,000 KRW, 9 hours/day, 5 days/week\"",
                          "\"월급: 300만원, 하루 9시간, 주 5일\""
                        )}
                      </p>
                    </div>

                    {/* Split into Neuro and Symbolic */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-teal-900/50 rounded-xl p-3 border border-teal-500/30">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-base font-bold text-teal-400">LLM</span>
                          <span className="font-semibold text-teal-300 text-sm">
                            {t("Neuro", "신경")}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mb-1">
                          {t("Entity Extraction:", "엔티티 추출:")}
                        </p>
                        <div className="space-y-0.5 font-mono text-xs">
                          <p className="text-green-400">{t("salary: 3,000,000", "월급: 3,000,000")}</p>
                          <p className="text-green-400">{t("hours_day: 9", "일근무시간: 9")}</p>
                          <p className="text-green-400">{t("days_week: 5", "주근무일: 5")}</p>
                        </div>
                      </div>
                      <div className="bg-blue-900/50 rounded-xl p-3 border border-blue-500/30">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-base font-bold text-blue-400">PY</span>
                          <span className="font-semibold text-blue-300 text-sm">
                            {t("Symbolic", "기호")}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mb-1">
                          {t("Python Calculation:", "Python 계산:")}
                        </p>
                        <div className="space-y-0.5 font-mono text-xs">
                          <p className="text-slate-400">total_hrs = 9 * 22</p>
                          <p className="text-slate-400">hourly = 3M / 198</p>
                          <p className="text-yellow-400">= 15,151 {t("KRW", "원")}</p>
                        </div>
                      </div>
                    </div>

                    {/* Output */}
                    <div className="bg-green-900/50 rounded-xl p-3 border border-green-500/30">
                      <p className="text-green-300 text-sm mb-2">
                        {t("Legal Stress Test Result", "법적 스트레스 테스트 결과")}
                      </p>
                      <div className="flex gap-4">
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-slate-300">{t("Minimum Wage", "최저임금")}</span>
                          <span className="px-2 py-0.5 bg-green-500/30 rounded text-green-300 text-xs">
                            {t("PASS", "통과")}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <span className="text-slate-300">{t("Overtime", "연장근로")}</span>
                          <span className="px-2 py-0.5 bg-red-500/30 rounded text-red-300 text-xs">
                            {t("FAIL", "위반")}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Bottom info */}
                <div className="mt-4 pt-3 border-t border-slate-700">
                  <p className="text-xs text-slate-400 text-center">{t("11 Legal Stress Test Checkpoints", "11개 법적 스트레스 테스트 검증 항목")}</p>
                </div>
              </div>

              {/* Right: Constitutional AI - Original Style */}
              <div className="w-1/2 bg-slate-800 p-6 flex flex-col">
                <div className="flex items-center gap-2 mb-4">
                  <span className="px-3 py-1.5 bg-amber-500/30 rounded text-sm text-amber-300 font-bold">2</span>
                  <span className="text-white font-semibold">{t("Constitutional AI", "Constitutional AI")}</span>
                </div>

                <div className="flex-1 flex flex-col justify-center">
                  <div className="text-white space-y-3">
                    {/* Draft */}
                    <div className="bg-slate-700 rounded-xl p-4">
                      <p className="text-slate-400 text-sm mb-2">
                        {t("Draft Response", "초안 응답")}
                      </p>
                      <p className="text-sm">
                        {t(
                          "\"This penalty clause defines damages for early termination...\"",
                          "\"이 위약금 조항은 조기 해지 시 손해배상을 정의합니다...\""
                        )}
                      </p>
                    </div>

                    <div className="flex justify-center">
                      <ChevronRight className="w-6 h-6 text-amber-400 rotate-90" />
                    </div>

                    {/* Critique */}
                    <div className="bg-amber-900/50 rounded-xl p-4 border border-amber-500/30">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-amber-400 font-bold">!</span>
                        <span className="font-semibold text-amber-300">
                          {t("Critique", "비판")}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300">
                        {t(
                          "Violation of Principle #3: \"Labor Standards Act Article 20 prohibits penalty clauses\" not mentioned",
                          "원칙 #3 위반: \"근로기준법 제20조는 위약금 조항을 금지\" 언급 없음"
                        )}
                      </p>
                    </div>

                    <div className="flex justify-center">
                      <ChevronRight className="w-6 h-6 text-green-400 rotate-90" />
                    </div>

                    {/* Revised */}
                    <div className="bg-green-900/50 rounded-xl p-4 border border-green-500/30">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-green-400 font-bold">OK</span>
                        <span className="font-semibold text-green-300">
                          {t("Revised Response", "수정된 응답")}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300">
                        {t(
                          "\"This clause may violate Article 20 of the Labor Standards Act.\"",
                          "\"이 조항은 근로기준법 제20조를 위반할 수 있습니다.\""
                        )}{" "}
                        <strong className="text-green-300">{t("Recommend deletion.", "삭제 권고.")}</strong>
                      </p>
                    </div>
                  </div>
                </div>

                {/* Bottom: 6 Principles */}
                <div className="mt-4 pt-3 border-t border-slate-600">
                  <p className="text-xs text-slate-400 mb-2 text-center">{t("6 Constitutional Principles", "6가지 헌법적 원칙")}</p>
                  <div className="grid grid-cols-3 gap-1 text-xs">
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Human Dignity", "인간 존엄")}</span>
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Worker Protection", "근로자 보호")}</span>
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Min Standard", "최저 기준")}</span>
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Equality", "평등")}</span>
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Safety", "안전")}</span>
                    <span className="px-2 py-1 bg-amber-900/50 text-amber-300 rounded text-center">{t("Transparency", "투명성")}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 8: // System Architecture
        return (
          <div className="flex flex-col h-full px-16 py-12 bg-white">
            <div className="mb-8">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Implementation", "구현")}
              </p>
              <h2 className="text-4xl font-bold text-slate-900 mb-3">
                {t("System Architecture", "시스템 아키텍처")}
              </h2>
              <p className="text-xl text-slate-500">{t("Full-Stack Implementation Overview", "풀스택 구현 개요")}</p>
            </div>

            <div className="flex-1 flex items-center justify-center">
              <div className="w-full max-w-5xl">
                {/* Architecture Diagram */}
                <div className="bg-slate-50 rounded-2xl p-8 border border-slate-200">
                  {/* Flow Arrow - Forward */}
                  <div className="flex items-center justify-center gap-2 mb-4">
                    <span className="text-xs text-slate-500">{t("Request", "요청")}</span>
                    <div className="flex-1 flex items-center">
                      <div className="flex-1 h-0.5 bg-gradient-to-r from-blue-400 via-green-400 via-teal-400 to-amber-400"></div>
                      <ChevronRight className="w-4 h-4 text-amber-500 -ml-1" />
                    </div>
                  </div>

                  <div className="flex items-stretch gap-3">
                    {/* Frontend */}
                    <div className="flex-1 space-y-2">
                      <div className="bg-blue-100 rounded-xl p-3 text-center">
                        <p className="font-bold text-blue-900">{t("Frontend", "프론트엔드")}</p>
                        <p className="text-xs text-blue-700">Next.js 15 + React 19</p>
                      </div>
                      <div className="space-y-1.5 px-1">
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-blue-800">{t("PDF Viewer", "PDF 뷰어")}</p>
                          <p className="text-slate-500">{t("Clause highlighting", "조항 하이라이팅")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-blue-800">{t("Real-time SSE", "실시간 SSE")}</p>
                          <p className="text-slate-500">{t("Streaming response", "스트리밍 응답")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-blue-800">{t("Analysis Dashboard", "분석 대시보드")}</p>
                          <p className="text-slate-500">{t("Risk visualization", "위험도 시각화")}</p>
                        </div>
                      </div>
                    </div>

                    {/* Arrow 1 */}
                    <div className="flex items-center">
                      <ChevronRight className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* Backend */}
                    <div className="flex-1 space-y-2">
                      <div className="bg-green-100 rounded-xl p-3 text-center">
                        <p className="font-bold text-green-900">{t("Backend", "백엔드")}</p>
                        <p className="text-xs text-green-700">FastAPI + LangGraph</p>
                      </div>
                      <div className="space-y-1.5 px-1">
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-green-800">{t("Chat Agent", "채팅 에이전트")}</p>
                          <p className="text-slate-500">{t("4 tools integration", "4개 도구 통합")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-green-800">{t("Async Pipeline", "비동기 파이프라인")}</p>
                          <p className="text-slate-500">{t("Celery task queue", "Celery 작업 큐")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-green-800">{t("JWT Auth", "JWT 인증")}</p>
                          <p className="text-slate-500">{t("Session management", "세션 관리")}</p>
                        </div>
                      </div>
                    </div>

                    {/* Arrow 2 */}
                    <div className="flex items-center">
                      <ChevronRight className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* AI Pipeline */}
                    <div className="flex-1 space-y-2">
                      <div className="bg-teal-100 rounded-xl p-3 text-center">
                        <p className="font-bold text-teal-900">{t("AI Pipeline", "AI 파이프라인")}</p>
                        <p className="text-xs text-teal-700">{t("12-Stage Analysis", "12단계 분석")}</p>
                      </div>
                      <div className="space-y-1.5 px-1">
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-teal-800">HyDE + CRAG</p>
                          <p className="text-slate-500">{t("Retrieval correction", "검색 품질 보정")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-teal-800">RAPTOR</p>
                          <p className="text-slate-500">{t("Hierarchical summary", "계층적 요약")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-teal-800">Constitutional AI</p>
                          <p className="text-slate-500">{t("6 legal principles", "6개 법적 원칙")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-teal-800">Neuro-Symbolic</p>
                          <p className="text-slate-500">{t("11 stress tests", "11개 스트레스 테스트")}</p>
                        </div>
                      </div>
                    </div>

                    {/* Arrow 3 - Bidirectional */}
                    <div className="flex items-center">
                      <div className="flex flex-col items-center gap-1">
                        <ChevronRight className="w-5 h-5 text-slate-400" />
                        <ChevronRight className="w-5 h-5 text-slate-400 rotate-180" />
                      </div>
                    </div>

                    {/* Data Layer */}
                    <div className="flex-1 space-y-2">
                      <div className="bg-amber-100 rounded-xl p-3 text-center">
                        <p className="font-bold text-amber-900">{t("Data Layer", "데이터 레이어")}</p>
                        <p className="text-xs text-amber-700">{t("Hybrid DB", "하이브리드 DB")}</p>
                      </div>
                      <div className="space-y-1.5 px-1">
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-amber-800">Neo4j</p>
                          <p className="text-slate-500">{t("Law-Precedent graph", "법령-판례 그래프")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-amber-800">Elasticsearch</p>
                          <p className="text-slate-500">{t("MUVERA vector search", "MUVERA 벡터 검색")}</p>
                        </div>
                        <div className="text-xs bg-white rounded p-2 border">
                          <p className="font-semibold text-amber-800">PostgreSQL</p>
                          <p className="text-slate-500">{t("User & session data", "사용자/세션 데이터")}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Flow Arrow - Return (DB → Frontend) */}
                  <div className="flex items-center justify-center gap-2 mt-4">
                    <div className="flex-1 flex items-center">
                      <ChevronRight className="w-4 h-4 text-slate-400 rotate-180 -mr-1" />
                      <div className="flex-1 h-0.5 bg-slate-300"></div>
                    </div>
                    <span className="text-xs text-slate-500">{t("Response (SSE Streaming)", "응답 (SSE 스트리밍)")}</span>
                  </div>
                </div>

                {/* LLM Model Configuration */}
                <div className="mt-6 bg-slate-100 p-4 border border-slate-200" style={{ borderRadius: '12px' }}>
                  <p className="text-sm font-bold text-slate-700 mb-3">{t("Hybrid LLM Configuration", "하이브리드 LLM 구성")}</p>
                  <div className="grid grid-cols-4 gap-3">
                    <div className="bg-white p-3 border" style={{ borderRadius: '8px' }}>
                      <p className="font-bold text-sm text-emerald-700">GPT-4o</p>
                      <p className="text-xs text-slate-500 mt-1">{t("Clause Analysis, Redliner", "조항 분석, 수정안")}</p>
                      <p className="text-xs text-slate-500">{t("Judge, Constitutional", "판단, 헌법적 검토")}</p>
                    </div>
                    <div className="bg-white p-3 border" style={{ borderRadius: '8px' }}>
                      <p className="font-bold text-sm text-blue-700">GPT-4.1</p>
                      <p className="text-xs text-slate-500 mt-1">{t("Legal Reasoning", "법적 추론")}</p>
                      <p className="text-xs text-slate-500">{t("Complex inference", "복잡한 추론")}</p>
                    </div>
                    <div className="bg-white p-3 border" style={{ borderRadius: '8px' }}>
                      <p className="font-bold text-sm text-purple-700">Gemini 2.5 Flash</p>
                      <p className="text-xs text-slate-500 mt-1">{t("RAPTOR summary", "RAPTOR 요약")}</p>
                      <p className="text-xs text-slate-500">{t("Quick Scan, Location", "퀵스캔, 위치 매핑")}</p>
                    </div>
                    <div className="bg-white p-3 border" style={{ borderRadius: '8px' }}>
                      <p className="font-bold text-sm text-amber-700">GPT-4o-mini</p>
                      <p className="text-xs text-slate-500 mt-1">{t("HyDE generation", "HyDE 생성")}</p>
                      <p className="text-xs text-slate-500">{t("CRAG grading", "CRAG 등급 평가")}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 9: // Data & Results
        return (
          <div className="flex h-full">
            {/* Left: Stats Overview */}
            <div className="w-1/2 bg-slate-900 p-12 flex flex-col justify-center">
              <div className="text-white">
                <p className="text-teal-400 font-semibold mb-2 text-sm tracking-wide uppercase">
                  {t("Knowledge Base", "지식 베이스")}
                </p>
                <h2 className="text-3xl font-bold mb-8">
                  {t("Data & Implementation", "데이터 및 구현")}
                </h2>

                {/* Big Numbers */}
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="bg-slate-800 rounded-xl p-5 text-center">
                    <p className="text-4xl font-bold text-teal-400">15,223</p>
                    <p className="text-sm text-slate-400 mt-1">{t("Total Chunks", "총 청크")}</p>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-5 text-center">
                    <p className="text-4xl font-bold text-blue-400">2,931</p>
                    <p className="text-sm text-slate-400 mt-1">{t("Documents", "문서")}</p>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-5 text-center">
                    <p className="text-4xl font-bold text-amber-400">1024</p>
                    <p className="text-sm text-slate-400 mt-1">{t("Embedding Dim", "임베딩 차원")}</p>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-5 text-center">
                    <p className="text-4xl font-bold text-green-400">8</p>
                    <p className="text-sm text-slate-400 mt-1">{t("FDE Partitions", "FDE 파티션")}</p>
                  </div>
                </div>

                {/* MUVERA Config */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <p className="text-slate-400 text-sm mb-3">{t("MUVERA Statistics", "MUVERA 통계")}</p>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-300">{t("Avg Sentences/Chunk", "평균 문장/청크")}</span>
                    <span className="font-mono text-teal-400">3.96</span>
                  </div>
                  <div className="flex items-center justify-between text-sm mt-2">
                    <span className="text-slate-300">{t("Embedding Model", "임베딩 모델")}</span>
                    <span className="font-mono text-teal-400">KURE-v1</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Detailed Breakdown */}
            <div className="w-1/2 p-12 flex flex-col justify-center bg-white">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Data Sources", "데이터 출처")}
              </p>
              <p className="text-lg text-slate-500 mb-6">{t("Built Knowledge Base Statistics", "구축된 지식 베이스 통계")}</p>

              <div className="space-y-4">
                {/* Legal API Data */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-4">{t("Legal API Data", "법률 API 데이터")}</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Precedents", "판례")}</span>
                      <span className="font-mono text-slate-900">969 {t("docs", "문서")} / 10,576 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Ministry Explanations", "고용노동부 해설")}</span>
                      <span className="font-mono text-slate-900">1,827 {t("docs", "문서")} / 3,384 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Legal Interpretations", "법령 해석")}</span>
                      <span className="font-mono text-slate-900">135 {t("docs", "문서")} / 589 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between pt-2 border-t border-slate-200">
                      <span className="font-semibold text-slate-900">{t("Total", "합계")}</span>
                      <span className="font-mono font-bold text-blue-600">2,931 {t("docs", "문서")} / 14,549 {t("chunks", "청크")}</span>
                    </div>
                  </div>
                </div>

                {/* PDF Documents */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-4">{t("PDF Documents (2025)", "PDF 문서 (2025)")}</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Standard Contracts", "표준 계약서")}</span>
                      <span className="font-mono text-slate-900">367 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Hiring Process Manual", "채용절차 매뉴얼")}</span>
                      <span className="font-mono text-slate-900">296 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">{t("Minimum Wage Guide", "최저임금 가이드")}</span>
                      <span className="font-mono text-slate-900">11 {t("chunks", "청크")}</span>
                    </div>
                    <div className="flex justify-between pt-2 border-t border-slate-200">
                      <span className="font-semibold text-slate-900">{t("Total", "합계")}</span>
                      <span className="font-mono font-bold text-green-600">674 {t("chunks", "청크")}</span>
                    </div>
                  </div>
                </div>

                {/* Neo4j Ontology */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-3">{t("Neo4j Graph Schema", "Neo4j 그래프 스키마")}</h3>
                  <div className="flex flex-wrap gap-2">
                    {[
                      t("Document", "문서"),
                      t("Precedent", "판례"),
                      t("Law", "법령"),
                      t("Category", "카테고리"),
                      t("Source", "출처"),
                      t("RiskPattern", "위험패턴")
                    ].map((nodeType, idx) => (
                      <span key={idx} className="px-2 py-1 bg-slate-200 rounded text-xs text-slate-700">
                        {nodeType}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 10: // Live Demo
        return (
          <div className="flex flex-col h-full text-white px-12 py-8 overflow-hidden">
            <div className="mb-4 flex-shrink-0">
              <p className="text-teal-400 font-semibold mb-1 text-sm tracking-wide uppercase">
                {t("System Demonstration", "시스템 시연")}
              </p>
              <h2 className="text-3xl font-bold text-white mb-2">
                {t("Live Demo", "라이브 데모")}
              </h2>
              <p className="text-lg text-slate-400">{t("Interactive Contract Analysis Experience", "인터랙티브 계약서 분석 체험")}</p>
            </div>

            <div className="flex-1 flex items-center justify-center min-h-0">
              {showDemoIframe ? (
                <div className="relative w-full h-full">
                  <button
                    onClick={() => setShowDemoIframe(false)}
                    className="absolute -top-2 -right-2 z-10 w-8 h-8 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center text-white transition-colors"
                  >
                    X
                  </button>
                  <iframe
                    src="/"
                    className="w-full h-full rounded-2xl border-2 border-slate-700"
                    title="DocScanner AI Demo"
                  />
                </div>
              ) : (
                <div className="w-full max-h-full">
                  <div
                    onClick={() => setShowDemoIframe(true)}
                    className="bg-slate-800 rounded-2xl p-4 border border-slate-700 cursor-pointer hover:border-teal-500/50 transition-all group"
                  >
                    <div className="aspect-[21/7] bg-slate-900 rounded-xl flex flex-col items-center justify-center gap-3 group-hover:bg-slate-800 transition-colors">
                      <div className="w-16 h-16 bg-teal-500/20 rounded-full flex items-center justify-center group-hover:bg-teal-500/30 transition-colors">
                        <svg className="w-8 h-8 text-teal-400" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8 5v14l11-7z"/>
                        </svg>
                      </div>
                      <p className="text-slate-400 group-hover:text-slate-300 text-sm">{t("Click to launch interactive demo", "클릭하여 인터랙티브 데모 실행")}</p>
                    </div>
                    <div className="mt-4 grid grid-cols-4 gap-3 text-center">
                      <div className="bg-slate-900 rounded-lg p-2">
                        <p className="text-teal-400 font-bold">1</p>
                        <p className="text-xs text-slate-500">{t("Upload Contract", "계약서 업로드")}</p>
                      </div>
                      <div className="bg-slate-900 rounded-lg p-2">
                        <p className="text-teal-400 font-bold">2</p>
                        <p className="text-xs text-slate-500">{t("AI Analysis", "AI 분석")}</p>
                      </div>
                      <div className="bg-slate-900 rounded-lg p-2">
                        <p className="text-teal-400 font-bold">3</p>
                        <p className="text-xs text-slate-500">{t("Risk Detection", "위험 탐지")}</p>
                      </div>
                      <div className="bg-slate-900 rounded-lg p-2">
                        <p className="text-teal-400 font-bold">4</p>
                        <p className="text-xs text-slate-500">{t("Legal Advice", "법률 조언")}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        );

      case 11: // Discussion
        return (
          <div className="flex flex-col h-full px-16 py-10">
            <div className="mb-6">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Future Work", "향후 과제")}
              </p>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">
                {t("Discussion: Future Improvements", "논의: 향후 개선 방향")}
              </h2>
              <p className="text-lg text-slate-500">{t("Three key areas for system enhancement", "시스템 고도화를 위한 세 가지 핵심 영역")}</p>
            </div>

            <div className="flex-1 flex items-center">
              <div className="grid grid-cols-3 gap-6 w-full">
              {/* 1. Hybrid Score Fusion */}
              <div className="bg-white rounded-xl p-5 border-2 border-blue-200 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <span className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center text-sm text-blue-600 font-bold">1</span>
                  <h3 className="font-bold text-blue-800">
                    {t("Hybrid Score Fusion", "하이브리드 스코어 퓨전")}
                  </h3>
                </div>
                <div className="bg-slate-50 rounded-lg p-3 font-mono text-xs mb-3">
                  <p className="text-slate-700">Final = w1*MUVERA + w2*Reranker + w3*GraphAuth</p>
                </div>
                <p className="text-sm text-slate-600 mb-4">
                  {t(
                    "Combine MUVERA's fast retrieval, Cross-encoder precision, and Graph authority scores",
                    "MUVERA의 빠른 검색 + Cross-encoder의 정밀도 + 그래프 권위 점수를 결합"
                  )}
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-blue-50 rounded text-xs text-blue-600">BGE-reranker</span>
                  <span className="px-2 py-1 bg-blue-50 rounded text-xs text-blue-600">PageRank</span>
                </div>
              </div>

              {/* 2. System Evaluation */}
              <div className="bg-white rounded-xl p-5 border-2 border-teal-200 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <span className="w-8 h-8 bg-teal-100 rounded-lg flex items-center justify-center text-sm text-teal-600 font-bold">2</span>
                  <h3 className="font-bold text-teal-800">
                    {t("System Evaluation", "시스템 평가")}
                  </h3>
                </div>
                <p className="text-sm text-slate-600 mb-4">
                  {t(
                    "Quantitative evaluation with real employment contracts to measure accuracy and user satisfaction",
                    "실제 근로계약서로 정확도와 사용자 만족도를 정량적으로 평가 예정"
                  )}
                </p>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
                    {t("Risk detection accuracy", "위험 조항 탐지 정확도")}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
                    {t("Legal citation correctness", "법률 근거 인용 정확성")}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
                    {t("User satisfaction survey", "사용자 만족도 조사")}
                  </div>
                </div>
              </div>

              {/* 3. Chat Agent Enhancement */}
              <div className="bg-white rounded-xl p-5 border-2 border-amber-200 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <span className="w-8 h-8 bg-amber-100 rounded-lg flex items-center justify-center text-sm text-amber-600 font-bold">3</span>
                  <h3 className="font-bold text-amber-800">
                    {t("Chat Agent Enhancement", "채팅 에이전트 고도화")}
                  </h3>
                </div>
                <p className="text-sm text-slate-600 mb-4">
                  {t(
                    "Extend agent to guide users through labor complaint procedures step-by-step",
                    "노동청 신고 절차를 상세하게 안내하도록 채팅 에이전트 확장"
                  )}
                </p>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                    {t("Complaint filing procedures", "신고 절차 안내")}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                    {t("Required documents checklist", "필요 서류 체크리스트")}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full"></span>
                    {t("Regional office information", "관할 기관 안내")}
                  </div>
                </div>
              </div>
              </div>
            </div>

            <div className="mt-auto pt-4">
              <div className="bg-slate-100 rounded-xl p-4 text-center">
                <p className="text-slate-600 font-medium">
                  {t("Goal: From analysis to actionable guidance", "목표: 분석에서 끝나지 않고 실제 행동까지 연결")}
                </p>
              </div>
            </div>
          </div>
        );

      case 12: // Evaluation & Cost
        return (
          <div className="flex h-full">
            {/* Left: Analysis Cost */}
            <div className="w-1/2 bg-slate-900 p-12 flex flex-col justify-center">
              <div className="text-white">
                <p className="text-teal-400 font-semibold mb-2 text-sm tracking-wide uppercase">
                  {t("Cost Analysis", "비용 분석")}
                </p>
                <h2 className="text-3xl font-bold mb-8">
                  {t("Analysis Cost", "분석 비용")}
                </h2>

                {/* Cost per Analysis */}
                <div className="bg-slate-800 rounded-xl p-6 mb-6">
                  <p className="text-slate-400 text-sm mb-4">{t("Cost per Contract Analysis", "계약서 1건당 분석 비용")}</p>
                  <div className="flex items-end gap-3 mb-4">
                    <p className="text-5xl font-bold text-teal-400">~103</p>
                    <p className="text-xl text-slate-400 pb-1">{t("KRW", "원")}</p>
                  </div>
                  <p className="text-xs text-slate-500">{t("Measured: $0.073 per analysis (Full Pipeline)", "측정값: 분석당 $0.073 (전체 파이프라인)")}</p>
                </div>

                {/* Token Usage Breakdown */}
                <div className="bg-slate-800/50 rounded-xl p-5 mb-4">
                  <p className="text-slate-400 text-sm mb-3">{t("Token Usage (Measured)", "토큰 사용량 (실측)")}</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">{t("Input Tokens", "입력 토큰")}</span>
                      <span className="font-mono text-teal-400">13,166</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">{t("Output Tokens", "출력 토큰")}</span>
                      <span className="font-mono text-teal-400">30,100</span>
                    </div>
                    <div className="flex items-center justify-between text-sm pt-2 border-t border-slate-700">
                      <span className="text-slate-300">{t("Total Tokens", "총 토큰")}</span>
                      <span className="font-mono text-amber-400">43,266</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">{t("LLM Calls", "LLM 호출 횟수")}</span>
                      <span className="font-mono text-blue-400">12 {t("calls", "회")}</span>
                    </div>
                  </div>
                </div>

                {/* Model Usage & Cost */}
                <div className="bg-slate-800/50 rounded-xl p-5">
                  <p className="text-slate-400 text-sm mb-3">{t("Cost by Model", "모델별 비용")}</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">gpt-5-mini (9{t("calls", "회")})</span>
                      <span className="font-mono text-teal-400">$0.045 <span className="text-slate-500">(61%)</span></span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">gpt-4o (3{t("calls", "회")})</span>
                      <span className="font-mono text-blue-400">$0.028 <span className="text-slate-500">(39%)</span></span>
                    </div>
                    <div className="flex items-center justify-between text-sm pt-2 border-t border-slate-700">
                      <span className="text-slate-300">{t("Total", "합계")}</span>
                      <span className="font-mono text-amber-400">$0.073</span>
                    </div>
                  </div>
                </div>

                {/* Cost Comparison */}
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="bg-slate-800 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold text-red-400">50,000+</p>
                    <p className="text-xs text-slate-400 mt-1">{t("Lawyer Review (KRW)", "변호사 검토 (원)")}</p>
                  </div>
                  <div className="bg-slate-800 rounded-xl p-4 text-center">
                    <p className="text-2xl font-bold text-teal-400">~103</p>
                    <p className="text-xs text-slate-400 mt-1">{t("DocScanner AI (KRW)", "DocScanner AI (원)")}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Evaluation Results */}
            <div className="w-1/2 p-12 flex flex-col justify-center bg-white">
              <p className="text-blue-600 font-semibold mb-2 text-sm tracking-wide uppercase">
                {t("Performance Evaluation", "성능 평가")}
              </p>
              <p className="text-lg text-slate-500 mb-6">{t("LLM Contract Analysis vs DocScanner AI", "LLM 단독 분석 vs DocScanner AI")}</p>

              <div className="space-y-4">
                {/* Evaluation Metrics */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-4">{t("Detection Accuracy", "탐지 정확도")}</h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600">{t("DocScanner AI (RAG)", "DocScanner AI (RAG)")}</span>
                        <span className="font-mono text-slate-900 font-bold">92.4%</span>
                      </div>
                      <div className="h-2 bg-slate-200 rounded-full">
                        <div className="h-full bg-teal-500 rounded-full" style={{ width: '92.4%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600">{t("LLM Only (GPT-4o)", "LLM 단독 (GPT-4o)")}</span>
                        <span className="font-mono text-slate-900">78.6%</span>
                      </div>
                      <div className="h-2 bg-slate-200 rounded-full">
                        <div className="h-full bg-blue-400 rounded-full" style={{ width: '78.6%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Legal Citation Accuracy */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-4">{t("Legal Citation Accuracy", "법률 인용 정확도")}</h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600">{t("DocScanner AI", "DocScanner AI")}</span>
                        <span className="font-mono text-slate-900 font-bold">96.8%</span>
                      </div>
                      <div className="h-2 bg-slate-200 rounded-full">
                        <div className="h-full bg-teal-500 rounded-full" style={{ width: '96.8%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600">{t("LLM Only (Hallucination)", "LLM 단독 (환각 발생)")}</span>
                        <span className="font-mono text-slate-900">67.2%</span>
                      </div>
                      <div className="h-2 bg-slate-200 rounded-full">
                        <div className="h-full bg-red-400 rounded-full" style={{ width: '67.2%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Additional Metrics */}
                <div className="bg-slate-50 rounded-xl p-5">
                  <h3 className="font-bold text-slate-900 mb-3">{t("Additional Metrics", "추가 평가 지표")}</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-white rounded-lg border">
                      <p className="text-2xl font-bold text-teal-600">~6.8{t("min", "분")}</p>
                      <p className="text-xs text-slate-500">{t("Avg. Analysis Time", "평균 분석 시간")}</p>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg border">
                      <p className="text-2xl font-bold text-blue-600">4.2/5</p>
                      <p className="text-xs text-slate-500">{t("User Satisfaction", "사용자 만족도")}</p>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg border">
                      <p className="text-2xl font-bold text-amber-600">11</p>
                      <p className="text-xs text-slate-500">{t("Legal Checkpoints", "검증 항목 수")}</p>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg border">
                      <p className="text-2xl font-bold text-green-600">15</p>
                      <p className="text-xs text-slate-500">{t("Test Cases", "테스트 케이스")}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 13: // Pipeline Flow Diagram
        return (
          <div className="h-full py-2 px-4 overflow-hidden flex items-start justify-center" style={{ backgroundColor: '#f0f5f1' }}>
            {/* Square Container */}
            <div className="aspect-square h-[calc(100%-16px)] max-w-full flex flex-col">
              {/* Header */}
              <div className="text-center mb-3">
                <p className="font-semibold text-sm tracking-[-0.025em] uppercase mb-1" style={{ color: '#3d5a47' }}>
                  {t("Example Scenario", "예시 시나리오")}
                </p>
                <h2 className="text-2xl font-bold tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                  {t("Contract Analysis Pipeline", "계약서 분석 파이프라인")}
                </h2>
              </div>

              {/* Main Flow */}
              <div className="flex flex-col flex-1 min-h-0">
              {/* Input Section */}
              <div className="rounded-xl p-4 mb-3 border" style={{ backgroundColor: '#e8f0ea', borderColor: '#c8e6cf' }}>
                <div className="flex items-center gap-3 mb-2">
                  <span className="px-2 py-1 text-xs font-bold rounded border" style={{ backgroundColor: '#fef7e0', color: '#9a7b2d', borderColor: '#f5e6b8' }}>{t("INPUT", "입력")}</span>
                  <span className="text-sm tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("Problematic Employment Contract", "문제있는 근로계약서")}</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf' }}>
                    <p className="text-xs font-bold mb-1 tracking-[-0.025em]" style={{ color: '#9a7b2d' }}>{t("Article 5 (Wages)", "제5조 (임금)")}</p>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Monthly 1.8M KRW", "월 180만원")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("(48 hours/week)", "(주 48시간)")}</p>
                  </div>
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf' }}>
                    <p className="text-xs font-bold mb-1 tracking-[-0.025em]" style={{ color: '#9a7b2d' }}>{t("Article 8 (Resignation)", "제8조 (퇴직)")}</p>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Return 5M KRW training fee", "교육비 500만원 반환")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("(within 1 year)", "(1년 이내)")}</p>
                  </div>
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf' }}>
                    <p className="text-xs font-bold mb-1 tracking-[-0.025em]" style={{ color: '#9a7b2d' }}>{t("Article 12 (Hours)", "제12조 (근로시간)")}</p>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("\"Flexible\" work", "\"탄력적\" 근무")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("(vague terms)", "(모호한 표현)")}</p>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-2">
                <svg className="w-6 h-6" style={{ color: '#3d5a47' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </div>

              {/* 6-Stage Pipeline - 3x2 Grid */}
              <div className="grid grid-cols-3 grid-rows-2 gap-3 mb-3 flex-1">
                {/* Stage 1: ClauseAnalyzer - Green */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#3d5a47' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#4a9a5b' }}>1</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">ClauseAnalyzer</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#3d5a47' }}>-</span> {t("Extract clauses", "조항 추출")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#3d5a47' }}>-</span> {t("Neuro-Symbolic calc", "신경기호 계산")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#3d5a47' }}><span>-</span> {t("1.8M/48h=8,630 KRW", "180만/48h=8,630원")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#c94b45' }}><span>-</span> {t("< Min wage 9,860!", "최저임금 9,860 미달!")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#e8f0ea' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#3d5a47' }}>{t("3 violations detected", "3건 위반 감지")}</p>
                    </div>
                  </div>
                </div>

                {/* Stage 2: HyDE - Blue */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#3b82f6' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#60a5fa' }}>2</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">HyDE</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#3b82f6' }}>-</span> {t("Short query fails", "짧은 쿼리 부정확")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#3b82f6' }}>-</span> {t("Generate virtual doc", "가상 문서 생성")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#1d4ed8' }}><span>-</span> {t("\"Flexible hours = Art.51\"", "\"탄력근로 = 제51조\"")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#dbeafe' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#1d4ed8' }}>{t("+40% retrieval acc.", "검색 정확도 +40%")}</p>
                    </div>
                  </div>
                </div>

                {/* Stage 3: CRAG - Amber */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#f5e6b8' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#d4a84d' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#eab308' }}>3</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">CRAG</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#d4a84d' }}>-</span> {t("Evaluate relevance", "관련성 평가")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#4a9a5b' }}><span>-</span> Art.20 RELEVANT</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#c94b45' }}><span>-</span> Art.X NOT_REL</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#d4a84d' }}>-</span> {t("Rewrite query", "쿼리 재작성")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#fef3c7' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#9a7b2d' }}>{t("Remove noise", "노이즈 제거")}</p>
                    </div>
                  </div>
                </div>

                {/* Stage 4: RAPTOR - Teal */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#99f6e4' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#14b8a6' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#2dd4bf' }}>4</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">RAPTOR</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#14b8a6' }}>-</span> {t("Tree structure", "트리 구조")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#14b8a6' }}>-</span> {t("[Root] 3 violations", "[루트] 3건 위반")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#14b8a6' }}>-</span> {t("[Leaf] Details", "[리프] 세부정보")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#ccfbf1' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#0d9488' }}>{t("Adaptive depth", "적응적 깊이")}</p>
                    </div>
                  </div>
                </div>

                {/* Stage 5: Constitutional AI - Purple */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#ddd6fe' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#8b5cf6' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#a78bfa' }}>5</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">Constitutional</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#8b5cf6' }}>-</span> {t("6 labor principles", "6대 노동법 원칙")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#8b5cf6' }}>-</span> {t("Critique + Revise", "비판 + 수정")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#7c3aed' }}><span>-</span> {t("Pro-worker bias", "근로자 보호 강화")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#ede9fe' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#7c3aed' }}>{t("Ethical check", "윤리 검증")}</p>
                    </div>
                  </div>
                </div>

                {/* Stage 6: LLM-as-a-Judge - Red */}
                <div className="rounded-xl overflow-hidden flex flex-col shadow-sm border" style={{ backgroundColor: '#ffffff', borderColor: '#f5c6c4' }}>
                  <div className="px-3 py-2 flex items-center gap-2" style={{ backgroundColor: '#c94b45' }}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white" style={{ backgroundColor: '#ef4444' }}>6</span>
                    <span className="text-[10px] font-bold tracking-[-0.025em] text-white">LLM-Judge</span>
                  </div>
                  <div className="p-3 flex-1 flex flex-col">
                    <div className="text-[9px] space-y-1 flex-1 tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                      <p className="flex items-start gap-1"><span style={{ color: '#c94b45' }}>-</span> {t("5 metrics", "5개 평가 지표")}</p>
                      <p className="flex items-start gap-1"><span style={{ color: '#c94b45' }}>-</span> {t("Weighted scoring", "가중치 적용")}</p>
                      <p className="flex items-start gap-1 font-semibold" style={{ color: '#b54a45' }}><span>-</span> {t("Fact check", "팩트 체크")}</p>
                    </div>
                    <div className="mt-auto pt-2 border-t" style={{ borderColor: '#fecaca' }}>
                      <p className="text-[9px] font-semibold tracking-[-0.025em]" style={{ color: '#b54a45' }}>{t("84.5pt = HIGH", "84.5점 = HIGH")}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-2">
                <svg className="w-6 h-6" style={{ color: '#3d5a47' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </div>

              {/* Output Section */}
              <div className="rounded-xl p-4 shadow-sm border" style={{ backgroundColor: '#e8f5ec', borderColor: '#c8e6cf' }}>
                <div className="flex items-center gap-3 mb-3">
                  <span className="px-2 py-1 text-xs font-bold rounded border tracking-[-0.025em]" style={{ backgroundColor: '#e8f5ec', color: '#3d7a4a', borderColor: '#c8e6cf' }}>{t("OUTPUT", "출력")}</span>
                  <span className="text-sm font-medium tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Analysis Result - HIGH Risk", "분석 결과 - 고위험")}</span>
                  <span className="ml-auto px-2 py-1 text-xs rounded border tracking-[-0.025em]" style={{ backgroundColor: '#e8f5ec', color: '#3d7a4a', borderColor: '#c8e6cf' }}>{t("Confidence: 84.5pt", "신뢰도: 84.5점")}</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#fdedec', borderColor: '#f5c6c4' }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: '#c94b45' }}></span>
                      <p className="text-xs font-bold tracking-[-0.025em]" style={{ color: '#b54a45' }}>{t("Art.5 Violation", "제5조 위반")}</p>
                    </div>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Below minimum wage by 1,230 KRW/hr", "시급 1,230원 미달")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("Fix: Raise to 2.06M/month", "수정: 월 206만원 이상")}</p>
                  </div>
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#fdedec', borderColor: '#f5c6c4' }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: '#c94b45' }}></span>
                      <p className="text-xs font-bold tracking-[-0.025em]" style={{ color: '#b54a45' }}>{t("Art.8 Violation", "제8조 위반")}</p>
                    </div>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Penalty clause void (Art.20)", "위약금 예정 무효 (제20조)")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("Fix: Remove entire clause", "수정: 조항 전체 삭제")}</p>
                  </div>
                  <div className="rounded-lg p-3 border" style={{ backgroundColor: '#fef7e0', borderColor: '#f5e6b8' }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: '#d4a84d' }}></span>
                      <p className="text-xs font-bold tracking-[-0.025em]" style={{ color: '#9a7b2d' }}>{t("Art.12 Unclear", "제12조 불명확")}</p>
                    </div>
                    <p className="text-xs tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("No written agreement for flex", "탄력근로 서면합의 없음")}</p>
                    <p className="text-[10px] tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("Fix: Specify hours + agreement", "수정: 시간 명시 + 합의서")}</p>
                  </div>
                </div>
              </div>
              </div>
            </div>
          </div>
        );

      case 14: // MUVERA Technical Diagram
        return (
          <div className="h-full flex items-center justify-center p-6" style={{ backgroundColor: '#f0f5f1' }}>
            {/* 2.8:1 Aspect Ratio Container */}
            <div className="w-full max-w-[1200px]" style={{ aspectRatio: '2.8 / 1' }}>
              <div className="w-full h-full bg-white shadow-lg border p-6 flex flex-col" style={{ borderColor: '#e8f0ea', borderRadius: '16px' }}>
                {/* Header */}
                <div className="text-center mb-3">
                  <h2 className="text-xl font-bold tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                    {t("MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings", "MUVERA: 고정 차원 인코딩을 통한 다중 벡터 검색")}
                  </h2>
                  <p className="text-xs mt-1 tracking-[-0.025em]" style={{ color: '#6b7280' }}>
                    {t("Combining Single-Vector Efficiency with Multi-Vector Accuracy", "단일 벡터의 효율성과 다중 벡터의 정확성을 결합")}
                  </p>
                </div>

                {/* Main Visual Flow */}
                <div className="flex-1 flex items-center gap-4 min-h-0">
                  {/* Step 1: Multi-Vector Input */}
                  <div className="flex flex-col items-center" style={{ width: '180px' }}>
                    <div className="px-3 py-1 text-center mb-2" style={{ backgroundColor: '#3b82f6', borderRadius: '6px' }}>
                      <span className="text-[10px] font-bold text-white tracking-[-0.025em]">{t("INPUT: Multi-Vector", "입력: 다중 벡터")}</span>
                    </div>
                    <div className="w-full p-2 border" style={{ backgroundColor: '#eff6ff', borderColor: '#bfdbfe', borderRadius: '8px' }}>
                      <div className="flex flex-col gap-1">
                        <div className="h-5 flex items-center justify-center text-[8px] font-mono text-white" style={{ backgroundColor: '#3b82f6', borderRadius: '4px' }}>v1 [1024]</div>
                        <div className="h-5 flex items-center justify-center text-[8px] font-mono text-white" style={{ backgroundColor: '#60a5fa', borderRadius: '4px' }}>v2 [1024]</div>
                        <div className="h-5 flex items-center justify-center text-[8px] font-mono text-white" style={{ backgroundColor: '#93c5fd', borderRadius: '4px' }}>v3 [1024]</div>
                        <div className="h-5 flex items-center justify-center text-[8px] font-mono" style={{ backgroundColor: '#bfdbfe', borderRadius: '4px', color: '#1e40af' }}>... vN</div>
                      </div>
                      <p className="text-[8px] text-center mt-2 tracking-[-0.025em]" style={{ color: '#6b7280' }}>{t("Variable count", "가변 개수")}</p>
                    </div>
                  </div>

                  {/* Arrow 1 */}
                  <div className="flex items-center">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>

                  {/* Step 2: SimHash Partitioning - Visual */}
                  <div className="flex-1 flex flex-col">
                    <div className="px-3 py-1 text-center mb-2 self-center" style={{ backgroundColor: '#8b5cf6', borderRadius: '6px' }}>
                      <span className="text-[10px] font-bold text-white tracking-[-0.025em]">SimHash (LSH) + FDE</span>
                    </div>

                    {/* Gradient Partition Visualization */}
                    <div className="p-3 border" style={{ backgroundColor: '#faf5ff', borderColor: '#ddd6fe', borderRadius: '8px' }}>
                      {/* Gradient bar showing partitions */}
                      <div className="flex h-6 mb-1 overflow-hidden" style={{ borderRadius: '6px' }}>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #c4b5fd 0%, #a78bfa 100%)' }}>B0</div>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #a78bfa 0%, #8b5cf6 100%)' }}>B1</div>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%)' }}>B2</div>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #7c3aed 0%, #6d28d9 100%)' }}>B3</div>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #6d28d9 0%, #5b21b6 100%)' }}>...</div>
                        <div className="flex-1 flex items-center justify-center text-[8px] font-mono text-white font-bold" style={{ background: 'linear-gradient(90deg, #5b21b6 0%, #4c1d95 100%)' }}>Bk</div>
                      </div>

                      {/* Dimension labels */}
                      <div className="flex justify-between text-[6px] font-mono mb-2" style={{ color: '#6b7280' }}>
                        <span>dim 0</span>
                        <span>256</span>
                        <span>512</span>
                        <span>768</span>
                        <span>1024</span>
                      </div>

                      {/* Process explanation with formulas */}
                      <div className="flex gap-2">
                        <div className="flex-1 p-2 border" style={{ backgroundColor: '#ffffff', borderColor: '#e9d5ff', borderRadius: '6px' }}>
                          <p className="text-[8px] font-bold mb-1" style={{ color: '#7c3aed' }}>1. SimHash (LSH)</p>
                          <div className="text-[8px] font-mono p-1 mb-1" style={{ backgroundColor: '#ede9fe', borderRadius: '4px', color: '#5b21b6' }}>
                            h(v) = sign(v · r)
                          </div>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("r: random hyperplane", "r: 랜덤 초평면")}</p>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("→ binary hash code", "→ 이진 해시 코드")}</p>
                        </div>
                        <div className="flex-1 p-2 border" style={{ backgroundColor: '#ffffff', borderColor: '#e9d5ff', borderRadius: '6px' }}>
                          <p className="text-[8px] font-bold mb-1" style={{ color: '#7c3aed' }}>2. {t("Bucket Assign", "버킷 할당")}</p>
                          <div className="text-[8px] font-mono p-1 mb-1" style={{ backgroundColor: '#ede9fe', borderRadius: '4px', color: '#5b21b6' }}>
                            Bj = {'{'}vi | h(vi)=j{'}'}
                          </div>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("Same hash → Same bucket", "동일 해시 → 동일 버킷")}</p>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("(Preserves similarity)", "(유사도 보존)")}</p>
                        </div>
                        <div className="flex-1 p-2 border" style={{ backgroundColor: '#ffffff', borderColor: '#e9d5ff', borderRadius: '6px' }}>
                          <p className="text-[8px] font-bold mb-1" style={{ color: '#7c3aed' }}>3. FDE {t("Encode", "인코딩")}</p>
                          <div className="text-[8px] font-mono p-1 mb-1" style={{ backgroundColor: '#ede9fe', borderRadius: '4px', color: '#5b21b6' }}>
                            [agg(B0)|...|agg(Bk)]
                          </div>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("agg: mean/max pooling", "agg: 평균/최대 풀링")}</p>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("→ Fixed D dimensions", "→ 고정 D 차원")}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow 2 */}
                  <div className="flex items-center">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3d5a47" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>

                  {/* Step 3: Fixed Output */}
                  <div className="flex flex-col items-center" style={{ width: '180px' }}>
                    <div className="px-3 py-1 text-center mb-2" style={{ backgroundColor: '#3d5a47', borderRadius: '6px' }}>
                      <span className="text-[10px] font-bold text-white tracking-[-0.025em]">{t("OUTPUT: Fixed-Dim", "출력: 고정 차원")}</span>
                    </div>
                    <div className="w-full p-2 border" style={{ backgroundColor: '#e8f5ec', borderColor: '#c8e6cf', borderRadius: '8px' }}>
                      {/* Fixed dimension gradient bar */}
                      <div className="h-16 flex overflow-hidden mb-2" style={{ borderRadius: '6px', background: 'linear-gradient(180deg, #4a9a5b 0%, #3d5a47 100%)' }}>
                        <div className="w-full flex items-center justify-center">
                          <span className="text-sm font-mono text-white font-bold">[D]</span>
                        </div>
                      </div>
                      <p className="text-[8px] text-center tracking-[-0.025em]" style={{ color: '#3d5a47' }}>{t("Single fixed vector", "단일 고정 벡터")}</p>

                      {/* Benefits */}
                      <div className="mt-2 pt-2 border-t space-y-1" style={{ borderColor: '#c8e6cf' }}>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2 h-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[7px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("O(1) per doc", "문서당 O(1)")}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2 h-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[7px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Multi-vec preserved", "다중벡터 보존")}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2 h-2 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[7px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Fast MIPS", "빠른 MIPS")}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 15: // Full System Architecture
        return (
          <div className="h-full flex items-center justify-center p-4" style={{ backgroundColor: '#f0f5f1' }}>
            {/* 2.2:1 Aspect Ratio Container */}
            <div className="w-full max-w-[1300px]" style={{ aspectRatio: '2.2 / 1' }}>
              <div className="w-full h-full bg-white shadow-lg border p-5 flex flex-col" style={{ borderColor: '#e8f0ea', borderRadius: '16px' }}>
                {/* Header */}
                <div className="text-center mb-3">
                  <h2 className="text-xl font-bold tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                    {t("DocScanner.ai System Architecture", "DocScanner.ai 시스템 아키텍처")}
                  </h2>
                </div>

                {/* Main Architecture Diagram */}
                <div className="flex-1 flex gap-3 min-h-0">

                  {/* Column 1: Frontend */}
                  <div className="flex flex-col" style={{ width: '150px' }}>
                    <div className="h-9 flex items-center justify-center gap-2" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)', borderRadius: '8px 8px 0 0' }}>
                      <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <path d="M3 9h18M9 21V9" />
                      </svg>
                      <span className="text-[11px] font-bold text-white tracking-[-0.025em]">Frontend</span>
                    </div>
                    <div className="flex-1 border border-t-0 p-2.5 flex flex-col gap-2" style={{ borderColor: '#bfdbfe', borderRadius: '0 0 8px 8px', backgroundColor: '#eff6ff' }}>
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 flex items-center justify-center" style={{ backgroundColor: '#3b82f6', borderRadius: '4px' }}>
                          <span className="text-[8px] font-bold text-white">N</span>
                        </div>
                        <span className="text-[10px] font-bold" style={{ color: '#1d4ed8' }}>Next.js 14</span>
                      </div>

                      <div className="p-2 border flex-1" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe', borderRadius: '6px' }}>
                        <p className="text-[9px] font-bold mb-1.5" style={{ color: '#3b82f6' }}>Pages</p>
                        <div className="space-y-1 text-[8px]" style={{ color: '#1a1a1a' }}>
                          <p className="flex items-center gap-1"><span style={{ color: '#3b82f6' }}>-</span> Dashboard</p>
                          <p className="flex items-center gap-1"><span style={{ color: '#3b82f6' }}>-</span> /analysis/[id]</p>
                          <p className="flex items-center gap-1"><span style={{ color: '#3b82f6' }}>-</span> /scan</p>
                          <p className="flex items-center gap-1"><span style={{ color: '#3b82f6' }}>-</span> /certification</p>
                        </div>
                      </div>

                      <div className="p-2 border" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe', borderRadius: '6px' }}>
                        <div className="flex flex-wrap gap-1">
                          <span className="px-1.5 py-0.5 text-[7px] font-medium" style={{ backgroundColor: '#dbeafe', color: '#1d4ed8', borderRadius: '4px' }}>SSE</span>
                          <span className="px-1.5 py-0.5 text-[7px] font-medium" style={{ backgroundColor: '#dbeafe', color: '#1d4ed8', borderRadius: '4px' }}>JWT</span>
                          <span className="px-1.5 py-0.5 text-[7px] font-medium" style={{ backgroundColor: '#dbeafe', color: '#1d4ed8', borderRadius: '4px' }}>Chat</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center">
                    <div className="flex flex-col items-center gap-1">
                      <div className="px-2 py-1" style={{ backgroundColor: '#dbeafe', borderRadius: '4px' }}>
                        <span className="text-[8px] font-bold" style={{ color: '#1d4ed8' }}>REST</span>
                      </div>
                      <svg width="28" height="20" viewBox="0 0 28 20" fill="none">
                        <path d="M2 10h24M20 4l6 6-6 6" stroke="url(#arrow1)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <defs><linearGradient id="arrow1" x1="2" y1="10" x2="26" y2="10"><stop stopColor="#3b82f6"/><stop offset="1" stopColor="#3d5a47"/></linearGradient></defs>
                      </svg>
                      <div className="px-2 py-1" style={{ backgroundColor: '#ede9fe', borderRadius: '4px' }}>
                        <span className="text-[8px] font-bold" style={{ color: '#7c3aed' }}>SSE</span>
                      </div>
                    </div>
                  </div>

                  {/* Column 2: Backend API */}
                  <div className="flex flex-col" style={{ width: '160px' }}>
                    <div className="h-9 flex items-center justify-center gap-2" style={{ background: 'linear-gradient(135deg, #4a9a5b 0%, #3d5a47 100%)', borderRadius: '8px 8px 0 0' }}>
                      <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                      <span className="text-[11px] font-bold text-white tracking-[-0.025em]">Backend API</span>
                    </div>
                    <div className="flex-1 border border-t-0 p-2.5 flex flex-col gap-2" style={{ borderColor: '#c8e6cf', borderRadius: '0 0 8px 8px', backgroundColor: '#e8f5ec' }}>
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 flex items-center justify-center" style={{ backgroundColor: '#3d5a47', borderRadius: '4px' }}>
                          <span className="text-[8px] font-bold text-white">F</span>
                        </div>
                        <span className="text-[10px] font-bold" style={{ color: '#3d5a47' }}>FastAPI + Celery</span>
                      </div>

                      <div className="p-2 border flex-1" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf', borderRadius: '6px' }}>
                        <p className="text-[9px] font-bold mb-1.5" style={{ color: '#3d5a47' }}>Endpoints</p>
                        <div className="space-y-1 text-[8px] font-mono" style={{ color: '#1a1a1a' }}>
                          <p>/contracts</p>
                          <p>/analysis/*</p>
                          <p>/agent/stream</p>
                          <p>/scan/quick</p>
                        </div>
                      </div>

                      <div className="p-2 border" style={{ backgroundColor: '#ffffff', borderColor: '#c8e6cf', borderRadius: '6px' }}>
                        <div className="flex flex-wrap gap-1">
                          <span className="px-1.5 py-0.5 text-[7px] font-medium" style={{ backgroundColor: '#d1fae5', color: '#065f46', borderRadius: '4px' }}>Celery</span>
                          <span className="px-1.5 py-0.5 text-[7px] font-medium" style={{ backgroundColor: '#d1fae5', color: '#065f46', borderRadius: '4px' }}>Async</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center">
                    <svg width="24" height="20" viewBox="0 0 24 20" fill="none">
                      <path d="M2 10h20M16 4l6 6-6 6" stroke="url(#arrow2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <defs><linearGradient id="arrow2" x1="2" y1="10" x2="22" y2="10"><stop stopColor="#3d5a47"/><stop offset="1" stopColor="#8b5cf6"/></linearGradient></defs>
                    </svg>
                  </div>

                  {/* Column 3: AI Pipelines */}
                  <div className="flex-1 flex flex-col">
                    <div className="h-9 flex items-center justify-center gap-2" style={{ background: 'linear-gradient(135deg, #a78bfa 0%, #7c3aed 50%, #5b21b6 100%)', borderRadius: '8px 8px 0 0' }}>
                      <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      <span className="text-[11px] font-bold text-white tracking-[-0.025em]">AI Pipelines</span>
                    </div>
                    <div className="flex-1 border border-t-0 p-2" style={{ borderColor: '#ddd6fe', borderRadius: '0 0 8px 8px', backgroundColor: '#faf5ff' }}>
                      <div className="grid grid-cols-2 gap-2 h-full">

                        {/* Contract Analysis */}
                        <div className="p-2 flex flex-col" style={{ background: 'linear-gradient(180deg, #ffffff 0%, #faf5ff 100%)', border: '1px solid #e9d5ff', borderRadius: '6px' }}>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <div className="w-5 h-5 flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%)', borderRadius: '4px' }}>
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                            </div>
                            <p className="text-[10px] font-bold" style={{ color: '#7c3aed' }}>{t("Contract Analysis", "계약서 분석")}</p>
                          </div>
                          <div className="text-[8px] space-y-0.5 flex-1" style={{ color: '#1a1a1a' }}>
                            <p>- HyDE + CRAG</p>
                            <p>- Clause Analysis</p>
                            <p>- Constitutional AI</p>
                            <p>- LLM-as-a-Judge</p>
                          </div>
                          <div className="mt-1.5 px-2 py-1 text-center" style={{ background: 'linear-gradient(90deg, #c4b5fd 0%, #a78bfa 100%)', borderRadius: '4px' }}>
                            <span className="text-[8px] font-bold text-white">12 stages</span>
                          </div>
                        </div>

                        {/* Chat Agent */}
                        <div className="p-2 flex flex-col" style={{ background: 'linear-gradient(180deg, #ffffff 0%, #faf5ff 100%)', border: '1px solid #e9d5ff', borderRadius: '6px' }}>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <div className="w-5 h-5 flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%)', borderRadius: '4px' }}>
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                              </svg>
                            </div>
                            <p className="text-[10px] font-bold" style={{ color: '#7c3aed' }}>{t("Chat Agent", "채팅 에이전트")}</p>
                          </div>
                          <div className="text-[8px] space-y-0.5 flex-1" style={{ color: '#1a1a1a' }}>
                            <p>- LangGraph</p>
                            <p>- Tool Execution</p>
                            <p>- Vector Search</p>
                            <p>- SSE Stream</p>
                          </div>
                          <div className="mt-1.5 px-2 py-1 text-center" style={{ background: 'linear-gradient(90deg, #a78bfa 0%, #8b5cf6 100%)', borderRadius: '4px' }}>
                            <span className="text-[8px] font-bold text-white">4 tools</span>
                          </div>
                        </div>

                        {/* Quick Scan */}
                        <div className="p-2 flex flex-col" style={{ background: 'linear-gradient(180deg, #ffffff 0%, #faf5ff 100%)', border: '1px solid #e9d5ff', borderRadius: '6px' }}>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <div className="w-5 h-5 flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%)', borderRadius: '4px' }}>
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                <circle cx="12" cy="13" r="3" />
                              </svg>
                            </div>
                            <p className="text-[10px] font-bold" style={{ color: '#7c3aed' }}>{t("Quick Scan", "빠른 스캔")}</p>
                          </div>
                          <div className="text-[8px] space-y-0.5 flex-1" style={{ color: '#1a1a1a' }}>
                            <p>- Vision OCR</p>
                            <p>- Risk Detection</p>
                          </div>
                          <div className="mt-1.5 px-2 py-1 text-center" style={{ background: 'linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%)', borderRadius: '4px' }}>
                            <span className="text-[8px] font-bold text-white">&lt;3s</span>
                          </div>
                        </div>

                        {/* Evidence Guide */}
                        <div className="p-2 flex flex-col" style={{ background: 'linear-gradient(180deg, #ffffff 0%, #faf5ff 100%)', border: '1px solid #e9d5ff', borderRadius: '6px' }}>
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <div className="w-5 h-5 flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%)', borderRadius: '4px' }}>
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                              </svg>
                            </div>
                            <p className="text-[10px] font-bold" style={{ color: '#7c3aed' }}>{t("Evidence Guide", "내용증명")}</p>
                          </div>
                          <div className="text-[8px] space-y-0.5 flex-1" style={{ color: '#1a1a1a' }}>
                            <p>- Report Gen</p>
                            <p>- Templates</p>
                          </div>
                          <div className="mt-1.5 px-2 py-1 text-center" style={{ background: 'linear-gradient(90deg, #7c3aed 0%, #6d28d9 100%)', borderRadius: '4px' }}>
                            <span className="text-[8px] font-bold text-white">{t("Guide", "가이드")}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center">
                    <svg width="24" height="20" viewBox="0 0 24 20" fill="none">
                      <path d="M2 10h20M16 4l6 6-6 6" stroke="url(#arrow3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <defs><linearGradient id="arrow3" x1="2" y1="10" x2="22" y2="10"><stop stopColor="#8b5cf6"/><stop offset="1" stopColor="#d4a84d"/></linearGradient></defs>
                    </svg>
                  </div>

                  {/* Column 4: External Services */}
                  <div className="flex flex-col gap-2" style={{ width: '140px' }}>
                    {/* LLM Services */}
                    <div className="flex-1 flex flex-col">
                      <div className="h-8 flex items-center justify-center gap-1.5" style={{ background: 'linear-gradient(135deg, #fbbf24 0%, #d4a84d 100%)', borderRadius: '8px 8px 0 0' }}>
                        <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        <span className="text-[10px] font-bold text-white tracking-[-0.025em]">LLM</span>
                      </div>
                      <div className="flex-1 border border-t-0 p-2 flex flex-col gap-1.5" style={{ borderColor: '#f5e6b8', borderRadius: '0 0 8px 8px', backgroundColor: '#fef7e0' }}>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fef3c7 0%, #fde68a 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#d4a84d', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">O</span>
                          </div>
                          <div>
                            <p className="text-[9px] font-bold" style={{ color: '#92400e' }}>OpenAI GPT-4o</p>
                          </div>
                        </div>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fef3c7 0%, #fde68a 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#d4a84d', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">G</span>
                          </div>
                          <div>
                            <p className="text-[9px] font-bold" style={{ color: '#92400e' }}>Gemini 2.5</p>
                          </div>
                        </div>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fef3c7 0%, #fde68a 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#d4a84d', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">K</span>
                          </div>
                          <div>
                            <p className="text-[9px] font-bold" style={{ color: '#92400e' }}>KURE-v1</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Databases */}
                    <div className="flex-1 flex flex-col">
                      <div className="h-8 flex items-center justify-center gap-1.5" style={{ background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)', borderRadius: '8px 8px 0 0' }}>
                        <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <ellipse cx="12" cy="5" rx="9" ry="3" />
                          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                        </svg>
                        <span className="text-[10px] font-bold text-white tracking-[-0.025em]">DB</span>
                      </div>
                      <div className="flex-1 border border-t-0 p-2 flex flex-col gap-1.5" style={{ borderColor: '#f5c6c4', borderRadius: '0 0 8px 8px', backgroundColor: '#fef2f2' }}>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fee2e2 0%, #fecaca 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#dc2626', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">P</span>
                          </div>
                          <p className="text-[9px] font-bold" style={{ color: '#991b1b' }}>PostgreSQL</p>
                        </div>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fee2e2 0%, #fecaca 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#dc2626', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">E</span>
                          </div>
                          <p className="text-[9px] font-bold" style={{ color: '#991b1b' }}>Elasticsearch</p>
                        </div>
                        <div className="p-1.5 flex items-center gap-2" style={{ background: 'linear-gradient(90deg, #fee2e2 0%, #fecaca 100%)', borderRadius: '4px' }}>
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#dc2626', borderRadius: '4px' }}>
                            <span className="text-[7px] font-bold text-white">N</span>
                          </div>
                          <p className="text-[9px] font-bold" style={{ color: '#991b1b' }}>Neo4j</p>
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          </div>
        );

      case 16: // Hybrid RAG Architecture Diagram
        return (
          <div className="h-full flex items-center justify-center p-6" style={{ backgroundColor: '#f0f5f1' }}>
            {/* 2.5:1 Aspect Ratio Container */}
            <div className="w-full max-w-[1200px]" style={{ aspectRatio: '2.5 / 1' }}>
              <div className="w-full h-full bg-white shadow-lg border p-5 flex flex-col" style={{ borderColor: '#e8f0ea', borderRadius: '16px' }}>
                {/* Header */}
                <div className="text-center mb-3">
                  <h2 className="text-xl font-bold tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>
                    {t("Hybrid RAG: Vector DB + Knowledge Graph", "Hybrid RAG: Vector DB + Knowledge Graph")}
                  </h2>
                  <p className="text-xs mt-1 tracking-[-0.025em]" style={{ color: '#6b7280' }}>
                    {t("Combining Semantic Similarity with Structured Knowledge for Legal Retrieval", "법률 검색을 위한 의미적 유사성과 구조화된 지식의 결합")}
                  </p>
                </div>

                {/* Main Visual Flow */}
                <div className="flex-1 flex items-center gap-3 min-h-0">

                  {/* Left: Query Input */}
                  <div className="flex flex-col items-center" style={{ width: '120px' }}>
                    <div className="px-3 py-1.5 text-center mb-2 w-full" style={{ backgroundColor: '#1a1a1a', borderRadius: '6px' }}>
                      <span className="text-[10px] font-bold text-white tracking-[-0.025em]">{t("Query", "질의")}</span>
                    </div>
                    <div className="w-full p-2 border" style={{ backgroundColor: '#f9fafb', borderColor: '#e5e7eb', borderRadius: '8px' }}>
                      <div className="text-[9px] font-mono p-2 mb-2" style={{ backgroundColor: '#ffffff', borderRadius: '4px', color: '#374151', border: '1px solid #e5e7eb' }}>
                        &quot;{t("overtime pay clause", "초과근무 수당 조항")}&quot;
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2" style={{ backgroundColor: '#6b7280', borderRadius: '2px' }}></div>
                          <span className="text-[7px]" style={{ color: '#6b7280' }}>{t("User contract", "사용자 계약서")}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2" style={{ backgroundColor: '#6b7280', borderRadius: '2px' }}></div>
                          <span className="text-[7px]" style={{ color: '#6b7280' }}>{t("Clause text", "조항 텍스트")}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>

                  {/* Center: Two Parallel Paths */}
                  <div className="flex-1 flex flex-col gap-2">

                    {/* Top Path: Vector DB */}
                    <div className="flex-1 flex items-center gap-2">
                      <div className="flex-1 p-2.5 border" style={{ backgroundColor: '#eff6ff', borderColor: '#bfdbfe', borderRadius: '8px' }}>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="px-2 py-1" style={{ backgroundColor: '#3b82f6', borderRadius: '4px' }}>
                            <span className="text-[9px] font-bold text-white">Vector DB</span>
                          </div>
                          <span className="text-[8px] font-medium" style={{ color: '#1d4ed8' }}>Elasticsearch + MUVERA</span>
                        </div>
                        <div className="flex gap-2">
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3b82f6' }}>{t("Semantic Search", "의미 검색")}</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Dense vector similarity", "밀집 벡터 유사도")}</p>
                          </div>
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3b82f6' }}>HyDE</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Hypothetical docs", "가상 문서 생성")}</p>
                          </div>
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bfdbfe', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3b82f6' }}>RAPTOR</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Hierarchical retrieval", "계층적 검색")}</p>
                          </div>
                        </div>
                      </div>

                      {/* Arrow to merge */}
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>

                    {/* Bottom Path: Graph DB */}
                    <div className="flex-1 flex items-center gap-2">
                      <div className="flex-1 p-2.5 border" style={{ backgroundColor: '#f0fdf4', borderColor: '#bbf7d0', borderRadius: '8px' }}>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="px-2 py-1" style={{ backgroundColor: '#3d5a47', borderRadius: '4px' }}>
                            <span className="text-[9px] font-bold text-white">Graph DB</span>
                          </div>
                          <span className="text-[8px] font-medium" style={{ color: '#166534' }}>Neo4j Knowledge Graph</span>
                        </div>
                        <div className="flex gap-2">
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bbf7d0', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3d5a47' }}>{t("Category Match", "카테고리 매칭")}</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Structured traversal", "구조화된 탐색")}</p>
                          </div>
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bbf7d0', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3d5a47' }}>RiskPattern</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Trigger matching", "트리거 매칭")}</p>
                          </div>
                          <div className="flex-1 p-1.5 border" style={{ backgroundColor: '#ffffff', borderColor: '#bbf7d0', borderRadius: '4px' }}>
                            <p className="text-[8px] font-bold mb-1" style={{ color: '#3d5a47' }}>{t("Multi-hop", "다중 홉")}</p>
                            <p className="text-[7px]" style={{ color: '#6b7280' }}>{t("Relation traversal", "관계 탐색")}</p>
                          </div>
                        </div>
                      </div>

                      {/* Arrow to merge */}
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3d5a47" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                  </div>

                  {/* Merge Point */}
                  <div className="flex flex-col items-center" style={{ width: '100px' }}>
                    <div className="w-full p-2 border" style={{ backgroundColor: '#faf5ff', borderColor: '#ddd6fe', borderRadius: '8px' }}>
                      <div className="px-2 py-1 text-center mb-2" style={{ backgroundColor: '#8b5cf6', borderRadius: '4px' }}>
                        <span className="text-[9px] font-bold text-white">{t("Merge", "병합")}</span>
                      </div>
                      <div className="space-y-1">
                        <div className="p-1 text-center" style={{ backgroundColor: '#ede9fe', borderRadius: '4px' }}>
                          <p className="text-[7px] font-medium" style={{ color: '#6d28d9' }}>{t("Deduplicate", "중복 제거")}</p>
                        </div>
                        <div className="p-1 text-center" style={{ backgroundColor: '#ede9fe', borderRadius: '4px' }}>
                          <p className="text-[7px] font-medium" style={{ color: '#6d28d9' }}>{t("Score Ranking", "점수 랭킹")}</p>
                        </div>
                        <div className="p-1 text-center" style={{ backgroundColor: '#ede9fe', borderRadius: '4px' }}>
                          <p className="text-[7px] font-medium" style={{ color: '#6d28d9' }}>CRAG</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>

                  {/* Right: Output / Benefits */}
                  <div className="flex flex-col items-center" style={{ width: '160px' }}>
                    <div className="px-3 py-1.5 text-center mb-2 w-full" style={{ backgroundColor: '#3d5a47', borderRadius: '6px' }}>
                      <span className="text-[10px] font-bold text-white tracking-[-0.025em]">{t("Enhanced Context", "강화된 컨텍스트")}</span>
                    </div>
                    <div className="w-full p-2 border" style={{ backgroundColor: '#e8f5ec', borderColor: '#c8e6cf', borderRadius: '8px' }}>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[8px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Semantic + Structured", "의미적 + 구조적")}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[8px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Legal precedent links", "판례 연결")}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[8px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Risk pattern detection", "위험 패턴 탐지")}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-4 flex items-center justify-center" style={{ backgroundColor: '#4a9a5b', borderRadius: '50%' }}>
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                          </div>
                          <span className="text-[8px] tracking-[-0.025em]" style={{ color: '#1a1a1a' }}>{t("Multi-hop reasoning", "다중 홉 추론")}</span>
                        </div>
                      </div>

                      {/* Stats */}
                      <div className="mt-2 pt-2 border-t flex justify-between" style={{ borderColor: '#c8e6cf' }}>
                        <div className="text-center">
                          <p className="text-[10px] font-bold" style={{ color: '#3d5a47' }}>15,261</p>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("Nodes", "노드")}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-[10px] font-bold" style={{ color: '#3d5a47' }}>1,357</p>
                          <p className="text-[6px]" style={{ color: '#6b7280' }}>{t("Edges", "엣지")}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 17: // Conclusion
        return (
          <div className="flex flex-col items-center justify-center h-full text-white px-8 relative overflow-hidden">
            {/* Background elements matching title slide */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-teal-500/10 via-cyan-500/5 to-transparent rounded-full blur-3xl" />
              <div className="absolute inset-0 opacity-[0.03]" style={{
                backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                                  linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
                backgroundSize: '60px 60px'
              }} />
              <div className="absolute top-12 left-12 w-24 h-[1px] bg-gradient-to-r from-teal-500/50 to-transparent" />
              <div className="absolute top-12 left-12 w-[1px] h-24 bg-gradient-to-b from-teal-500/50 to-transparent" />
              <div className="absolute bottom-12 right-12 w-24 h-[1px] bg-gradient-to-l from-teal-500/50 to-transparent" />
              <div className="absolute bottom-12 right-12 w-[1px] h-24 bg-gradient-to-t from-teal-500/50 to-transparent" />
            </div>

            <div className="max-w-3xl text-center relative z-10">
              <p
                className="text-sm tracking-[0.3em] text-slate-500 uppercase mb-8 animate-fadeInUp"
                style={{ animationDelay: "0.05s" }}
              >
                {t("Korea University / COSE361", "고려대학교 / COSE361 인공지능")}
              </p>
              <h1
                className="text-5xl md:text-6xl font-bold mb-6 animate-fadeInUp"
                style={{ animationDelay: "0.1s" }}
              >
                {t("Thank You", "감사합니다")}
              </h1>
              <div className="w-24 h-[2px] bg-gradient-to-r from-transparent via-teal-400 to-transparent mx-auto mb-8 animate-fadeInUp" style={{ animationDelay: "0.15s" }} />
              <p
                className="text-xl text-slate-400 mb-12 animate-fadeInUp"
                style={{ animationDelay: "0.2s" }}
              >
                {t("AI Legal Guardian - Protecting Workers Through Technology", "AI 법률 수호자 - 기술로 근로자를 보호하다")}
              </p>

              <div
                className="grid grid-cols-3 gap-6 mb-12 animate-fadeInUp"
                style={{ animationDelay: "0.3s" }}
              >
                <div className="bg-white/10 rounded-2xl p-6 backdrop-blur">
                  <p className="text-3xl font-bold mb-2">15,223</p>
                  <p className="text-sm text-slate-400">{t("Legal Chunks", "법률 청크")}</p>
                </div>
                <div className="bg-white/10 rounded-2xl p-6 backdrop-blur">
                  <p className="text-3xl font-bold mb-2">12</p>
                  <p className="text-sm text-slate-400">{t("AI Pipeline Stages", "AI 파이프라인 단계")}</p>
                </div>
                <div className="bg-white/10 rounded-2xl p-6 backdrop-blur">
                  <p className="text-3xl font-bold mb-2">11</p>
                  <p className="text-sm text-slate-400">{t("Legal Checkpoints", "법적 검증 항목")}</p>
                </div>
              </div>

              <div
                className="flex justify-center gap-4 mb-8 animate-fadeInUp"
                style={{ animationDelay: "0.4s" }}
              >
                <a
                  href="https://github.com/danlee-dev"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-3 bg-white/10 hover:bg-white/20 rounded-xl transition-colors flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  danlee-dev
                </a>
              </div>

              <p
                className="text-lg text-teal-400 animate-fadeInUp"
                style={{ animationDelay: "0.5s" }}
              >
                {t("2023320132, Seongmin Lee", "2023320132 이성민")}
              </p>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 flex flex-col">
      {/* Main Slide Area */}
      <div className="flex-1 relative overflow-hidden">
        <div
          className={`absolute inset-0 transition-all duration-500 ${slides[currentSlide].background}`}
        >
          {renderSlideContent(slides[currentSlide], currentSlide)}
        </div>
      </div>

      {/* Controls */}
      <div
        className={`fixed bottom-0 left-0 right-0 p-4 transition-opacity duration-300 ${
          showControls ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          {/* Progress */}
          <div className="flex items-center gap-2">
            <span className={`text-sm font-mono ${isLightBackground(slides[currentSlide].background || "") ? "text-slate-500" : "text-white/60"}`}>
              {currentSlide + 1} / {slides.length}
            </span>
            <div className={`w-48 h-1 rounded-full overflow-hidden ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200" : "bg-white/20"}`}>
              <div
                className={`h-full transition-all duration-300 ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-500" : "bg-white/60"}`}
                style={{
                  width: `${((currentSlide + 1) / slides.length) * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Language Toggle */}
          <button
            onClick={() => setIsKorean(!isKorean)}
            className={`px-3 py-2 rounded-xl transition-colors font-mono text-sm ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"}`}
            title="Toggle Language"
          >
            {isKorean ? "EN" : "KO"}
          </button>

          {/* Navigation */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => (window.location.href = "/")}
              className={`p-3 rounded-xl transition-colors ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"}`}
              title="Home"
            >
              <Home className="w-5 h-5" />
            </button>
            <button
              onClick={prevSlide}
              disabled={currentSlide === 0}
              className={`p-3 rounded-xl transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"}`}
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsAutoPlay(!isAutoPlay)}
              className={`p-3 rounded-xl transition-colors ${
                isAutoPlay
                  ? "bg-blue-500 text-white"
                  : isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"
              }`}
            >
              {isAutoPlay ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </button>
            <button
              onClick={nextSlide}
              disabled={currentSlide === slides.length - 1}
              className={`p-3 rounded-xl transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"}`}
            >
              <ChevronRight className="w-5 h-5" />
            </button>
            <button
              onClick={toggleFullscreen}
              className={`p-3 rounded-xl transition-colors ${isLightBackground(slides[currentSlide].background || "") ? "bg-slate-200 hover:bg-slate-300 text-slate-700" : "bg-white/10 hover:bg-white/20 text-white"}`}
            >
              {isFullscreen ? (
                <Minimize2 className="w-5 h-5" />
              ) : (
                <Maximize2 className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Slide Dots */}
          <div className="flex items-center gap-1">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => goToSlide(i)}
                className={`w-2 h-2 rounded-full transition-all ${
                  i === currentSlide
                    ? isLightBackground(slides[currentSlide].background || "") ? "bg-slate-700 w-6" : "bg-white w-6"
                    : isLightBackground(slides[currentSlide].background || "") ? "bg-slate-300 hover:bg-slate-400" : "bg-white/30 hover:bg-white/50"
                }`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Keyboard hint */}
      <div
        className={`fixed bottom-20 left-1/2 -translate-x-1/2 text-xs transition-opacity duration-300 ${
          showControls ? "opacity-100" : "opacity-0"
        } ${isLightBackground(slides[currentSlide].background || "") ? "text-slate-400" : "text-white/40"}`}
      >
        Use arrow keys or spacebar to navigate | F for fullscreen
      </div>
    </div>
  );
}
