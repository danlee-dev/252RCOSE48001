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
    // Slide 14: Conclusion
    {
      id: 14,
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
                    <div className="flex-1 space-y-3">
                      <div className="bg-blue-100 rounded-xl p-4 text-center">
                        <p className="font-bold text-blue-900">{t("Frontend", "프론트엔드")}</p>
                        <p className="text-xs text-blue-700">Next.js 15</p>
                      </div>
                      <div className="space-y-2 px-2">
                        <div className="text-xs bg-white rounded p-2 border">React 19</div>
                        <div className="text-xs bg-white rounded p-2 border">Tailwind CSS</div>
                        <div className="text-xs bg-white rounded p-2 border">react-pdf</div>
                      </div>
                    </div>

                    {/* Arrow 1 */}
                    <div className="flex items-center">
                      <ChevronRight className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* Backend */}
                    <div className="flex-1 space-y-3">
                      <div className="bg-green-100 rounded-xl p-4 text-center">
                        <p className="font-bold text-green-900">{t("Backend", "백엔드")}</p>
                        <p className="text-xs text-green-700">FastAPI</p>
                      </div>
                      <div className="space-y-2 px-2">
                        <div className="text-xs bg-white rounded p-2 border">LangGraph Agent</div>
                        <div className="text-xs bg-white rounded p-2 border">Celery Tasks</div>
                        <div className="text-xs bg-white rounded p-2 border">SQLAlchemy</div>
                      </div>
                    </div>

                    {/* Arrow 2 */}
                    <div className="flex items-center">
                      <ChevronRight className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* AI Pipeline */}
                    <div className="flex-1 space-y-3">
                      <div className="bg-teal-100 rounded-xl p-4 text-center">
                        <p className="font-bold text-teal-900">{t("AI Pipeline", "AI 파이프라인")}</p>
                        <p className="text-xs text-teal-700">{t("12 Stages", "12단계")}</p>
                      </div>
                      <div className="space-y-2 px-2">
                        <div className="text-xs bg-white rounded p-2 border">GPT-4o / 5-mini</div>
                        <div className="text-xs bg-white rounded p-2 border">KURE-v1 Embed</div>
                        <div className="text-xs bg-white rounded p-2 border">Tavily Search</div>
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
                    <div className="flex-1 space-y-3">
                      <div className="bg-amber-100 rounded-xl p-4 text-center">
                        <p className="font-bold text-amber-900">{t("Data Layer", "데이터 레이어")}</p>
                        <p className="text-xs text-amber-700">{t("Hybrid DB", "하이브리드 DB")}</p>
                      </div>
                      <div className="space-y-2 px-2">
                        <div className="text-xs bg-white rounded p-2 border">PostgreSQL</div>
                        <div className="text-xs bg-white rounded p-2 border">Elasticsearch</div>
                        <div className="text-xs bg-white rounded p-2 border">Neo4j Graph</div>
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

                {/* Tech Stack Summary */}
                <div className="grid grid-cols-6 gap-4 mt-6">
                  {[
                    { name: "GPT-5-mini", type: "LLM" },
                    { name: "Gemini 2.5-flash", type: t("Vision", "비전") },
                    { name: "KURE-v1", type: t("Embed", "임베딩") },
                    { name: "Neo4j", type: t("Graph", "그래프") },
                    { name: "ES", type: t("Vector", "벡터") },
                    { name: "Redis", type: t("Cache", "캐시") },
                  ].map((tech, i) => (
                    <div
                      key={i}
                      className="text-center p-3 bg-white rounded-lg border"
                    >
                      <p className="font-bold text-slate-900">{tech.name}</p>
                      <p className="text-xs text-slate-500">{tech.type}</p>
                    </div>
                  ))}
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

      case 13: // Conclusion
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
