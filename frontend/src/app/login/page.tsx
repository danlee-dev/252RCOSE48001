"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { IconLoading, IconEye, IconEyeOff, IconArrowRight, Logo } from "@/components/icons";

function IconAlertTriangle({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IconShieldCheck({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}

function IconXCircle({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  );
}

function DeviceMockup() {
  return (
    <div className="device-mockup-container">
      <div className="device-mockup">
        <div className="device-screen">
          <div className="device-content">
            <div className="contract-doc">
              {/* Document lines */}
              <div className="contract-line title" />
              <div className="contract-line long" />
              <div className="contract-line medium" />
              <div className="contract-line long" />
              <div className="contract-line short" />
              <div className="contract-line long" />
              <div className="contract-line medium" />
              <div className="contract-line long" />
              <div className="contract-line short" />

              {/* Scan overlay */}
              <div className="scan-overlay">
                <div className="scan-line" />
              </div>

              {/* Highlight reveals */}
              <div className="highlight-reveal danger" />
              <div className="highlight-reveal warning" />
              <div className="highlight-reveal success" />
            </div>
          </div>
        </div>

        {/* Multiple result badges */}
        <div className="result-badge badge-1">
          <div className="result-badge-icon danger">
            <IconXCircle size={16} />
          </div>
          <div>
            <div className="text-[11px] font-semibold text-gray-900 tracking-tight">불공정 조항 발견</div>
            <div className="text-[10px] text-gray-500 tracking-tight">제7조 위약금 조항</div>
          </div>
        </div>

        <div className="result-badge badge-2">
          <div className="result-badge-icon warning">
            <IconAlertTriangle size={16} />
          </div>
          <div>
            <div className="text-[11px] font-semibold text-gray-900 tracking-tight">주의 필요</div>
            <div className="text-[10px] text-gray-500 tracking-tight">자동 갱신 조항</div>
          </div>
        </div>

        <div className="result-badge badge-3">
          <div className="result-badge-icon success">
            <IconShieldCheck size={16} />
          </div>
          <div>
            <div className="text-[11px] font-semibold text-gray-900 tracking-tight">분석 완료</div>
            <div className="text-[10px] text-gray-500 tracking-tight">2건 검토 필요</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await authApi.login(email, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "로그인에 실패했습니다");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-dvh flex">
      {/* Left Panel - Feature Showcase */}
      <div className="hidden lg:flex flex-1 relative overflow-hidden bg-gradient-to-br from-gray-50 via-gray-100 to-gray-200">
        {/* Liquid blobs */}
        <div className="liquid-blob liquid-blob-1" style={{ top: '-5%', left: '20%' }} />
        <div className="liquid-blob liquid-blob-2" style={{ top: '40%', left: '-10%' }} />
        <div className="liquid-blob liquid-blob-3" style={{ bottom: '10%', right: '10%' }} />

        {/* Content */}
        <div className="relative z-10 flex items-center w-full h-full pl-[8%] xl:pl-[10%] pr-8">
          <div className="flex items-center gap-16 xl:gap-20">
            {/* Left: Text content */}
            <div className="max-w-md">
              {/* Logo */}
              <div className="flex items-center gap-3 mb-10">
                <Logo size={40} color="#111827" />
                <span className="text-xl font-bold text-gray-900 tracking-tight">DocScanner AI</span>
              </div>

              {/* Headline */}
              <h1 className="text-4xl xl:text-5xl font-bold text-gray-900 tracking-tight leading-[1.45] mb-5">
                계약서 분석,
                <br />
                AI에게 맡기세요
              </h1>

              <p className="text-base text-gray-600 tracking-tight mb-8 leading-relaxed">
                복잡한 법률 용어와 숨겨진 위험 조항을
                <br />
                AI가 빠르고 정확하게 분석해드립니다.
              </p>

              {/* Stats */}
              <div className="flex gap-10">
                <div>
                  <div className="text-2xl font-bold text-gray-900 tracking-tight">10초</div>
                  <div className="text-sm text-gray-500 tracking-tight mt-1">빠른 스캔</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-gray-900 tracking-tight">5분</div>
                  <div className="text-sm text-gray-500 tracking-tight mt-1">정밀 분석</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-gray-900 tracking-tight">99%</div>
                  <div className="text-sm text-gray-500 tracking-tight mt-1">위험 탐지율</div>
                </div>
              </div>
            </div>

            {/* Right: 3D Device Mockup */}
            <div className="flex items-center justify-center ml-24 xl:ml-36">
              <DeviceMockup />
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Login Form */}
      <div className="flex-1 lg:flex-none lg:w-[460px] flex items-center justify-center p-6 sm:p-8 bg-white relative overflow-hidden">
        {/* Mobile blobs */}
        <div className="lg:hidden">
          <div className="liquid-blob liquid-blob-1" style={{ top: '-20%', right: '-20%', opacity: 0.3 }} />
          <div className="liquid-blob liquid-blob-2" style={{ bottom: '-10%', left: '-20%', opacity: 0.3 }} />
        </div>

        <div className="w-full max-w-[360px] relative z-10">
          {/* Mobile Logo */}
          <div className="text-center mb-10 lg:hidden animate-slide-in-bottom">
            <div className="flex justify-center mb-4">
              <Logo size={48} color="#111827" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
              DocScanner AI
            </h1>
            <p className="text-gray-500 mt-2 tracking-tight text-sm">
              계약서 AI 분석 서비스
            </p>
          </div>

          {/* Desktop Header */}
          <div className="hidden lg:block mb-8 animate-slide-in-bottom">
            <h2 className="text-2xl font-bold text-gray-900 tracking-tight">
              로그인
            </h2>
            <p className="text-gray-500 mt-2 tracking-tight">
              계정에 로그인하여 서비스를 이용하세요
            </p>
          </div>

          {/* Login Form */}
          <div className="liquid-glass-card p-7 animate-slide-in-bottom stagger-2">
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Error Message */}
              {error && (
                <div className="p-4 text-sm text-red-700 bg-red-50/80 border border-red-100 rounded-xl animate-fadeIn flex items-start gap-3">
                  <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-red-600 text-xs font-bold">!</span>
                  </div>
                  <span className="tracking-tight">{error}</span>
                </div>
              )}

              {/* Email Field */}
              <div className="space-y-2">
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 tracking-tight">
                  이메일
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full px-4 py-3.5 liquid-input text-gray-900 tracking-tight outline-none"
                  placeholder="name@example.com"
                  autoComplete="email"
                />
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 tracking-tight">
                  비밀번호
                </label>
                <div className="relative">
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="w-full px-4 py-3.5 pr-12 liquid-input text-gray-900 tracking-tight outline-none"
                    placeholder="비밀번호를 입력하세요"
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
                    tabIndex={-1}
                  >
                    {showPassword ? <IconEyeOff size={20} /> : <IconEye size={20} />}
                  </button>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full py-4 liquid-button flex items-center justify-center gap-2 tracking-tight text-[15px] group mt-2"
              >
                {loading ? (
                  <>
                    <IconLoading size={18} className="text-white/80" />
                    <span>로그인 중...</span>
                  </>
                ) : (
                  <>
                    <span>로그인</span>
                    <IconArrowRight size={18} className="transition-transform duration-300 group-hover:translate-x-1" />
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Footer Link */}
          <div className="text-center mt-8 animate-slide-in-bottom stagger-3">
            <p className="text-gray-500 tracking-tight">
              계정이 없으신가요?{" "}
              <Link
                href="/register"
                className="font-semibold text-gray-900 hover:text-gray-700 transition-colors"
              >
                회원가입
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
