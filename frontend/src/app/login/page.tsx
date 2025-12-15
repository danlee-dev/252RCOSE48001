"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { IconLoading, IconEye, IconEyeOff, IconArrowRight, Logo } from "@/components/icons";

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
    <div className="min-h-[100dvh] flex items-center justify-center bg-gray-50 px-4 py-8 overflow-hidden relative">
      {/* Animated Background Shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="auth-shape animate-float-slow animate-morph-blob w-[500px] h-[500px] bg-gray-200/60 -top-48 -left-48"
          style={{ position: 'absolute' }}
        />
        <div
          className="auth-shape animate-float-medium animate-morph-blob w-[400px] h-[400px] bg-gray-300/40 -bottom-32 -right-32"
          style={{ position: 'absolute', animationDelay: '-5s' }}
        />
        <div
          className="auth-shape animate-float-slow w-[200px] h-[200px] bg-gray-200/50 top-1/4 right-1/4"
          style={{ position: 'absolute', animationDelay: '-10s' }}
        />
      </div>

      <div className="w-full max-w-[420px] relative z-10">
        {/* Logo & Header */}
        <div className="text-center mb-10 animate-slide-in-bottom">
          <div className="flex justify-center mb-5">
            <div className="relative group">
              <div className="absolute inset-0 bg-gray-900/5 rounded-2xl blur-xl group-hover:bg-gray-900/10 transition-all duration-500" />
              <Logo size={64} color="#111827" className="relative transition-transform duration-300 group-hover:scale-105" />
            </div>
          </div>
          <h1 className="text-[1.75rem] font-bold text-gray-900 tracking-tight">
            DocScanner AI
          </h1>
          <p className="text-gray-500 mt-2 tracking-tight">
            계약서 AI 분석 서비스에 로그인하세요
          </p>
        </div>

        {/* Login Card */}
        <div className="auth-card p-7 sm:p-8 animate-slide-in-bottom stagger-2">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Error Message */}
            {error && (
              <div className="p-4 text-sm text-red-700 bg-red-50 border border-red-100 rounded-xl animate-fadeIn flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs font-bold">!</span>
                </div>
                <span>{error}</span>
              </div>
            )}

            {/* Email Field */}
            <div className="floating-input-wrapper">
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="floating-input tracking-tight pr-4"
                placeholder=" "
                autoComplete="email"
              />
              <label htmlFor="email" className="floating-label tracking-tight">
                이메일
              </label>
              <div className="input-focus-line" />
            </div>

            {/* Password Field */}
            <div className="floating-input-wrapper">
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="floating-input tracking-tight pr-12"
                placeholder=" "
                autoComplete="current-password"
              />
              <label htmlFor="password" className="floating-label tracking-tight">
                비밀번호
              </label>
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="password-toggle"
                tabIndex={-1}
              >
                {showPassword ? <IconEyeOff size={20} /> : <IconEye size={20} />}
              </button>
              <div className="input-focus-line" />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="auth-button flex items-center justify-center gap-2 tracking-tight"
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
              className="font-semibold text-gray-900 hover:text-gray-700 transition-colors relative inline-block group"
            >
              회원가입
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-900 transform scale-x-0 transition-transform duration-300 origin-left group-hover:scale-x-100" />
            </Link>
          </p>
        </div>

        {/* Visual Accent Line */}
        <div className="mt-12 flex items-center justify-center gap-3 animate-slide-in-bottom stagger-4">
          <div className="h-px w-12 bg-gray-200" />
          <span className="text-xs text-gray-400 tracking-tight">Secure Login</span>
          <div className="h-px w-12 bg-gray-200" />
        </div>
      </div>
    </div>
  );
}
