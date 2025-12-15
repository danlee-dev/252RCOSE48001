"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { IconLoading, IconEye, IconEyeOff, IconArrowRight, IconCheck, Logo } from "@/components/icons";

export default function RegisterPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showPasswordConfirm, setShowPasswordConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Password strength indicator
  const getPasswordStrength = (pwd: string) => {
    if (pwd.length === 0) return { level: 0, text: "", color: "" };
    if (pwd.length < 4) return { level: 1, text: "너무 짧음", color: "bg-red-400" };
    if (pwd.length < 6) return { level: 2, text: "약함", color: "bg-orange-400" };
    if (pwd.length < 8) return { level: 3, text: "보통", color: "bg-yellow-400" };
    return { level: 4, text: "강함", color: "bg-green-500" };
  };

  const passwordStrength = getPasswordStrength(password);
  const passwordsMatch = password === passwordConfirm;
  const isFormValid = password.length >= 4 && passwordsMatch && passwordConfirm.length > 0;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (password !== passwordConfirm) {
      setError("비밀번호가 일치하지 않습니다");
      return;
    }

    setLoading(true);

    try {
      await authApi.register(email, password, name);
      await authApi.login(email, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "회원가입에 실패했습니다");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-[100dvh] flex items-center justify-center bg-gray-50 px-4 py-8 overflow-hidden relative">
      {/* Animated Background Shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="auth-shape animate-float-slow animate-morph-blob w-[500px] h-[500px] bg-gray-200/60 -top-48 -right-48"
          style={{ position: 'absolute' }}
        />
        <div
          className="auth-shape animate-float-medium animate-morph-blob w-[400px] h-[400px] bg-gray-300/40 -bottom-32 -left-32"
          style={{ position: 'absolute', animationDelay: '-5s' }}
        />
        <div
          className="auth-shape animate-float-slow w-[200px] h-[200px] bg-gray-200/50 bottom-1/4 right-1/4"
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
            새 계정을 만들어 시작하세요
          </p>
        </div>

        {/* Register Card */}
        <div className="auth-card p-7 sm:p-8 animate-slide-in-bottom stagger-2">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Error Message */}
            {error && (
              <div className="p-4 text-sm text-red-700 bg-red-50 border border-red-100 rounded-xl animate-fadeIn flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs font-bold">!</span>
                </div>
                <span>{error}</span>
              </div>
            )}

            {/* Name Field */}
            <div className="floating-input-wrapper">
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="floating-input tracking-tight pr-4"
                placeholder=" "
                autoComplete="name"
              />
              <label htmlFor="name" className="floating-label tracking-tight">
                이름
              </label>
              <div className="input-focus-line" />
            </div>

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
            <div className="space-y-2">
              <div className="floating-input-wrapper">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={4}
                  className="floating-input tracking-tight pr-12"
                  placeholder=" "
                  autoComplete="new-password"
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

              {/* Password Strength Indicator */}
              {password.length > 0 && (
                <div className="animate-fadeIn">
                  <div className="flex gap-1 h-1">
                    {[1, 2, 3, 4].map((level) => (
                      <div
                        key={level}
                        className={`flex-1 rounded-full transition-all duration-300 ${
                          level <= passwordStrength.level
                            ? passwordStrength.color
                            : "bg-gray-200"
                        }`}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-1.5 tracking-tight flex items-center gap-1">
                    {passwordStrength.level >= 3 && (
                      <IconCheck size={12} className="text-green-500" />
                    )}
                    {passwordStrength.text}
                  </p>
                </div>
              )}
            </div>

            {/* Password Confirm Field */}
            <div className="space-y-2">
              <div className="floating-input-wrapper">
                <input
                  id="passwordConfirm"
                  type={showPasswordConfirm ? "text" : "password"}
                  value={passwordConfirm}
                  onChange={(e) => setPasswordConfirm(e.target.value)}
                  required
                  className="floating-input tracking-tight pr-12"
                  placeholder=" "
                  autoComplete="new-password"
                />
                <label htmlFor="passwordConfirm" className="floating-label tracking-tight">
                  비밀번호 확인
                </label>
                <button
                  type="button"
                  onClick={() => setShowPasswordConfirm(!showPasswordConfirm)}
                  className="password-toggle"
                  tabIndex={-1}
                >
                  {showPasswordConfirm ? <IconEyeOff size={20} /> : <IconEye size={20} />}
                </button>
                <div className="input-focus-line" />
              </div>

              {/* Password Match Indicator */}
              {passwordConfirm.length > 0 && (
                <div className="animate-fadeIn">
                  <p className={`text-xs tracking-tight flex items-center gap-1 ${
                    passwordsMatch ? "text-green-600" : "text-red-500"
                  }`}>
                    {passwordsMatch ? (
                      <>
                        <IconCheck size={12} />
                        비밀번호가 일치합니다
                      </>
                    ) : (
                      <>
                        <span className="w-3 h-3 flex items-center justify-center text-[10px] font-bold">!</span>
                        비밀번호가 일치하지 않습니다
                      </>
                    )}
                  </p>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !isFormValid}
              className="auth-button flex items-center justify-center gap-2 tracking-tight mt-6"
            >
              {loading ? (
                <>
                  <IconLoading size={18} className="text-white/80" />
                  <span>계정 생성 중...</span>
                </>
              ) : (
                <>
                  <span>회원가입</span>
                  <IconArrowRight size={18} className="transition-transform duration-300 group-hover:translate-x-1" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Footer Link */}
        <div className="text-center mt-8 animate-slide-in-bottom stagger-3">
          <p className="text-gray-500 tracking-tight">
            이미 계정이 있으신가요?{" "}
            <Link
              href="/login"
              className="font-semibold text-gray-900 hover:text-gray-700 transition-colors relative inline-block group"
            >
              로그인
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-900 transform scale-x-0 transition-transform duration-300 origin-left group-hover:scale-x-100" />
            </Link>
          </p>
        </div>

        {/* Visual Accent Line */}
        <div className="mt-12 flex items-center justify-center gap-3 animate-slide-in-bottom stagger-4">
          <div className="h-px w-12 bg-gray-200" />
          <span className="text-xs text-gray-400 tracking-tight">Secure Registration</span>
          <div className="h-px w-12 bg-gray-200" />
        </div>
      </div>
    </div>
  );
}
