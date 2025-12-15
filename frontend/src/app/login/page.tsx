"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { IconLoading, Logo } from "@/components/icons";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
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
    <div className="min-h-[100dvh] flex items-center justify-center bg-gradient-to-br from-gray-50 via-white to-gray-50 px-4 py-8">
      <div className="w-full max-w-sm animate-fadeInUp">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="animate-float drop-shadow-lg">
              <Logo size={56} color="#111827" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">DocScanner AI</h1>
          <p className="text-sm text-gray-500 mt-2 tracking-tight">계약서 AI 분석 서비스에 로그인하세요</p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-soft border border-gray-100 p-5 sm:p-6">
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="p-3 text-sm text-red-700 bg-red-50 border border-red-100 rounded-xl animate-fadeIn">
                {error}
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1.5 tracking-tight">
                이메일
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="input-field text-sm h-11"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1.5 tracking-tight">
                비밀번호
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="input-field text-sm h-11"
                placeholder="비밀번호 입력"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 text-sm font-medium text-white bg-gray-900 rounded-xl shadow-sm hover:bg-gray-800 hover:shadow-md active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-sm flex items-center justify-center gap-2 transition-all duration-200 min-h-[48px]"
            >
              {loading && <IconLoading size={16} />}
              로그인
            </button>
          </form>
        </div>

        {/* Footer Link */}
        <p className="text-center text-sm text-gray-500 mt-6 tracking-tight">
          계정이 없으신가요?{" "}
          <Link href="/register" className="font-medium text-gray-900 hover:text-gray-700 transition-colors py-1">
            회원가입
          </Link>
        </p>
      </div>
    </div>
  );
}
