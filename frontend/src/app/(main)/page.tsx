"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Contract, contractsApi, authApi, User } from "@/lib/api";
import {
  IconDocument,
  IconCheck,
  IconDanger,
  IconLoading,
  IconChevronRight,
  IconScan,
  IconChecklist,
} from "@/components/icons";
import { cn } from "@/lib/utils";
import {
  DonutChart,
  BarChart,
  RingProgress,
} from "@/components/charts";

function getRiskBadge(riskLevel: string | null) {
  if (!riskLevel) return null;

  const level = riskLevel.toLowerCase();
  if (level === "high" || level === "danger") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-red-50 text-red-600 border border-red-100">
        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
        High
      </span>
    );
  }
  if (level === "medium" || level === "warning") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 text-amber-600 border border-amber-100">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
        Medium
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-green-50 text-green-600 border border-green-100">
      <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
      Low
    </span>
  );
}

function getStatusBadge(status: string) {
  switch (status) {
    case "COMPLETED":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-green-50 text-green-600 border border-green-100">
          <IconCheck size={10} />
          완료
        </span>
      );
    case "PROCESSING":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-blue-50 text-blue-600 border border-blue-100">
          <IconLoading size={10} />
          분석중
        </span>
      );
    case "PENDING":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-500">
          대기중
        </span>
      );
    case "FAILED":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-red-50 text-red-600 border border-red-100">
          <IconDanger size={10} />
          실패
        </span>
      );
    default:
      return null;
  }
}

function formatRelativeTime(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "방금 전";
  if (diffMins < 60) return `${diffMins}분 전`;
  if (diffHours < 24) return `${diffHours}시간 전`;
  if (diffDays < 7) return `${diffDays}일 전`;
  return date.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

export default function DashboardPage() {
  const router = useRouter();
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    loadContracts();
    loadUser();
  }, []);

  const hasPendingOrProcessing = contracts.some(
    (c) => c.status === "PENDING" || c.status === "PROCESSING"
  );

  useEffect(() => {
    if (!hasPendingOrProcessing) return;

    const interval = setInterval(async () => {
      try {
        const data = await contractsApi.list();
        setContracts(data);
      } catch {
        // Silently fail
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [hasPendingOrProcessing]);

  async function loadContracts() {
    try {
      setLoading(true);
      const data = await contractsApi.list();
      setContracts(data);
    } catch {
      // Error handling
    } finally {
      setLoading(false);
    }
  }

  async function loadUser() {
    try {
      const userData = await authApi.getMe();
      setUser(userData);
    } catch {
      // Silently fail
    }
  }

  // Stats
  const stats = {
    total: contracts.length,
    completed: contracts.filter((c) => c.status === "COMPLETED").length,
    processing: contracts.filter((c) => c.status === "PROCESSING" || c.status === "PENDING").length,
    failed: contracts.filter((c) => c.status === "FAILED").length,
    highRisk: contracts.filter((c) => c.risk_level?.toLowerCase() === "high" || c.risk_level?.toLowerCase() === "danger").length,
    mediumRisk: contracts.filter((c) => c.risk_level?.toLowerCase() === "medium" || c.risk_level?.toLowerCase() === "warning").length,
    lowRisk: contracts.filter((c) => c.risk_level?.toLowerCase() === "low" || c.risk_level?.toLowerCase() === "safe").length,
  };

  // Completion rate
  const completionRate = stats.total > 0 ? Math.round((stats.completed / stats.total) * 100) : 0;

  // Recent contracts (last 5)
  const recentContracts = contracts.slice(0, 5);

  // Monthly data simulation (last 6 months)
  const monthlyData = [
    { label: "7월", value: Math.floor(stats.total * 0.6) },
    { label: "8월", value: Math.floor(stats.total * 0.75) },
    { label: "9월", value: Math.floor(stats.total * 0.5) },
    { label: "10월", value: Math.floor(stats.total * 0.9) },
    { label: "11월", value: Math.floor(stats.total * 0.85) },
    { label: "12월", value: stats.total, color: "default" as const },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3 animate-fadeIn">
          <IconLoading size={32} className="text-gray-400" />
          <p className="text-sm text-gray-500">불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-4">
      {/* Welcome & Overview Section */}
      <section>
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-6">
          <div>
            <p className="text-sm text-gray-500 mb-1">
              {new Date().toLocaleDateString("ko-KR", {
                year: "numeric",
                month: "long",
                day: "numeric",
                weekday: "long",
              })}
            </p>
            <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
              {user?.username ? `${user.username}님, 안녕하세요` : "안녕하세요"}
            </h1>
          </div>
        </div>

        {/* Primary Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Total */}
          <div className="liquid-glass-card p-4 sm:p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-gray-100 rounded-xl flex items-center justify-center">
                <IconDocument size={18} className="text-gray-600" />
              </div>
              <span className="text-xs text-gray-500 font-medium">전체</span>
            </div>
            <p className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
              {stats.total}
            </p>
            <p className="text-[11px] text-gray-400 mt-1">등록된 계약서</p>
          </div>

          {/* Completed */}
          <div className="liquid-glass-card p-4 sm:p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-green-50 rounded-xl flex items-center justify-center">
                <IconCheck size={18} className="text-green-600" />
              </div>
              <span className="text-xs text-green-600 font-medium">완료</span>
            </div>
            <p className="text-2xl sm:text-3xl font-bold text-green-700 tracking-tight">
              {stats.completed}
            </p>
            <p className="text-[11px] text-gray-400 mt-1">분석 완료</p>
          </div>

          {/* High Risk */}
          <div className="liquid-glass-card p-4 sm:p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-red-50 rounded-xl flex items-center justify-center">
                <IconDanger size={18} className="text-red-500" />
              </div>
              <span className="text-xs text-red-500 font-medium">위험</span>
            </div>
            <p className="text-2xl sm:text-3xl font-bold text-red-600 tracking-tight">
              {stats.highRisk}
            </p>
            <p className="text-[11px] text-gray-400 mt-1">주의 필요</p>
          </div>

          {/* Processing */}
          <div className="liquid-glass-card p-4 sm:p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 sm:w-10 sm:h-10 bg-blue-50 rounded-xl flex items-center justify-center">
                <IconLoading size={18} className="text-blue-500" />
              </div>
              <span className="text-xs text-blue-500 font-medium">진행중</span>
            </div>
            <p className="text-2xl sm:text-3xl font-bold text-blue-600 tracking-tight">
              {stats.processing}
            </p>
            <p className="text-[11px] text-gray-400 mt-1">분석 진행중</p>
          </div>
        </div>
      </section>

      {/* Analytics Section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Risk Distribution Chart */}
        <div className="liquid-glass-card p-5 sm:p-6">
          <h3 className="text-sm font-semibold text-gray-900 tracking-tight mb-4">
            위험도 분포
          </h3>
          <div className="flex items-center justify-center gap-6">
            <DonutChart
              segments={[
                { value: stats.lowRisk || 1, color: "#22c55e", label: "Low" },
                { value: stats.mediumRisk, color: "#f59e0b", label: "Medium" },
                { value: stats.highRisk, color: "#ef4444", label: "High" },
              ]}
              size={140}
              strokeWidth={16}
              centerValue={stats.completed}
              centerLabel="분석 완료"
            />
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 tracking-tight">{stats.lowRisk}</p>
                  <p className="text-[11px] text-gray-500">Low Risk</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.4)]" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 tracking-tight">{stats.mediumRisk}</p>
                  <p className="text-[11px] text-gray-500">Medium Risk</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]" />
                <div>
                  <p className="text-sm font-semibold text-gray-900 tracking-tight">{stats.highRisk}</p>
                  <p className="text-[11px] text-gray-500">High Risk</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Monthly Analysis Chart */}
        <div className="liquid-glass-card p-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-900 tracking-tight">
              월별 분석 현황
            </h3>
            <span className="text-[11px] text-gray-400 font-medium">최근 6개월</span>
          </div>
          <BarChart
            data={monthlyData}
            height={100}
            showLabels={true}
            showValues={true}
          />
        </div>
      </section>

      {/* Progress Overview */}
      <section className="liquid-glass-card p-5 sm:p-6">
        <h3 className="text-sm font-semibold text-gray-900 tracking-tight mb-4">
          분석 진행률
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {/* Completion Rate */}
          <div className="flex items-center gap-4">
            <RingProgress
              value={completionRate}
              size={64}
              strokeWidth={6}
              color="#22c55e"
              showPercentage={true}
            />
            <div>
              <p className="text-sm font-semibold text-gray-900 tracking-tight">완료율</p>
              <p className="text-xs text-gray-500">{stats.completed}/{stats.total} 완료</p>
            </div>
          </div>

          {/* Processing Rate */}
          <div className="flex items-center gap-4">
            <RingProgress
              value={stats.total > 0 ? (stats.processing / stats.total) * 100 : 0}
              size={64}
              strokeWidth={6}
              color="#3b82f6"
              showPercentage={true}
            />
            <div>
              <p className="text-sm font-semibold text-gray-900 tracking-tight">진행중</p>
              <p className="text-xs text-gray-500">{stats.processing}건 분석중</p>
            </div>
          </div>

          {/* Risk Rate */}
          <div className="flex items-center gap-4">
            <RingProgress
              value={stats.completed > 0 ? (stats.highRisk / stats.completed) * 100 : 0}
              size={64}
              strokeWidth={6}
              color="#ef4444"
              showPercentage={true}
            />
            <div>
              <p className="text-sm font-semibold text-gray-900 tracking-tight">고위험 비율</p>
              <p className="text-xs text-gray-500">{stats.highRisk}건 주의 필요</p>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Actions & Recent Contracts */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Quick Actions */}
        <div className="lg:col-span-1 space-y-3">
          <h3 className="text-sm font-semibold text-gray-900 tracking-tight px-1">
            빠른 시작
          </h3>
          <Link
            href="/scan"
            className="liquid-glass-card p-4 flex items-center gap-4 hover:scale-[1.02] active:scale-[0.98] transition-all group"
          >
            <div className="w-11 h-11 bg-gray-900 rounded-xl flex items-center justify-center group-hover:scale-105 transition-transform shadow-md">
              <IconScan size={20} className="text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-gray-900 tracking-tight">빠른 스캔</p>
              <p className="text-xs text-gray-500">카메라로 계약서 촬영</p>
            </div>
            <IconChevronRight size={16} className="text-gray-400 group-hover:translate-x-1 transition-transform" />
          </Link>

          <Link
            href="/checklist"
            className="liquid-glass-card p-4 flex items-center gap-4 hover:scale-[1.02] active:scale-[0.98] transition-all group"
          >
            <div className="w-11 h-11 bg-green-600 rounded-xl flex items-center justify-center group-hover:scale-105 transition-transform shadow-md shadow-green-200">
              <IconChecklist size={20} className="text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-gray-900 tracking-tight">체크리스트</p>
              <p className="text-xs text-gray-500">계약서 검토 가이드</p>
            </div>
            <IconChevronRight size={16} className="text-gray-400 group-hover:translate-x-1 transition-transform" />
          </Link>

          <div className="liquid-glass-card p-4 flex items-center gap-4 opacity-80">
            <div className="w-11 h-11 bg-gray-100 rounded-xl flex items-center justify-center">
              <IconDocument size={20} className="text-gray-500" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-gray-900 tracking-tight">새 계약서 분석</p>
              <p className="text-xs text-gray-500">하단 버튼을 눌러주세요</p>
            </div>
          </div>
        </div>

        {/* Recent Contracts */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-3 px-1">
            <h3 className="text-sm font-semibold text-gray-900 tracking-tight">
              최근 계약서
            </h3>
            <Link
              href="/history"
              className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-0.5 transition-colors"
            >
              전체 보기
              <IconChevronRight size={14} />
            </Link>
          </div>

          {recentContracts.length === 0 ? (
            <div className="liquid-glass-card p-8 text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-gray-100 rounded-xl mb-3">
                <IconDocument size={24} className="text-gray-400" />
              </div>
              <p className="text-sm text-gray-700 mb-1 font-medium tracking-tight">
                아직 분석된 계약서가 없습니다
              </p>
              <p className="text-xs text-gray-500">
                첫 번째 계약서를 업로드해보세요
              </p>
            </div>
          ) : (
            <div className="liquid-glass-card divide-y divide-gray-100/50 overflow-hidden">
              {recentContracts.map((contract) => (
                <div
                  key={contract.id}
                  className={cn(
                    "p-4 flex items-center gap-4 transition-colors cursor-pointer hover:bg-white/30",
                    contract.status !== "COMPLETED" && "opacity-70"
                  )}
                  onClick={() => {
                    if (contract.status === "COMPLETED") {
                      router.push(`/analysis/${contract.id}`);
                    }
                  }}
                >
                  <div className="w-9 h-9 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <IconDocument size={16} className="text-gray-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate tracking-tight">
                      {contract.title}
                    </p>
                    <p className="text-[11px] text-gray-400 mt-0.5">
                      {formatRelativeTime(contract.created_at)}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {getStatusBadge(contract.status)}
                    {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                    {contract.status === "COMPLETED" && (
                      <IconChevronRight size={14} className="text-gray-300" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
