"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Contract, contractsApi, authApi, User } from "@/lib/api";
import { IconClose } from "@/components/icons";
import {
  IconDocument,
  IconCheck,
  IconDanger,
  IconLoading,
  IconChevronRight,
  IconScan,
  IconChecklist,
  IconSearch,
  IconBell,
} from "@/components/icons";
import { cn } from "@/lib/utils";
import {
  DonutChart,
  BarChart,
  RingProgress,
  Sparkline,
} from "@/components/charts";

function getRiskBadge(riskLevel: string | null) {
  if (!riskLevel) return null;

  const level = riskLevel.toLowerCase();
  if (level === "high" || level === "danger") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#fdedec] text-[#b54a45] border border-[#f5c6c4]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#c94b45]" />
        High
      </span>
    );
  }
  if (level === "medium" || level === "warning") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#d4a84d]" />
        Medium
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]">
      <span className="w-1.5 h-1.5 rounded-full bg-[#4a9a5b]" />
      Low
    </span>
  );
}

function getStatusBadge(status: string) {
  switch (status) {
    case "COMPLETED":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]">
          <IconCheck size={10} />
          완료
        </span>
      );
    case "PROCESSING":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#e8f0ea] text-[#3d5a47] border border-[#c8e6cf]">
          <IconLoading size={10} />
          분석중
        </span>
      );
    case "PENDING":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#fef7e0] text-[#9a7b2d] border border-[#f5e6b8]">
          대기중
        </span>
      );
    case "FAILED":
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#fdedec] text-[#b54a45] border border-[#f5c6c4]">
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
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [showNotificationDropdown, setShowNotificationDropdown] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const notificationRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadContracts();
    loadUser();
  }, []);

  // Click outside to close dropdowns
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowSearchDropdown(false);
      }
      if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
        setShowNotificationDropdown(false);
      }
    }
    if (showSearchDropdown || showNotificationDropdown) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showSearchDropdown, showNotificationDropdown]);

  // Filter contracts by search query
  const filteredContracts = contracts.filter((contract) =>
    contract.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Mock notifications
  const notifications = [
    { id: "1", type: "complete", title: "분석 완료", message: "근로계약서_2024.pdf 분석이 완료되었습니다.", time: "5분 전", unread: true },
    { id: "2", type: "complete", title: "분석 완료", message: "임대차계약서.pdf 분석이 완료되었습니다.", time: "30분 전", unread: true },
    { id: "3", type: "system", title: "서비스 업데이트", message: "새로운 AI 분석 기능이 추가되었습니다.", time: "1일 전", unread: false },
  ];
  const unreadCount = notifications.filter((n) => n.unread).length;

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

  // Simulated sparkline data
  const totalSparkline = [2, 4, 3, 5, 4, 6, 5, 7, 6, stats.total || 1];
  const completedSparkline = [1, 2, 2, 3, 3, 4, 4, 5, 5, stats.completed || 0];
  const riskSparkline = [0, 1, 1, 1, 2, 1, 2, 2, 1, stats.highRisk || 0];

  // Monthly data - 실제 계약서 created_at 기반 집계
  const monthlyData = (() => {
    const now = new Date();
    const months: { label: string; value: number; color?: "default" }[] = [];

    // 최근 6개월 데이터 생성
    for (let i = 5; i >= 0; i--) {
      const targetDate = new Date(now.getFullYear(), now.getMonth() - i, 1);
      const monthLabel = `${targetDate.getMonth() + 1}월`;

      // 해당 월의 계약서 수 계산
      const count = contracts.filter((c) => {
        const createdAt = new Date(c.created_at);
        return (
          createdAt.getFullYear() === targetDate.getFullYear() &&
          createdAt.getMonth() === targetDate.getMonth()
        );
      }).length;

      months.push({
        label: monthLabel,
        value: count,
        ...(i === 0 ? { color: "default" as const } : {}),
      });
    }

    return months;
  })();

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
    <div className="space-y-5 pb-4">
      {/* Welcome Section */}
      <section className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#3d5a47] tracking-tight">
            {user?.username ? `${user.username}님, 안녕하세요` : "안녕하세요"}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            계약서 정보와 현황을 확인하세요
          </p>
        </div>

        {/* Search & Notification */}
        <div className="flex items-center gap-3">
          {/* Search Bar with Dropdown */}
          <div ref={searchRef} className="relative">
            <div className="relative flex items-center">
              <input
                ref={searchInputRef}
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setShowSearchDropdown(true)}
                className="w-56 sm:w-80 h-12 pl-5 pr-14 text-sm bg-white border border-gray-200 rounded-full outline-none focus:border-gray-400 transition-colors"
              />
              <button
                onClick={() => {
                  setShowSearchDropdown(true);
                  searchInputRef.current?.focus();
                }}
                className="absolute right-1.5 w-9 h-9 bg-gray-900 rounded-full flex items-center justify-center hover:bg-gray-800 transition-colors"
              >
                <IconSearch size={16} className="text-white" />
              </button>
            </div>

            {/* Search Dropdown */}
            {showSearchDropdown && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden z-50 animate-fadeIn">
                <div className="max-h-72 overflow-y-auto">
                  {searchQuery === "" ? (
                    <div className="py-8 text-center">
                      <IconSearch size={24} className="mx-auto text-gray-300 mb-2" />
                      <p className="text-sm text-gray-500">검색어를 입력하세요</p>
                    </div>
                  ) : filteredContracts.length === 0 ? (
                    <div className="py-8 text-center">
                      <IconDocument size={24} className="mx-auto text-gray-300 mb-2" />
                      <p className="text-sm text-gray-500">검색 결과가 없습니다</p>
                    </div>
                  ) : (
                    <div className="divide-y divide-gray-100">
                      {filteredContracts.slice(0, 5).map((contract) => (
                        <button
                          key={contract.id}
                          onClick={() => {
                            if (contract.status === "COMPLETED") {
                              router.push(`/analysis/${contract.id}`);
                            }
                            setShowSearchDropdown(false);
                            setSearchQuery("");
                          }}
                          className="w-full p-3 text-left hover:bg-gray-50 transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-[8px] bg-[#e8f0ea] flex items-center justify-center text-[#3d5a47] flex-shrink-0">
                              <IconDocument size={14} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-gray-900 truncate tracking-tight">{contract.title}</p>
                              <p className="text-[11px] text-gray-500">
                                {new Date(contract.created_at).toLocaleDateString("ko-KR")}
                              </p>
                            </div>
                            {contract.status === "COMPLETED" && (
                              <span className="text-[10px] text-[#3d7a4a] bg-[#e8f5ec] px-1.5 py-0.5 rounded-[4px]">완료</span>
                            )}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Notification Bell with Dropdown */}
          <div ref={notificationRef} className="relative">
            <button
              onClick={() => setShowNotificationDropdown(!showNotificationDropdown)}
              className="relative w-12 h-12 bg-white border border-gray-200 rounded-full flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <IconBell size={20} className="text-gray-600" />
              {unreadCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-5 h-5 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center border-2 border-white">
                  {unreadCount}
                </span>
              )}
            </button>

            {/* Notification Dropdown */}
            {showNotificationDropdown && (
              <div className="absolute top-full right-0 mt-2 w-80 bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden z-50 animate-fadeIn">
                <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-gray-900 tracking-tight">알림</h3>
                    {unreadCount > 0 && (
                      <span className="px-1.5 py-0.5 text-[10px] font-medium bg-red-500 text-white rounded-full">
                        {unreadCount}
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => setShowNotificationDropdown(false)}
                    className="p-1 text-gray-400 hover:text-gray-600 rounded-lg transition-colors"
                  >
                    <IconClose size={16} />
                  </button>
                </div>
                <div className="max-h-72 overflow-y-auto">
                  {notifications.length === 0 ? (
                    <div className="py-8 text-center">
                      <IconBell size={24} className="mx-auto text-gray-300 mb-2" />
                      <p className="text-sm text-gray-500">알림이 없습니다</p>
                    </div>
                  ) : (
                    <div className="divide-y divide-gray-100">
                      {notifications.map((notification) => (
                        <div
                          key={notification.id}
                          className={cn(
                            "p-3 hover:bg-gray-50 transition-colors cursor-pointer",
                            notification.unread && "bg-blue-50/50"
                          )}
                        >
                          <div className="flex items-start gap-3">
                            <div className={cn(
                              "w-8 h-8 rounded-[8px] flex items-center justify-center flex-shrink-0",
                              notification.type === "complete" ? "bg-green-100 text-green-600" : "bg-blue-100 text-blue-600"
                            )}>
                              {notification.type === "complete" ? <IconCheck size={14} /> : <IconBell size={14} />}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <p className="text-sm font-medium text-gray-900 tracking-tight">{notification.title}</p>
                                {notification.unread && (
                                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full flex-shrink-0" />
                                )}
                              </div>
                              <p className="text-[11px] text-gray-500 mt-0.5 line-clamp-1">{notification.message}</p>
                              <p className="text-[10px] text-gray-400 mt-1">{notification.time}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Stats Grid with Sparklines */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Total - White card */}
        <div className="card-apple p-4 sm:p-5">
          <div className="flex items-start justify-between mb-2">
            <div>
              <p className="text-xs font-medium text-gray-500 mb-1">전체 계약서</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
                {stats.total}
              </p>
            </div>
            <Sparkline data={totalSparkline} width={56} height={28} color="#6b7280" />
          </div>
          <p className="text-[11px] text-gray-400">이번 달 등록</p>
        </div>

        {/* Completed - White card */}
        <div className="card-apple p-4 sm:p-5">
          <div className="flex items-start justify-between mb-2">
            <div>
              <p className="text-xs font-medium text-[#3d7a4a] mb-1">분석 완료</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
                {stats.completed}
              </p>
            </div>
            <Sparkline data={completedSparkline} width={56} height={28} color="#4a9a5b" />
          </div>
          <p className="text-[11px] text-[#3d7a4a]/70">+{Math.max(0, stats.completed - 2)} 이번 주</p>
        </div>

        {/* High Risk - White card */}
        <div className="card-apple p-4 sm:p-5">
          <div className="flex items-start justify-between mb-2">
            <div>
              <p className="text-xs font-medium text-[#b54a45] mb-1">주의 필요</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
                {stats.highRisk}
              </p>
            </div>
            <Sparkline data={riskSparkline} width={56} height={28} color="#c94b45" />
          </div>
          <p className="text-[11px] text-[#b54a45]/70">고위험 계약서</p>
        </div>

        {/* Processing - Accent colored card (dark green) */}
        <div className="bg-[#3d5a47] rounded-[18px] p-4 sm:p-5 shadow-sm relative overflow-hidden">
          <div className="relative z-10">
            <p className="text-xs font-medium text-white/70 mb-1">진행률</p>
            <p className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              {completionRate}%
            </p>
            <p className="text-[11px] text-white/60 mt-2">{stats.completed}/{stats.total} 완료</p>
          </div>
          {/* Decorative curve */}
          <svg
            className="absolute right-0 bottom-0 w-24 h-24 text-white/10"
            viewBox="0 0 100 100"
            fill="currentColor"
          >
            <path d="M100 100 C100 44.8 55.2 0 0 0 L0 100 Z" />
          </svg>
        </div>
      </section>

      {/* Main Content Grid */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-4">
          {/* Monthly Analysis & Risk Distribution */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Monthly Analysis Chart */}
            <div className="card-apple p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-900 tracking-tight">
                  월별 현황
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

            {/* Risk Distribution Chart */}
            <div className="card-apple p-5">
              <h3 className="text-sm font-semibold text-gray-900 tracking-tight mb-4">
                위험도 분포
              </h3>
              <div className="flex items-center justify-center gap-5">
                <DonutChart
                  segments={[
                    { value: stats.lowRisk || 1, color: "#4a9a5b", label: "Low" },
                    { value: stats.mediumRisk, color: "#d4a84d", label: "Medium" },
                    { value: stats.highRisk, color: "#c94b45", label: "High" },
                  ]}
                  size={100}
                  strokeWidth={12}
                  centerValue={stats.completed}
                  centerLabel="완료"
                />
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-[#4a9a5b]" />
                    <span className="text-xs text-gray-600">Low</span>
                    <span className="text-xs font-semibold text-gray-900 ml-auto">{stats.lowRisk}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-[#d4a84d]" />
                    <span className="text-xs text-gray-600">Medium</span>
                    <span className="text-xs font-semibold text-gray-900 ml-auto">{stats.mediumRisk}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-[#c94b45]" />
                    <span className="text-xs text-gray-600">High</span>
                    <span className="text-xs font-semibold text-gray-900 ml-auto">{stats.highRisk}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Contracts */}
          <div className="card-apple overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
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
              <div className="p-8 text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-[#e8f0ea] rounded-xl mb-3">
                  <IconDocument size={24} className="text-[#3d5a47]" />
                </div>
                <p className="text-sm text-gray-700 mb-1 font-medium tracking-tight">
                  아직 분석된 계약서가 없습니다
                </p>
                <p className="text-xs text-gray-500">
                  첫 번째 계약서를 업로드해보세요
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-50">
                {recentContracts.map((contract) => (
                  <div
                    key={contract.id}
                    className={cn(
                      "px-5 py-3.5 flex items-center gap-4 transition-colors cursor-pointer hover:bg-gray-50/50",
                      contract.status !== "COMPLETED" && "opacity-70"
                    )}
                    onClick={() => {
                      if (contract.status === "COMPLETED") {
                        router.push(`/analysis/${contract.id}`);
                      }
                    }}
                  >
                    <div className="w-9 h-9 bg-[#e8f0ea] rounded-[10px] flex items-center justify-center flex-shrink-0">
                      <IconDocument size={16} className="text-[#3d5a47]" />
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
        </div>

        {/* Right Column - Quick Actions & Progress */}
        <div className="space-y-4">
          {/* Progress Card */}
          <div className="card-apple p-5">
            <h3 className="text-sm font-semibold text-gray-900 tracking-tight mb-4">
              분석 현황
            </h3>
            <div className="flex items-center justify-center mb-4">
              <RingProgress
                value={completionRate}
                size={120}
                strokeWidth={10}
                color="#4a9a5b"
                showPercentage={true}
              />
            </div>
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-lg font-bold text-[#3d7a4a]">{stats.completed}</p>
                <p className="text-[10px] text-gray-500">완료</p>
              </div>
              <div>
                <p className="text-lg font-bold text-[#3d5a47]">{stats.processing}</p>
                <p className="text-[10px] text-gray-500">진행중</p>
              </div>
              <div>
                <p className="text-lg font-bold text-[#c94b45]">{stats.highRisk}</p>
                <p className="text-[10px] text-gray-500">위험</p>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card-apple p-5">
            <h3 className="text-sm font-semibold text-gray-900 tracking-tight mb-4">
              빠른 시작
            </h3>
            <div className="space-y-2.5">
              <Link
                href="/scan"
                className="flex items-center gap-3 p-3 rounded-xl bg-[#e8f0ea] hover:bg-[#dce8de] transition-colors group"
              >
                <div className="w-10 h-10 bg-[#3d5a47] rounded-xl flex items-center justify-center shadow-sm group-hover:scale-105 transition-transform">
                  <IconScan size={18} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 tracking-tight">빠른 스캔</p>
                  <p className="text-[11px] text-gray-500">카메라로 촬영</p>
                </div>
                <IconChevronRight size={16} className="text-[#3d5a47]" />
              </Link>

              <Link
                href="/checklist"
                className="flex items-center gap-3 p-3 rounded-xl bg-[#e8f0ea] hover:bg-[#dce8de] transition-colors group"
              >
                <div className="w-10 h-10 bg-[#4a9a5b] rounded-xl flex items-center justify-center shadow-sm shadow-[#c8e6cf] group-hover:scale-105 transition-transform">
                  <IconChecklist size={18} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 tracking-tight">체크리스트</p>
                  <p className="text-[11px] text-gray-500">검토 가이드</p>
                </div>
                <IconChevronRight size={16} className="text-[#3d5a47]" />
              </Link>

              <button
                onClick={() => window.dispatchEvent(new CustomEvent("openUploadSidebar"))}
                className="flex items-center gap-3 p-3 rounded-xl bg-[#e8f0ea] hover:bg-[#dce8de] transition-colors group w-full text-left"
              >
                <div className="w-10 h-10 bg-[#3d5a47]/80 rounded-xl flex items-center justify-center shadow-sm group-hover:scale-105 transition-transform">
                  <IconDocument size={18} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 tracking-tight">새 계약서 분석</p>
                  <p className="text-[11px] text-gray-500">파일 업로드</p>
                </div>
                <IconChevronRight size={16} className="text-[#3d5a47]" />
              </button>
            </div>
          </div>

          {/* Safety Tip Card */}
          <div className="bg-[#fef7e0] rounded-[18px] p-5 border border-[#f5e6b8]">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 bg-[#f5e6b8] rounded-xl flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-[#9a7b2d]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-[#7a6223] tracking-tight mb-1">계약서 검토 팁</p>
                <p className="text-xs text-[#9a7b2d]/90 leading-relaxed">
                  계약 전 반드시 모든 조항을 꼼꼼히 읽고, 위약금 조항과 해지 조건을 확인하세요.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
