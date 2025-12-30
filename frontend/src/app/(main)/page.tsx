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
  IconWarning,
  IconLoading,
  IconChevronRight,
  IconScan,
  IconChecklist,
  IconSearch,
  IconBell,
  IconHome,
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

  // Filter contracts by search query (supports Korean)
  const filteredContracts = contracts.filter((contract) => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return true;
    const title = contract.title.toLowerCase();
    // 정규화된 문자열 비교 (NFD/NFC 호환)
    const normalizedTitle = title.normalize("NFC");
    const normalizedQuery = query.normalize("NFC");
    return normalizedTitle.includes(normalizedQuery);
  });

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
        const data = await contractsApi.list(0, 100);
        setContracts(data.items);
      } catch {
        // Silently fail
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [hasPendingOrProcessing]);

  async function loadContracts() {
    try {
      setLoading(true);
      const data = await contractsApi.list(0, 100);
      setContracts(data.items);
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
    <div className="space-y-10 pb-10">
      {/* Header Section */}
      <section className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="relative">
          <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-[#3d5a47]/10 to-transparent rounded-full blur-2xl" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                <IconHome size={20} className="text-white" />
              </div>
              <span className="text-sm font-medium text-[#3d5a47] tracking-tight">Dashboard</span>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
              {user?.username ? `${user.username}님, 안녕하세요` : "안녕하세요"}
            </h1>
          </div>
        </div>

        {/* Search & Notification */}
        <div className="flex items-center gap-4">
          {/* Search Bar with Dropdown */}
          <div ref={searchRef} className="relative">
            <div className="relative flex items-center liquid-input-wrapper">
              <input
                ref={searchInputRef}
                type="text"
                placeholder="계약서 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setShowSearchDropdown(true)}
                className="w-64 sm:w-96 h-14 pl-6 pr-16 text-base liquid-input rounded-full outline-none transition-all"
              />
              <button
                onClick={() => {
                  setShowSearchDropdown(true);
                  searchInputRef.current?.focus();
                }}
                className="absolute right-2 w-10 h-10 bg-gray-900 rounded-full flex items-center justify-center hover:bg-gray-800 transition-colors"
              >
                <IconSearch size={18} className="text-white" />
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
              className="relative w-14 h-14 bg-white border border-gray-200 rounded-full flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <IconBell size={22} className="text-gray-600" />
              {unreadCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-5 h-5 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center border-2 border-white">
                  {unreadCount}
                </span>
              )}
            </button>

            {/* Notification Dropdown - Liquid Glass Style */}
            {showNotificationDropdown && (
              <div className="absolute top-full right-0 mt-3 w-96 rounded-2xl overflow-hidden z-50 animate-fadeIn notification-dropdown">
                {/* Header */}
                <div className="px-5 py-4 border-b border-white/20 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-[#3d5a47] flex items-center justify-center">
                      <IconBell size={16} className="text-white" />
                    </div>
                    <div>
                      <h3 className="text-base font-semibold text-gray-900 tracking-tight">알림</h3>
                      {unreadCount > 0 && (
                        <p className="text-xs text-gray-500">{unreadCount}개의 새로운 알림</p>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setShowNotificationDropdown(false)}
                    className="w-8 h-8 rounded-xl bg-gray-100 hover:bg-gray-200 flex items-center justify-center transition-colors"
                  >
                    <IconClose size={16} className="text-gray-500" />
                  </button>
                </div>

                {/* Content */}
                <div className="max-h-80 overflow-y-auto">
                  {notifications.length === 0 ? (
                    <div className="py-12 text-center">
                      <div className="w-14 h-14 rounded-2xl bg-[#e8f0ea] flex items-center justify-center mx-auto mb-4">
                        <IconBell size={24} className="text-[#3d5a47]" />
                      </div>
                      <p className="text-base font-medium text-gray-700 tracking-tight">알림이 없습니다</p>
                      <p className="text-sm text-gray-400 mt-1">새로운 알림이 오면 여기에 표시됩니다</p>
                    </div>
                  ) : (
                    <div className="p-2">
                      {notifications.map((notification) => (
                        <div
                          key={notification.id}
                          className={cn(
                            "p-4 rounded-xl transition-all cursor-pointer group",
                            notification.unread
                              ? "bg-[#e8f0ea]/50 hover:bg-[#e8f0ea]"
                              : "hover:bg-gray-50"
                          )}
                        >
                          <div className="flex items-start gap-4">
                            <div className={cn(
                              "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-105",
                              notification.type === "complete"
                                ? "bg-[#e8f5ec] text-[#3d7a4a]"
                                : notification.type === "warning"
                                ? "bg-[#fef7e0] text-[#9a7b2d]"
                                : "bg-[#e8f0ea] text-[#3d5a47]"
                            )}>
                              {notification.type === "complete" ? (
                                <IconCheck size={18} />
                              ) : notification.type === "warning" ? (
                                <IconWarning size={18} />
                              ) : (
                                <IconBell size={18} />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <p className="text-sm font-semibold text-gray-900 tracking-tight">{notification.title}</p>
                                {notification.unread && (
                                  <span className="w-2 h-2 bg-[#3d5a47] rounded-full flex-shrink-0 animate-pulse" />
                                )}
                              </div>
                              <p className="text-sm text-gray-500 mt-1 line-clamp-2">{notification.message}</p>
                              <p className="text-xs text-gray-400 mt-2 font-medium">{notification.time}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Footer */}
                {notifications.length > 0 && (
                  <div className="px-5 py-3 border-t border-white/20 bg-gray-50/50">
                    <button className="w-full text-center text-sm font-medium text-[#3d5a47] hover:text-[#4a6b52] transition-colors py-2 rounded-xl hover:bg-[#e8f0ea]/50">
                      모든 알림 보기
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Stats Grid with Sparklines */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total - White card */}
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">전체 계약서</p>
              <p className="stat-number text-5xl text-gray-900">
                {stats.total}
              </p>
            </div>
            <Sparkline data={totalSparkline} width={72} height={36} color="#6b7280" />
          </div>
          <p className="text-base text-gray-500">이번 달 등록</p>
        </div>

        {/* Completed - White card */}
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-sm font-medium text-[#3d7a4a] uppercase tracking-wider mb-2">분석 완료</p>
              <p className="stat-number text-5xl text-gray-900">
                {stats.completed}
              </p>
            </div>
            <Sparkline data={completedSparkline} width={72} height={36} color="#4a9a5b" />
          </div>
          <p className="text-base text-[#3d7a4a]">+{Math.max(0, stats.completed - 2)} 이번 주</p>
        </div>

        {/* High Risk - White card */}
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-sm font-medium text-[#b54a45] uppercase tracking-wider mb-2">주의 필요</p>
              <p className="stat-number text-5xl text-gray-900">
                {stats.highRisk}
              </p>
            </div>
            <Sparkline data={riskSparkline} width={72} height={36} color="#c94b45" />
          </div>
          <p className="text-base text-[#b54a45]">고위험 계약서</p>
        </div>

        {/* Completion Rate - Accent colored card */}
        <div className="card-v2 p-7 bg-gradient-to-br from-[#e8f0ea] to-white border-[#c8e6cf]">
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-sm font-medium text-[#3d5a47] uppercase tracking-wider mb-2">진행률</p>
              <p className="stat-number text-5xl text-[#3d5a47]">
                {completionRate}<span className="text-3xl">%</span>
              </p>
            </div>
          </div>
          <p className="text-base text-[#3d5a47]/70">{stats.completed}/{stats.total} 완료</p>
        </div>
      </section>

      {/* Charts Row - 3 columns */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Monthly Analysis Chart */}
        <div className="card-v2 p-7">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">
              월별 현황
            </h3>
            <span className="text-sm text-gray-400 font-medium px-3 py-1.5 bg-gray-50 rounded-lg">6개월</span>
          </div>
          <BarChart
            data={monthlyData}
            height={160}
            showLabels={true}
            showValues={true}
          />
        </div>

        {/* Risk Distribution Chart */}
        <div className="card-v2 p-7 flex flex-col">
          <h3 className="text-lg font-semibold text-gray-900 tracking-tight mb-5">
            위험도 분포
          </h3>
          <div className="flex-1 flex items-center justify-center gap-8 -mt-2">
            <DonutChart
              segments={[
                { value: stats.lowRisk || 1, color: "#4a9a5b", label: "Low" },
                { value: stats.mediumRisk, color: "#d4a84d", label: "Medium" },
                { value: stats.highRisk, color: "#c94b45", label: "High" },
              ]}
              size={120}
              strokeWidth={14}
              centerValue={stats.completed}
              centerLabel="완료"
            />
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div className="w-3.5 h-3.5 rounded-full bg-[#4a9a5b]" />
                <span className="text-base text-gray-500 w-16">Low</span>
                <span className="font-inter text-lg font-semibold text-gray-900">{stats.lowRisk}</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3.5 h-3.5 rounded-full bg-[#d4a84d]" />
                <span className="text-base text-gray-500 w-16">Medium</span>
                <span className="font-inter text-lg font-semibold text-gray-900">{stats.mediumRisk}</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3.5 h-3.5 rounded-full bg-[#c94b45]" />
                <span className="text-base text-gray-500 w-16">High</span>
                <span className="font-inter text-lg font-semibold text-gray-900">{stats.highRisk}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Card */}
        <div className="card-v2 p-7">
          <h3 className="text-lg font-semibold text-gray-900 tracking-tight mb-6">
            분석 현황
          </h3>
          <div className="flex items-center justify-center mb-5">
            <RingProgress
              value={completionRate}
              size={120}
              strokeWidth={12}
              color="#4a9a5b"
              showPercentage={true}
            />
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="font-inter text-xl font-bold text-[#3d7a4a]">{stats.completed}</p>
              <p className="text-sm text-gray-400 uppercase tracking-wide">완료</p>
            </div>
            <div>
              <p className="font-inter text-xl font-bold text-[#3d5a47]">{stats.processing}</p>
              <p className="text-sm text-gray-400 uppercase tracking-wide">진행중</p>
            </div>
            <div>
              <p className="font-inter text-xl font-bold text-[#c94b45]">{stats.highRisk}</p>
              <p className="text-sm text-gray-400 uppercase tracking-wide">위험</p>
            </div>
          </div>
        </div>
      </section>

      {/* Bottom Section - 60:40 split */}
      <section className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Recent Contracts - 3/5 width */}
        <div className="lg:col-span-3 card-v2 overflow-hidden">
          <div className="flex items-center justify-between px-7 py-6 border-b border-gray-100/80">
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">
              최근 계약서
            </h3>
            <Link
              href="/history"
              className="text-base text-gray-500 hover:text-gray-700 flex items-center gap-1.5 transition-colors px-3 py-2 hover:bg-gray-50 rounded-lg"
            >
              전체 보기
              <IconChevronRight size={18} />
            </Link>
          </div>

          {recentContracts.length === 0 ? (
            <div className="p-12 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-[#e8f0ea] rounded-xl mb-5">
                <IconDocument size={32} className="text-[#3d5a47]" />
              </div>
              <p className="text-lg text-gray-700 mb-1.5 font-medium tracking-tight">
                아직 분석된 계약서가 없습니다
              </p>
              <p className="text-base text-gray-500">
                첫 번째 계약서를 업로드해보세요
              </p>
            </div>
          ) : (
            <div className="divide-y divide-gray-50">
              {recentContracts.map((contract) => (
                <div
                  key={contract.id}
                  className={cn(
                    "px-7 py-5 flex items-center gap-5 transition-colors cursor-pointer hover:bg-gray-50/50",
                    contract.status !== "COMPLETED" && "opacity-70"
                  )}
                  onClick={() => {
                    if (contract.status === "COMPLETED") {
                      router.push(`/analysis/${contract.id}`);
                    }
                  }}
                >
                  <div className="w-12 h-12 bg-[#e8f0ea] rounded-xl flex items-center justify-center flex-shrink-0">
                    <IconDocument size={24} className="text-[#3d5a47]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-lg font-medium text-gray-900 truncate tracking-tight">
                      {contract.title}
                    </p>
                    <p className="text-base text-gray-400 mt-1">
                      {formatRelativeTime(contract.created_at)}
                    </p>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    {getStatusBadge(contract.status)}
                    {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                    {contract.status === "COMPLETED" && (
                      <IconChevronRight size={18} className="text-gray-300" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column - Quick Actions & Tip - 2/5 width */}
        <div className="lg:col-span-2 space-y-6">
          {/* Quick Actions */}
          <div className="card-v2 p-7">
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight mb-6">
              빠른 시작
            </h3>
            <div className="space-y-4">
              <Link
                href="/scan"
                className="flex items-center gap-5 p-5 rounded-xl bg-gray-50 hover:bg-gray-100/80 transition-all group border border-transparent hover:border-gray-100"
              >
                <div className="w-12 h-12 bg-[#3d5a47] rounded-xl flex items-center justify-center group-hover:scale-105 transition-transform">
                  <IconScan size={22} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-base font-medium text-gray-900 tracking-tight">빠른 스캔</p>
                  <p className="text-sm text-gray-400 mt-0.5">카메라로 촬영</p>
                </div>
                <IconChevronRight size={18} className="text-gray-300 group-hover:text-gray-400 transition-colors" />
              </Link>

              <Link
                href="/checklist"
                className="flex items-center gap-5 p-5 rounded-xl bg-gray-50 hover:bg-gray-100/80 transition-all group border border-transparent hover:border-gray-100"
              >
                <div className="w-12 h-12 bg-[#4a9a5b] rounded-xl flex items-center justify-center group-hover:scale-105 transition-transform">
                  <IconChecklist size={22} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-base font-medium text-gray-900 tracking-tight">체크리스트</p>
                  <p className="text-sm text-gray-400 mt-0.5">검토 가이드</p>
                </div>
                <IconChevronRight size={18} className="text-gray-300 group-hover:text-gray-400 transition-colors" />
              </Link>

              <button
                onClick={() => window.dispatchEvent(new CustomEvent("openUploadSidebar"))}
                className="flex items-center gap-5 p-5 rounded-xl bg-gray-50 hover:bg-gray-100/80 transition-all group w-full text-left border border-transparent hover:border-gray-100"
              >
                <div className="w-12 h-12 bg-[#3d5a47] rounded-xl flex items-center justify-center group-hover:scale-105 transition-transform">
                  <IconDocument size={22} className="text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-base font-medium text-gray-900 tracking-tight">새 계약서 분석</p>
                  <p className="text-sm text-gray-400 mt-0.5">파일 업로드</p>
                </div>
                <IconChevronRight size={18} className="text-gray-300 group-hover:text-gray-400 transition-colors" />
              </button>
            </div>
          </div>

          {/* Safety Tip Card */}
          <div className="card-v2 p-6 bg-gradient-to-br from-[#fffbf0] to-white border-[#f5e6b8]/50">
            <div className="flex items-start gap-5">
              <div className="w-12 h-12 bg-[#fef7e0] rounded-xl flex items-center justify-center flex-shrink-0 border border-[#f5e6b8]/30">
                <svg className="w-6 h-6 text-[#d4a84d]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
              </div>
              <div>
                <p className="text-base font-semibold text-gray-900 tracking-tight mb-1.5">계약서 검토 팁</p>
                <p className="text-base text-gray-500 leading-relaxed">
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
