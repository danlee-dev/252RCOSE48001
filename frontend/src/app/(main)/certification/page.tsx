"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { LegalNoticeSessionListItem, legalNoticeApi } from "@/lib/api";
import {
  IconCheck,
  IconLoading,
  IconChevronRight,
  IconTrash,
  IconSearch,
  IconFileText,
  IconChevronLeft,
  IconPlus,
  IconShield,
} from "@/components/icons";
import { Sparkline } from "@/components/charts";
import { cn } from "@/lib/utils";

const ITEMS_PER_PAGE = 10;

function getStatusBadge(status: string) {
  switch (status) {
    case "completed":
      return (
        <span className="badge badge-success">
          <IconCheck size={12} />
          완료
        </span>
      );
    case "generating":
      return (
        <span className="badge badge-neutral">
          <IconLoading size={12} />
          생성중
        </span>
      );
    case "collecting":
      return (
        <span className="badge badge-warning">
          정보수집중
        </span>
      );
    default:
      return (
        <span className="badge badge-neutral">
          {status}
        </span>
      );
  }
}

function formatDate(dateString: string) {
  const date = new Date(dateString);
  return date.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

type FilterType = "all" | "completed" | "collecting";

interface SessionStats {
  total: number;
  completed: number;
  collecting: number;
}

export default function CertificationPage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<LegalNoticeSessionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [stats, setStats] = useState<SessionStats>({
    total: 0,
    completed: 0,
    collecting: 0,
  });

  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);

  const loadSessions = useCallback(async (page: number, statusFilter?: string, search?: string) => {
    try {
      setLoading(true);
      const skip = (page - 1) * ITEMS_PER_PAGE;
      const data = await legalNoticeApi.listSessions(
        skip,
        ITEMS_PER_PAGE,
        statusFilter === "all" ? undefined : statusFilter,
        search || undefined
      );
      setSessions(data.items);
      setTotalItems(data.total);
      setStats({
        total: data.stats.total || 0,
        completed: data.stats.completed || 0,
        collecting: data.stats.collecting || 0,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "세션 목록을 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions(currentPage, filterType, searchQuery);
  }, [currentPage, filterType, loadSessions]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (currentPage === 1) {
        loadSessions(1, filterType, searchQuery);
      } else {
        setCurrentPage(1);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  async function handleDelete(id: string, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm("이 세션을 삭제하시겠습니까?")) {
      return;
    }

    try {
      setDeletingId(id);
      await legalNoticeApi.deleteSession(id);
      loadSessions(currentPage, filterType, searchQuery);
    } catch (err) {
      alert(err instanceof Error ? err.message : "삭제에 실패했습니다");
    } finally {
      setDeletingId(null);
    }
  }

  function handlePageChange(page: number) {
    if (page < 1 || page > totalPages) return;
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function getPageNumbers(): (number | "...")[] {
    const pages: (number | "...")[] = [];
    const maxVisiblePages = 5;

    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= 4; i++) pages.push(i);
        pages.push("...");
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        pages.push(1);
        pages.push("...");
        for (let i = totalPages - 3; i <= totalPages; i++) pages.push(i);
      } else {
        pages.push(1);
        pages.push("...");
        for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
        pages.push("...");
        pages.push(totalPages);
      }
    }

    return pages;
  }

  // Sparkline data
  const sparklineData = (() => {
    const days = 10;
    const now = new Date();
    now.setHours(23, 59, 59, 999);

    const totalByDay: number[] = [];
    const completedByDay: number[] = [];
    const collectingByDay: number[] = [];

    for (let i = days - 1; i >= 0; i--) {
      const dayEnd = new Date(now);
      dayEnd.setDate(now.getDate() - i);
      dayEnd.setHours(23, 59, 59, 999);

      const sessionsUntilDay = sessions.filter((s) => {
        const createdAt = new Date(s.created_at);
        return createdAt <= dayEnd;
      });

      totalByDay.push(sessionsUntilDay.length);
      completedByDay.push(sessionsUntilDay.filter((s) => s.status === "completed").length);
      collectingByDay.push(sessionsUntilDay.filter((s) => s.status === "collecting").length);
    }

    return { totalByDay, completedByDay, collectingByDay };
  })();

  if (loading && sessions.length === 0) {
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
    <div className="space-y-10">
      {/* Header Section */}
      <section className="relative">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-[#3d5a47]/10 to-transparent rounded-full blur-2xl" />
        <div className="relative flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
                <IconShield size={20} className="text-white" />
              </div>
              <span className="text-sm font-medium text-[#3d5a47] tracking-tight">Legal Document</span>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
              내용증명 작성
            </h1>
            <p className="text-base text-gray-500 mt-2">
              AI와 대화하며 내용증명을 작성하세요
            </p>
          </div>
          <Link
            href="/certification/new"
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 transition-all"
          >
            <IconPlus size={20} />
            새 세션 시작
          </Link>
        </div>
      </section>

      {/* Stats Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">전체</p>
              <p className="stat-number text-5xl text-gray-900">{stats.total}</p>
            </div>
            <Sparkline
              data={sparklineData.totalByDay}
              width={72}
              height={36}
              color="#6b7280"
            />
          </div>
        </div>
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-[#3d7a4a] uppercase tracking-wider mb-2">완료</p>
              <p className="stat-number text-5xl text-[#3d7a4a]">{stats.completed}</p>
            </div>
            <Sparkline
              data={sparklineData.completedByDay}
              width={72}
              height={36}
              color="#4a9a5b"
            />
          </div>
        </div>
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-[#9a7b2d] uppercase tracking-wider mb-2">작성중</p>
              <p className="stat-number text-5xl text-[#9a7b2d]">{stats.collecting}</p>
            </div>
            <Sparkline
              data={sparklineData.collectingByDay}
              width={72}
              height={36}
              color="#d4a84d"
            />
          </div>
        </div>
      </section>

      {/* Search and Filter */}
      <section className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1 liquid-input-wrapper">
          <IconSearch size={20} className="absolute left-5 top-1/2 -translate-y-1/2 text-gray-500 z-10" />
          <input
            type="text"
            placeholder="세션 검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-14 pr-5 py-4 liquid-input rounded-xl text-base outline-none transition-all placeholder:text-gray-400"
          />
        </div>
        <div className="flex gap-3">
          {[
            { value: "all" as const, label: "전체" },
            { value: "completed" as const, label: "완료" },
            { value: "collecting" as const, label: "작성중" },
          ].map((filter) => (
            <button
              key={filter.value}
              onClick={() => {
                setFilterType(filter.value);
                setCurrentPage(1);
              }}
              className={cn(
                "px-5 py-3 text-base font-medium rounded-xl transition-all",
                filterType === filter.value
                  ? "bg-[#3d5a47] text-white"
                  : "bg-white text-gray-600 border border-gray-200/60 hover:bg-gray-50 hover:border-gray-300/60"
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </section>

      {error && (
        <div className="mb-5 p-5 text-base text-[#b54a45] bg-[#fdedec] border border-[#f5c6c4] rounded-xl animate-fadeIn">
          {error}
        </div>
      )}

      {sessions.length === 0 ? (
        <div className="py-20 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[#e8f0ea] rounded-xl mb-5">
            <IconFileText size={32} className="text-[#3d5a47]" />
          </div>
          {filterType === "all" && !searchQuery ? (
            <>
              <p className="text-lg text-gray-700 mb-1.5 font-medium tracking-tight">아직 작성된 내용증명이 없습니다</p>
              <p className="text-base text-gray-500 mb-6">새 세션을 시작하여 첫 내용증명을 작성해보세요</p>
              <Link
                href="/certification/new"
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 transition-all"
              >
                <IconPlus size={20} />
                새 세션 시작
              </Link>
            </>
          ) : (
            <>
              <p className="text-lg text-gray-700 mb-1.5 font-medium tracking-tight">검색 결과가 없습니다</p>
              <p className="text-base text-gray-500">다른 검색어나 필터를 사용해보세요</p>
            </>
          )}
        </div>
      ) : (
        <section className="space-y-5">
          {/* Mobile Card View */}
          <div className="space-y-4 sm:hidden">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "card-v2 p-5 active:scale-[0.99] transition-all cursor-pointer",
                  deletingId === session.id && "opacity-50 pointer-events-none"
                )}
                onClick={() => router.push(`/certification/${session.id}`)}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4 flex-1 min-w-0">
                    <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center bg-[#e8f0ea] text-[#3d5a47]">
                      <IconFileText size={24} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-base font-medium text-gray-900 truncate tracking-tight">
                        {session.title || "제목 없음"}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        {formatDate(session.created_at)}
                      </p>
                      <div className="flex items-center gap-2 mt-2.5">
                        {getStatusBadge(session.status)}
                        {session.contract_title && (
                          <span className="text-sm text-gray-400 truncate">
                            {session.contract_title}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => handleDelete(session.id, e)}
                      className="w-10 h-10 flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg transition-all"
                      disabled={deletingId === session.id}
                    >
                      {deletingId === session.id ? (
                        <IconLoading size={18} />
                      ) : (
                        <IconTrash size={18} />
                      )}
                    </button>
                    <IconChevronRight size={18} className="text-gray-400" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Desktop Table View */}
          <div className="hidden sm:block card-v2 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50/50 border-b border-gray-100">
                  <th className="text-left px-6 py-5 text-sm font-medium text-gray-400 uppercase tracking-wider">
                    제목
                  </th>
                  <th className="text-left px-6 py-5 text-sm font-medium text-gray-400 uppercase tracking-wider">
                    상태
                  </th>
                  <th className="text-left px-6 py-5 text-sm font-medium text-gray-400 uppercase tracking-wider">
                    연결된 계약서
                  </th>
                  <th className="text-left px-6 py-5 text-sm font-medium text-gray-400 uppercase tracking-wider">
                    생성일
                  </th>
                  <th className="w-28"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {sessions.map((session) => (
                  <tr
                    key={session.id}
                    className={cn(
                      "group hover:bg-gray-50/50 cursor-pointer transition-colors",
                      deletingId === session.id && "opacity-50 pointer-events-none"
                    )}
                    onClick={() => router.push(`/certification/${session.id}`)}
                  >
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-4">
                        <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center bg-[#e8f0ea] text-[#3d5a47]">
                          <IconFileText size={22} />
                        </div>
                        <span className="text-base font-medium text-gray-900 truncate max-w-md tracking-tight">
                          {session.title || "제목 없음"}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      {getStatusBadge(session.status)}
                    </td>
                    <td className="px-6 py-5">
                      {session.contract_title ? (
                        <span className="text-base text-gray-700">{session.contract_title}</span>
                      ) : (
                        <span className="text-base text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-6 py-5">
                      <span className="text-base text-gray-500">
                        {formatDate(session.created_at)}
                      </span>
                    </td>
                    <td className="px-6 py-5">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={(e) => handleDelete(session.id, e)}
                          className="p-2.5 text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                          disabled={deletingId === session.id}
                          title="삭제"
                        >
                          {deletingId === session.id ? (
                            <IconLoading size={18} />
                          ) : (
                            <IconTrash size={18} />
                          )}
                        </button>
                        <div className="p-2.5 text-gray-400 group-hover:text-gray-600 transition-colors">
                          <IconChevronRight size={18} />
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-1.5 pt-6">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className={cn(
                  "p-2.5 rounded-[10px] transition-all",
                  currentPage === 1
                    ? "text-gray-300 cursor-not-allowed"
                    : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
                )}
              >
                <IconChevronLeft size={20} />
              </button>

              {getPageNumbers().map((page, index) => (
                <button
                  key={index}
                  onClick={() => typeof page === "number" && handlePageChange(page)}
                  disabled={page === "..."}
                  className={cn(
                    "min-w-[40px] h-10 px-3 rounded-[10px] text-base font-medium transition-all",
                    page === currentPage
                      ? "bg-[#3d5a47] text-white"
                      : page === "..."
                        ? "text-gray-400 cursor-default"
                        : "text-gray-600 hover:bg-gray-100"
                  )}
                >
                  {page}
                </button>
              ))}

              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className={cn(
                  "p-2.5 rounded-[10px] transition-all",
                  currentPage === totalPages
                    ? "text-gray-300 cursor-not-allowed"
                    : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
                )}
              >
                <IconChevronRight size={20} />
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
