"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Contract, ContractStats, contractsApi } from "@/lib/api";
import {
  IconDocument,
  IconCheck,
  IconWarning,
  IconDanger,
  IconLoading,
  IconChevronRight,
  IconTrash,
  IconSearch,
  IconFileText,
  IconChevronLeft,
} from "@/components/icons";
import { Sparkline } from "@/components/charts";
import { cn } from "@/lib/utils";

const ITEMS_PER_PAGE = 10;

function getRiskBadge(riskLevel: string | null) {
  if (!riskLevel) return null;

  const level = riskLevel.toLowerCase();
  if (level === "high" || level === "danger") {
    return (
      <span className="badge badge-danger">
        <IconDanger size={12} />
        High
      </span>
    );
  }
  if (level === "medium" || level === "warning") {
    return (
      <span className="badge badge-warning">
        <IconWarning size={12} />
        Medium
      </span>
    );
  }
  return (
    <span className="badge badge-success">
      <IconCheck size={12} />
      Low
    </span>
  );
}

function getStatusBadge(status: string) {
  switch (status) {
    case "COMPLETED":
      return (
        <span className="badge badge-success">
          <IconCheck size={12} />
          완료
        </span>
      );
    case "PROCESSING":
      return (
        <span className="badge badge-neutral">
          <IconLoading size={12} />
          분석중
        </span>
      );
    case "PENDING":
      return (
        <span className="badge badge-neutral">
          대기중
        </span>
      );
    case "FAILED":
      return (
        <span className="badge badge-danger">
          <IconDanger size={12} />
          실패
        </span>
      );
    default:
      return null;
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

type FilterType = "all" | "completed" | "processing" | "failed";

export default function HistoryPage() {
  const router = useRouter();
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [stats, setStats] = useState<ContractStats>({
    total: 0,
    completed: 0,
    processing: 0,
    failed: 0,
  });

  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);

  const loadContracts = useCallback(async (page: number) => {
    try {
      setLoading(true);
      const skip = (page - 1) * ITEMS_PER_PAGE;
      const data = await contractsApi.list(skip, ITEMS_PER_PAGE);
      setContracts(data.items);
      setTotalItems(data.total);
      setStats(data.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "계약서 목록을 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadContracts(currentPage);
  }, [currentPage, loadContracts]);

  // Auto-refresh for pending/processing contracts
  const hasPendingOrProcessing = contracts.some(
    (c) => c.status === "PENDING" || c.status === "PROCESSING"
  );

  useEffect(() => {
    if (!hasPendingOrProcessing) return;

    const interval = setInterval(async () => {
      try {
        const skip = (currentPage - 1) * ITEMS_PER_PAGE;
        const data = await contractsApi.list(skip, ITEMS_PER_PAGE);
        setContracts(data.items);
        setTotalItems(data.total);
        setStats(data.stats);
      } catch {
        // Silently fail
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [hasPendingOrProcessing, currentPage]);

  async function handleDelete(id: number, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm("이 계약서를 삭제하시겠습니까?")) {
      return;
    }

    try {
      setDeletingId(id);
      await contractsApi.delete(id);
      // 삭제 후 현재 페이지 다시 로드
      loadContracts(currentPage);
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

  // 페이지 번호 배열 생성
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

  // Filter contracts (supports Korean search)
  const filteredContracts = contracts.filter((contract) => {
    const query = searchQuery.trim().toLowerCase();
    const title = contract.title.toLowerCase();
    // NFC 정규화로 한글 검색 지원
    const matchesSearch = !query || title.normalize("NFC").includes(query.normalize("NFC"));
    const matchesFilter =
      filterType === "all" ||
      (filterType === "completed" && contract.status === "COMPLETED") ||
      (filterType === "processing" && (contract.status === "PROCESSING" || contract.status === "PENDING")) ||
      (filterType === "failed" && contract.status === "FAILED");
    return matchesSearch && matchesFilter;
  });

  // Sparkline data - 최근 10일간 일별 집계
  const sparklineData = (() => {
    const days = 10;
    const now = new Date();
    now.setHours(23, 59, 59, 999);

    const totalByDay: number[] = [];
    const completedByDay: number[] = [];
    const processingByDay: number[] = [];
    const failedByDay: number[] = [];

    for (let i = days - 1; i >= 0; i--) {
      const dayStart = new Date(now);
      dayStart.setDate(now.getDate() - i);
      dayStart.setHours(0, 0, 0, 0);

      const dayEnd = new Date(dayStart);
      dayEnd.setHours(23, 59, 59, 999);

      // 해당 날짜까지의 누적 데이터
      const contractsUntilDay = contracts.filter((c) => {
        const createdAt = new Date(c.created_at);
        return createdAt <= dayEnd;
      });

      totalByDay.push(contractsUntilDay.length);
      completedByDay.push(contractsUntilDay.filter((c) => c.status === "COMPLETED").length);
      processingByDay.push(contractsUntilDay.filter((c) => c.status === "PROCESSING" || c.status === "PENDING").length);
      failedByDay.push(contractsUntilDay.filter((c) => c.status === "FAILED").length);
    }

    return { totalByDay, completedByDay, processingByDay, failedByDay };
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
    <div className="space-y-10">
      {/* Header Section */}
      <section className="relative">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-[#3d5a47]/10 to-transparent rounded-full blur-2xl" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
              <IconFileText size={20} className="text-white" />
            </div>
            <span className="text-sm font-medium text-[#3d5a47] tracking-tight">Analysis History</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
            계약서 분석 기록
          </h1>
        </div>
      </section>

      {/* Stats Cards */}
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-6">
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
              <p className="text-sm font-medium text-[#3d5a47] uppercase tracking-wider mb-2">진행중</p>
              <p className="stat-number text-5xl text-[#3d5a47]">{stats.processing}</p>
            </div>
            <Sparkline
              data={sparklineData.processingByDay}
              width={72}
              height={36}
              color="#3d5a47"
            />
          </div>
        </div>
        <div className="card-v2 p-7">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-[#b54a45] uppercase tracking-wider mb-2">실패</p>
              <p className="stat-number text-5xl text-[#b54a45]">{stats.failed}</p>
            </div>
            <Sparkline
              data={sparklineData.failedByDay}
              width={72}
              height={36}
              color="#c94b45"
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
            placeholder="계약서 검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-14 pr-5 py-4 liquid-input rounded-xl text-base outline-none transition-all placeholder:text-gray-400"
          />
        </div>
        <div className="flex gap-3">
          {[
            { value: "all" as const, label: "전체" },
            { value: "completed" as const, label: "완료" },
            { value: "processing" as const, label: "진행중" },
            { value: "failed" as const, label: "실패" },
          ].map((filter) => (
            <button
              key={filter.value}
              onClick={() => setFilterType(filter.value)}
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

      {filteredContracts.length === 0 ? (
        <div className="py-20 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[#e8f0ea] rounded-xl mb-5">
            <IconDocument size={32} className="text-[#3d5a47]" />
          </div>
          {contracts.length === 0 ? (
            <>
              <p className="text-lg text-gray-700 mb-1.5 font-medium tracking-tight">아직 분석된 계약서가 없습니다</p>
              <p className="text-base text-gray-500 mb-6">첫 번째 계약서를 업로드해보세요</p>
              <p className="text-base text-gray-400">하단의 업로드 버튼을 눌러주세요</p>
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
            {filteredContracts.map((contract) => (
              <div
                key={contract.id}
                className={cn(
                  "card-v2 p-5 active:scale-[0.99] transition-all",
                  deletingId === contract.id && "opacity-50 pointer-events-none",
                  contract.status === "COMPLETED" && "cursor-pointer"
                )}
                onClick={() => {
                  if (contract.status === "COMPLETED") {
                    router.push(`/analysis/${contract.id}`);
                  }
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4 flex-1 min-w-0">
                    <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center bg-[#e8f0ea] text-[#3d5a47]">
                      <IconDocument size={24} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-base font-medium text-gray-900 truncate tracking-tight">
                        {contract.title}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        {formatDate(contract.created_at)}
                      </p>
                      <div className="flex items-center gap-2 mt-2.5">
                        {getStatusBadge(contract.status)}
                        {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => handleDelete(contract.id, e)}
                      className="w-10 h-10 flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg transition-all"
                      disabled={deletingId === contract.id}
                    >
                      {deletingId === contract.id ? (
                        <IconLoading size={18} />
                      ) : (
                        <IconTrash size={18} />
                      )}
                    </button>
                    {contract.status === "COMPLETED" && (
                      <IconChevronRight size={18} className="text-gray-400" />
                    )}
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
                    위험도
                  </th>
                  <th className="text-left px-6 py-5 text-sm font-medium text-gray-400 uppercase tracking-wider">
                    등록일
                  </th>
                  <th className="w-28"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {filteredContracts.map((contract) => (
                  <tr
                    key={contract.id}
                    className={cn(
                      "group hover:bg-gray-50/50 cursor-pointer transition-colors",
                      deletingId === contract.id && "opacity-50 pointer-events-none"
                    )}
                    onClick={() => {
                      if (contract.status === "COMPLETED") {
                        router.push(`/analysis/${contract.id}`);
                      }
                    }}
                  >
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-4">
                        <div className="flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center bg-[#e8f0ea] text-[#3d5a47]">
                          <IconDocument size={22} />
                        </div>
                        <span className="text-base font-medium text-gray-900 truncate max-w-md tracking-tight">
                          {contract.title}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      {getStatusBadge(contract.status)}
                    </td>
                    <td className="px-6 py-5">
                      {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                    </td>
                    <td className="px-6 py-5">
                      <span className="text-base text-gray-500">
                        {formatDate(contract.created_at)}
                      </span>
                    </td>
                    <td className="px-6 py-5">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={(e) => handleDelete(contract.id, e)}
                          className="p-2.5 text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                          disabled={deletingId === contract.id}
                          title="삭제"
                        >
                          {deletingId === contract.id ? (
                            <IconLoading size={18} />
                          ) : (
                            <IconTrash size={18} />
                          )}
                        </button>
                        {contract.status === "COMPLETED" && (
                          <div className="p-2.5 text-gray-400 group-hover:text-gray-600 transition-colors">
                            <IconChevronRight size={18} />
                          </div>
                        )}
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
