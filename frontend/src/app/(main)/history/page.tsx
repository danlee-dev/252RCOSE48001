"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Contract, contractsApi } from "@/lib/api";
import {
  IconDocument,
  IconCheck,
  IconWarning,
  IconDanger,
  IconLoading,
  IconChevronRight,
  IconTrash,
  IconSearch,
} from "@/components/icons";
import { cn } from "@/lib/utils";

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

  useEffect(() => {
    loadContracts();
  }, []);

  // Auto-refresh for pending/processing contracts
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
    } catch (err) {
      setError(err instanceof Error ? err.message : "계약서 목록을 불러오지 못했습니다");
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: number, e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm("이 계약서를 삭제하시겠습니까?")) {
      return;
    }

    try {
      setDeletingId(id);
      await contractsApi.delete(id);
      setContracts((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      alert(err instanceof Error ? err.message : "삭제에 실패했습니다");
    } finally {
      setDeletingId(null);
    }
  }

  // Filter contracts
  const filteredContracts = contracts.filter((contract) => {
    const matchesSearch = contract.title.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter =
      filterType === "all" ||
      (filterType === "completed" && contract.status === "COMPLETED") ||
      (filterType === "processing" && (contract.status === "PROCESSING" || contract.status === "PENDING")) ||
      (filterType === "failed" && contract.status === "FAILED");
    return matchesSearch && matchesFilter;
  });

  // Stats
  const stats = {
    total: contracts.length,
    completed: contracts.filter((c) => c.status === "COMPLETED").length,
    processing: contracts.filter((c) => c.status === "PROCESSING" || c.status === "PENDING").length,
    failed: contracts.filter((c) => c.status === "FAILED").length,
  };

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
    <>
      {/* Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        <div className="card-apple p-4">
          <p className="text-xs text-gray-500 mb-1">전체</p>
          <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
        </div>
        <div className="card-apple p-4">
          <p className="text-xs text-[#3d7a4a] mb-1">완료</p>
          <p className="text-2xl font-bold text-[#3d7a4a]">{stats.completed}</p>
        </div>
        <div className="card-apple p-4">
          <p className="text-xs text-[#3d5a47] mb-1">진행중</p>
          <p className="text-2xl font-bold text-[#3d5a47]">{stats.processing}</p>
        </div>
        <div className="card-apple p-4">
          <p className="text-xs text-[#b54a45] mb-1">실패</p>
          <p className="text-2xl font-bold text-[#b54a45]">{stats.failed}</p>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <IconSearch size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="계약서 검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 liquid-input text-sm outline-none"
          />
        </div>
        <div className="flex gap-2">
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
                "px-3 py-2 text-sm font-medium rounded-xl transition-all",
                filterType === filter.value
                  ? "bg-gray-900 text-white"
                  : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 text-sm text-[#b54a45] bg-[#fdedec] border border-[#f5c6c4] rounded-xl animate-fadeIn">
          {error}
        </div>
      )}

      {filteredContracts.length === 0 ? (
        <div className="py-16 text-center">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-[#e8f0ea] rounded-xl mb-4">
            <IconDocument size={28} className="text-[#3d5a47]" />
          </div>
          {contracts.length === 0 ? (
            <>
              <p className="text-base text-gray-700 mb-1 font-medium tracking-tight">아직 분석된 계약서가 없습니다</p>
              <p className="text-sm text-gray-500 mb-6">첫 번째 계약서를 업로드해보세요</p>
              <p className="text-sm text-gray-400">하단의 업로드 버튼을 눌러주세요</p>
            </>
          ) : (
            <>
              <p className="text-base text-gray-700 mb-1 font-medium tracking-tight">검색 결과가 없습니다</p>
              <p className="text-sm text-gray-500">다른 검색어나 필터를 사용해보세요</p>
            </>
          )}
        </div>
      ) : (
        <>
          {/* Mobile Card View */}
          <div className="space-y-3 sm:hidden">
            {filteredContracts.map((contract) => (
              <div
                key={contract.id}
                className={cn(
                  "card-apple p-4 active:scale-[0.99]",
                  deletingId === contract.id && "opacity-50 pointer-events-none",
                  contract.status === "COMPLETED" && "cursor-pointer"
                )}
                onClick={() => {
                  if (contract.status === "COMPLETED") {
                    router.push(`/analysis/${contract.id}`);
                  }
                }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center bg-gray-100 text-gray-500">
                      <IconDocument size={20} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate tracking-tight">
                        {contract.title}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {formatDate(contract.created_at)}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        {getStatusBadge(contract.status)}
                        {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => handleDelete(contract.id, e)}
                      className="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg transition-all"
                      disabled={deletingId === contract.id}
                    >
                      {deletingId === contract.id ? (
                        <IconLoading size={16} />
                      ) : (
                        <IconTrash size={16} />
                      )}
                    </button>
                    {contract.status === "COMPLETED" && (
                      <IconChevronRight size={16} className="text-gray-400" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Desktop Table View */}
          <div className="hidden sm:block card-apple overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-white/40 border-b border-white/50">
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                    제목
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                    상태
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                    위험도
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                    등록일
                  </th>
                  <th className="w-20"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/30">
                {filteredContracts.map((contract) => (
                  <tr
                    key={contract.id}
                    className={cn(
                      "group hover:bg-white/30 cursor-pointer transition-colors",
                      deletingId === contract.id && "opacity-50 pointer-events-none"
                    )}
                    onClick={() => {
                      if (contract.status === "COMPLETED") {
                        router.push(`/analysis/${contract.id}`);
                      }
                    }}
                  >
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-3">
                        <div className="flex-shrink-0 w-9 h-9 rounded-[10px] flex items-center justify-center bg-gray-100 text-gray-500">
                          <IconDocument size={18} />
                        </div>
                        <span className="text-sm font-medium text-gray-900 truncate max-w-xs tracking-tight">
                          {contract.title}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {getStatusBadge(contract.status)}
                    </td>
                    <td className="px-4 py-3">
                      {contract.status === "COMPLETED" && getRiskBadge(contract.risk_level)}
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-sm text-gray-500">
                        {formatDate(contract.created_at)}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={(e) => handleDelete(contract.id, e)}
                          className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50/50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                          disabled={deletingId === contract.id}
                          title="삭제"
                        >
                          {deletingId === contract.id ? (
                            <IconLoading size={16} />
                          ) : (
                            <IconTrash size={16} />
                          )}
                        </button>
                        {contract.status === "COMPLETED" && (
                          <div className="p-2 text-gray-400 group-hover:text-gray-600 transition-colors">
                            <IconChevronRight size={16} />
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
