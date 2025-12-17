"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import {
  IconLoading,
  IconDocument,
  IconChevronRight,
  IconHistory,
  IconEdit,
  IconCheck,
  IconInfo,
  IconSearch,
  IconClose,
} from "@/components/icons";
import { cn } from "@/lib/utils";
import { Contract, contractsApi, DocumentVersion } from "@/lib/api";

// Subtle accent colors for contract distinction (left border only)
const CONTRACT_ACCENTS = [
  "#3d5a47", // green
  "#5a6b8f", // blue-gray
  "#7a6b5a", // brown
  "#6b5a7a", // purple-gray
  "#5a7a6b", // teal
];

interface VersionWithContract extends DocumentVersion {
  contractTitle: string;
}

function getChangeTypeInfo(_createdBy: string | undefined, changeSummary: string | undefined) {
  // Determine type based on change_summary content
  if (!changeSummary) {
    return { label: "변경", color: "bg-gray-100 text-gray-600 border-gray-200" };
  }

  const summary = changeSummary.toLowerCase();
  if (summary.includes("원본") || summary.includes("초기")) {
    return { label: "원본", color: "bg-[#e8f0ea] text-[#3d5a47] border-[#c8e6cf]" };
  }
  if (summary.includes("복원")) {
    return { label: "복원", color: "bg-[#fef7e0] text-[#9a7b2d] border-[#f5e6b8]" };
  }
  if (summary.includes("추가")) {
    return { label: "추가", color: "bg-[#e8f5ec] text-[#3d7a4a] border-[#c8e6cf]" };
  }
  if (summary.includes("삭제") || summary.includes("제거")) {
    return { label: "삭제", color: "bg-[#fdedec] text-[#b54a45] border-[#f5c6c4]" };
  }
  if (summary.includes("위험") || summary.includes("해소")) {
    return { label: "위험 해소", color: "bg-[#e8f5ec] text-[#3d7a4a] border-[#c8e6cf]" };
  }
  return { label: "수정", color: "bg-[#e8f0ea] text-[#3d5a47] border-[#c8e6cf]" };
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

export default function ContractRevisionsPage() {
  const [loading, setLoading] = useState(true);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [versions, setVersions] = useState<VersionWithContract[]>([]);
  const [selectedContractId, setSelectedContractId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Get accent color for contract based on its ID
  function getContractAccent(contractId: number) {
    return CONTRACT_ACCENTS[contractId % CONTRACT_ACCENTS.length];
  }

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // 1. 모든 계약서 목록 조회
      const contractData = await contractsApi.list(0, 100);
      setContracts(contractData.items);

      // 2. 각 계약서의 버전 목록 조회
      const allVersions: VersionWithContract[] = [];

      for (const contract of contractData.items) {
        try {
          const versionData = await contractsApi.getVersions(contract.id);
          const versionsWithTitle = versionData.versions.map((v) => ({
            ...v,
            contractTitle: contract.title,
          }));
          allVersions.push(...versionsWithTitle);
        } catch {
          // 버전이 없는 계약서는 건너뜀
        }
      }

      // 3. 최신순 정렬
      allVersions.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setVersions(allVersions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "데이터를 불러오는 데 실패했습니다");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Filter contracts by search query
  const filteredContracts = contracts.filter((c) => {
    if (!searchQuery.trim()) return true;
    return c.title.toLowerCase().normalize("NFC").includes(searchQuery.toLowerCase().normalize("NFC"));
  });

  // Get unique contracts with versions (filtered by search)
  const contractsWithVersions = filteredContracts.filter((c) =>
    versions.some((v) => v.contract_id === c.id)
  );

  // Get recent 3 contract IDs (when not searching)
  const recentContractIds = !searchQuery.trim() && !selectedContractId
    ? [...new Set(versions.map(v => v.contract_id))].slice(0, 3)
    : null;

  // Filter versions based on selection and search
  const filteredVersions = versions.filter((v) => {
    // When searching, filter by search query
    if (searchQuery.trim()) {
      return v.contractTitle.toLowerCase().normalize("NFC").includes(searchQuery.toLowerCase().normalize("NFC"));
    }
    // When a specific contract is selected
    if (selectedContractId) {
      return v.contract_id === selectedContractId;
    }
    // Default: show only recent 3 contracts
    return recentContractIds?.includes(v.contract_id) ?? true;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4 animate-fadeIn">
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-xl shadow-[#3d5a47]/30">
              <IconLoading size={36} className="text-white" />
            </div>
            <div className="absolute -inset-2 rounded-3xl border-2 border-[#3d5a47]/20 animate-pulse" />
          </div>
          <p className="text-lg text-gray-600 font-medium tracking-tight">수정 기록을 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="liquid-glass rounded-2xl p-10 text-center max-w-md border border-white/40">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#fdedec] to-white rounded-2xl mb-6 border border-[#f5c6c4]">
            <IconInfo size={36} className="text-[#b54a45]" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-3">오류가 발생했습니다</h2>
          <p className="text-base text-gray-500 mb-6">{error}</p>
          <button
            onClick={loadData}
            className="px-6 py-3 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 transition-all"
          >
            다시 시도
          </button>
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
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
              <IconEdit size={24} className="text-white" />
            </div>
            <span className="text-base font-medium text-[#3d5a47] tracking-tight">Revisions</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
            수정 기록
          </h1>
          <p className="text-lg text-gray-500 mt-2">
            계약서 분석 결과를 바탕으로 수정한 내역을 확인할 수 있습니다
          </p>
        </div>
      </section>

      {/* Search with Dropdown */}
      <section ref={searchRef} className="relative">
        <div className="relative liquid-input-wrapper">
          <IconSearch size={20} className="absolute left-5 top-1/2 -translate-y-1/2 text-gray-500 z-10" />
          <input
            type="text"
            placeholder="계약서 검색..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowDropdown(true);
              setSelectedContractId(null);
            }}
            onFocus={() => setShowDropdown(true)}
            className="w-full pl-14 pr-12 py-4 liquid-input rounded-xl text-base outline-none transition-all placeholder:text-gray-400"
          />
          {(searchQuery || selectedContractId) && (
            <button
              onClick={() => {
                setSearchQuery("");
                setSelectedContractId(null);
                setShowDropdown(false);
              }}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors z-10"
            >
              <IconClose size={18} />
            </button>
          )}
        </div>

        {/* Dropdown */}
        {showDropdown && searchQuery.trim() && filteredContracts.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl border border-gray-200 shadow-xl overflow-hidden z-20 max-h-80 overflow-y-auto">
            <div className="px-4 py-2 bg-gray-50 border-b border-gray-100">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                검색 결과 ({filteredContracts.length}건)
              </p>
            </div>
            {filteredContracts.map((contract) => {
              const contractVersions = versions.filter(v => v.contract_id === contract.id);
              const latestVersion = contractVersions[0];
              const createdDate = new Date(contract.created_at);
              const dateStr = createdDate.toLocaleDateString("ko-KR", {
                year: "numeric",
                month: "short",
                day: "numeric",
              });

              return (
                <button
                  key={contract.id}
                  onClick={() => {
                    setSelectedContractId(contract.id);
                    setSearchQuery(contract.title);
                    setShowDropdown(false);
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors text-left border-b border-gray-50 last:border-b-0"
                >
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-gray-100 text-gray-600">
                    <IconDocument size={18} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{contract.title}</p>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{dateStr}</span>
                      {contractVersions.length > 0 && (
                        <>
                          <span className="text-gray-300">|</span>
                          <span>{contractVersions.length}개 버전</span>
                        </>
                      )}
                    </div>
                  </div>
                  {latestVersion && (
                    <span className="text-xs text-gray-400">
                      {formatRelativeTime(latestVersion.created_at)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}

        {/* Selected Contract Indicator */}
        {selectedContractId && (
          <div className="mt-3 flex items-center gap-2">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-200 bg-gray-50 text-gray-700">
              <IconDocument size={14} />
              <span className="text-sm font-medium">
                {contracts.find(c => c.id === selectedContractId)?.title}
              </span>
            </div>
            <button
              onClick={() => {
                setSelectedContractId(null);
                setSearchQuery("");
              }}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              전체 보기
            </button>
          </div>
        )}
      </section>

      {versions.length === 0 ? (
        <section className="liquid-glass rounded-2xl p-12 text-center border border-white/40">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-[#e8f0ea] to-[#d8e8dc] rounded-2xl mb-8">
            <IconHistory size={48} className="text-[#3d5a47]" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 tracking-tight mb-3">
            수정 기록이 없습니다
          </h2>
          <p className="text-lg text-gray-500 mb-8 max-w-md mx-auto">
            계약서 분석 결과를 바탕으로 조항을 수정하면 이곳에 버전 히스토리가 기록됩니다.
          </p>
          <Link
            href="/history"
            className="inline-flex items-center gap-3 px-8 py-4 text-lg font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl hover:shadow-lg hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 transition-all duration-200"
          >
            <IconDocument size={22} />
            분석 기록 보기
          </Link>
        </section>
      ) : (
        <section className="space-y-6">
          {/* Stats Summary */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="liquid-glass rounded-2xl p-6 border border-white/40">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#e8f0ea] to-[#d8e8dc] flex items-center justify-center">
                  <IconDocument size={28} className="text-[#3d5a47]" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-gray-900 tracking-tight">
                    {contractsWithVersions.filter(c =>
                      versions.some(v => v.contract_id === c.id && v.version_number >= 2)
                    ).length}
                  </p>
                  <p className="text-base text-gray-500">수정된 계약서</p>
                </div>
              </div>
            </div>
            <div className="liquid-glass rounded-2xl p-6 border border-white/40">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#e8f5ec] to-[#d8f0dc] flex items-center justify-center">
                  <IconHistory size={28} className="text-[#4a9a5b]" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-gray-900 tracking-tight">{versions.length}</p>
                  <p className="text-base text-gray-500">전체 버전</p>
                </div>
              </div>
            </div>
            <div className="liquid-glass rounded-2xl p-6 border border-white/40">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-[#fef7e0] to-[#fef3c7] flex items-center justify-center">
                  <IconEdit size={28} className="text-[#9a7b2d]" />
                </div>
                <div>
                  <p className="text-3xl font-bold text-gray-900 tracking-tight">
                    {versions.filter((v) => v.version_number > 1).length}
                  </p>
                  <p className="text-base text-gray-500">수정 횟수</p>
                </div>
              </div>
            </div>
          </div>

          {/* Revision List */}
          <div className="liquid-glass rounded-2xl border border-white/40 overflow-hidden">
            <div className="bg-gradient-to-r from-gray-50/80 to-white/80 px-6 py-5 border-b border-gray-100">
              <h3 className="text-xl font-bold text-gray-900 tracking-tight">버전 히스토리</h3>
              <p className="text-base text-gray-500 mt-1">
                {selectedContractId ? "선택된 계약서의 버전 기록입니다" : "모든 계약서의 버전 기록입니다"}
              </p>
            </div>
            <div>
              {filteredVersions.map((version, index) => {
                const changeInfo = getChangeTypeInfo(version.created_by, version.change_summary);
                const accentColor = getContractAccent(version.contract_id);
                const showContractHeader =
                  !selectedContractId &&
                  (index === 0 ||
                    filteredVersions[index - 1].contract_id !== version.contract_id);
                const isLastOfContract =
                  index === filteredVersions.length - 1 ||
                  filteredVersions[index + 1]?.contract_id !== version.contract_id;

                return (
                  <div
                    key={`${version.contract_id}-${version.version_number}`}
                    className={cn(
                      !selectedContractId && isLastOfContract && "mb-3 last:mb-0"
                    )}
                  >
                    {showContractHeader && (
                      <div className="flex items-center gap-3 px-6 py-4 bg-gray-50/80 border-b border-gray-100">
                        <div
                          className="w-1 h-8 rounded-full"
                          style={{ backgroundColor: accentColor }}
                        />
                        <div className="flex-1">
                          <h4 className="text-base font-semibold text-gray-900 tracking-tight">
                            {version.contractTitle}
                          </h4>
                          <p className="text-sm text-gray-500">
                            {versions.filter((v) => v.contract_id === version.contract_id).length}개 버전
                          </p>
                        </div>
                      </div>
                    )}
                    <Link
                      href={`/analysis/${version.contract_id}?version=${version.version_number}`}
                      className={cn(
                        "flex items-start gap-5 p-6 transition-all duration-200 group border-b border-gray-100",
                        !selectedContractId && "border-l-3",
                        "hover:bg-gray-50/50"
                      )}
                      style={!selectedContractId ? { borderLeftWidth: '3px', borderLeftColor: accentColor } : undefined}
                    >
                      {/* Version Badge */}
                      <div className={cn(
                        "flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center border-2 transition-all duration-200",
                        version.is_current
                          ? "bg-[#3d5a47] text-white border-[#3d5a47]"
                          : "bg-white text-gray-600 border-gray-200 group-hover:border-gray-300"
                      )}>
                        <span className="text-base font-bold">v{version.version_number}</span>
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2 flex-wrap">
                          <span className={cn(
                            "px-3 py-1.5 text-sm font-semibold rounded-lg border",
                            changeInfo.color
                          )}>
                            {changeInfo.label}
                          </span>
                          {version.is_current && (
                            <span className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold rounded-lg bg-[#e8f5ec] text-[#3d7a4a] border border-[#c8e6cf]">
                              <IconCheck size={14} />
                              현재 버전
                            </span>
                          )}
                          <span className="text-sm text-gray-400">
                            {formatRelativeTime(version.created_at)}
                          </span>
                        </div>
                        <p className="text-lg font-medium text-gray-900 tracking-tight mb-1">
                          {version.change_summary || "변경 사항 없음"}
                        </p>
                        <p className="text-base text-gray-500">
                          {version.created_by === "ai" ? "AI" : version.created_by === "system" ? "시스템" : "사용자"}에 의해 {version.created_by === "system" ? "생성됨" : "수정됨"}
                        </p>
                      </div>

                      {/* Arrow */}
                      <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gray-100 group-hover:bg-[#3d5a47] flex items-center justify-center transition-all duration-200">
                        <IconChevronRight size={20} className="text-gray-400 group-hover:text-white transition-colors" />
                      </div>
                    </Link>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      )}

      {/* Timeline Visual (for Desktop) */}
      {filteredVersions.length > 0 && selectedContractId && (
        <section className="hidden sm:block liquid-glass rounded-2xl p-8 border border-white/40">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center">
              <IconHistory size={24} className="text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 tracking-tight">버전 타임라인</h3>
              <p className="text-base text-gray-500">시간순 버전 변경 기록</p>
            </div>
          </div>
          <div className="relative pl-8">
            {/* Timeline line */}
            <div className="absolute left-6 top-0 bottom-0 w-1 bg-gradient-to-b from-[#3d5a47] via-[#4a6b52] to-gray-200 rounded-full" />

            {filteredVersions.map((version, index) => {
              const changeInfo = getChangeTypeInfo(version.created_by, version.change_summary);
              return (
                <div key={`${version.contract_id}-${version.version_number}`} className="relative flex items-start gap-6 pb-8 last:pb-0">
                  {/* Timeline dot */}
                  <div className={cn(
                    "relative z-10 w-14 h-14 rounded-2xl flex items-center justify-center border-4 border-white shadow-lg transition-all",
                    index === 0
                      ? "bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] text-white shadow-[#3d5a47]/30"
                      : "bg-white text-gray-600 shadow-gray-200/50"
                  )}>
                    <span className="text-lg font-bold">v{version.version_number}</span>
                  </div>

                  {/* Content */}
                  <div className="flex-1 pt-2">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={cn(
                        "px-3 py-1.5 text-sm font-semibold rounded-lg border",
                        changeInfo.color
                      )}>
                        {changeInfo.label}
                      </span>
                      <span className="text-base text-gray-400">
                        {formatRelativeTime(version.created_at)}
                      </span>
                    </div>
                    <p className="text-lg text-gray-700 tracking-tight">
                      {version.change_summary || "버전 생성됨"}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
