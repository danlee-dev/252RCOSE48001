"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import {
  IconArrowLeft,
  IconCheck,
  IconWarning,
  IconInfo,
  IconChevronDown,
} from "@/components/icons";
import { cn } from "@/lib/utils";

// Mock checklist data - will be replaced with API data
const MOCK_CHECKLIST_CATEGORIES = [
  {
    id: "basic",
    title: "기본 근로조건",
    description: "근로계약의 필수 사항",
    icon: "contract",
    items: [
      {
        id: "basic-1",
        text: "근로계약 기간이 명시되어 있는가?",
        importance: "high",
        tip: "기간제 근로자의 경우 2년을 초과할 수 없습니다.",
      },
      {
        id: "basic-2",
        text: "근무 장소와 업무 내용이 구체적으로 기재되어 있는가?",
        importance: "high",
        tip: "업무 내용이 모호하면 나중에 부당한 업무 지시를 받을 수 있습니다.",
      },
      {
        id: "basic-3",
        text: "근로시간(시작/종료)이 명확히 기재되어 있는가?",
        importance: "high",
        tip: "법정 근로시간은 1일 8시간, 1주 40시간입니다.",
      },
      {
        id: "basic-4",
        text: "휴게시간이 명시되어 있는가?",
        importance: "medium",
        tip: "4시간 근무 시 30분, 8시간 근무 시 1시간의 휴게시간이 보장되어야 합니다.",
      },
    ],
  },
  {
    id: "wage",
    title: "임금 관련",
    description: "급여 및 수당 사항",
    icon: "money",
    items: [
      {
        id: "wage-1",
        text: "기본급이 최저임금 이상인가?",
        importance: "high",
        tip: "2024년 최저시급은 9,860원입니다.",
      },
      {
        id: "wage-2",
        text: "급여 지급일과 지급 방법이 명시되어 있는가?",
        importance: "high",
        tip: "매월 1회 이상 일정한 날짜에 지급되어야 합니다.",
      },
      {
        id: "wage-3",
        text: "연장/야간/휴일 근로수당 지급 기준이 명시되어 있는가?",
        importance: "medium",
        tip: "연장/야간/휴일 근로 시 통상임금의 50% 이상 가산되어야 합니다.",
      },
      {
        id: "wage-4",
        text: "상여금, 성과급 등 부가 급여 조건이 명확한가?",
        importance: "low",
        tip: "지급 조건과 산정 기준을 명확히 확인하세요.",
      },
    ],
  },
  {
    id: "leave",
    title: "휴가 및 휴일",
    description: "연차, 휴일 관련 사항",
    icon: "calendar",
    items: [
      {
        id: "leave-1",
        text: "주휴일이 보장되어 있는가?",
        importance: "high",
        tip: "1주 15시간 이상 근무 시 유급 주휴일이 보장됩니다.",
      },
      {
        id: "leave-2",
        text: "연차휴가 부여 기준이 명시되어 있는가?",
        importance: "medium",
        tip: "1년 미만 근무자는 1개월 개근 시 1일의 유급휴가가 발생합니다.",
      },
      {
        id: "leave-3",
        text: "경조사 휴가 규정이 있는가?",
        importance: "low",
        tip: "법적 의무는 아니지만 많은 회사에서 제공합니다.",
      },
    ],
  },
  {
    id: "termination",
    title: "계약 종료",
    description: "퇴직, 해고 관련 사항",
    icon: "exit",
    items: [
      {
        id: "term-1",
        text: "해고 사유와 절차가 명시되어 있는가?",
        importance: "high",
        tip: "정당한 사유 없는 해고는 무효입니다.",
      },
      {
        id: "term-2",
        text: "퇴직금 지급 기준이 명시되어 있는가?",
        importance: "high",
        tip: "1년 이상 근무 시 30일분 이상의 평균임금이 지급되어야 합니다.",
      },
      {
        id: "term-3",
        text: "경업금지 조항이 있다면 기간과 범위가 합리적인가?",
        importance: "medium",
        tip: "과도한 경업금지 조항은 법적으로 무효가 될 수 있습니다.",
      },
    ],
  },
  {
    id: "others",
    title: "기타 확인사항",
    description: "추가 확인이 필요한 사항",
    icon: "shield",
    items: [
      {
        id: "other-1",
        text: "4대 보험 가입이 명시되어 있는가?",
        importance: "high",
        tip: "국민연금, 건강보험, 고용보험, 산재보험 가입은 의무입니다.",
      },
      {
        id: "other-2",
        text: "위약금/손해배상 조항이 과도하지 않은가?",
        importance: "high",
        tip: "근로자에게 부당한 손해배상을 예정하는 계약은 무효입니다.",
      },
      {
        id: "other-3",
        text: "교육비 반환 조항이 있다면 합리적인가?",
        importance: "medium",
        tip: "실제 교육비용을 초과하거나 근속 의무 기간이 과도하면 무효가 될 수 있습니다.",
      },
    ],
  },
];

function ImportanceBadge({ importance }: { importance: string }) {
  if (importance === "high") {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold bg-red-50 text-red-600 rounded-full whitespace-nowrap flex-shrink-0">
        <IconWarning size={11} />
        필수
      </span>
    );
  }
  if (importance === "medium") {
    return (
      <span className="inline-flex items-center px-2.5 py-1 text-[11px] font-semibold bg-amber-50 text-amber-600 rounded-full whitespace-nowrap flex-shrink-0">
        중요
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2.5 py-1 text-[11px] font-medium bg-gray-100 text-gray-500 rounded-full whitespace-nowrap flex-shrink-0">
      권장
    </span>
  );
}

function CircularProgress({ progress, size = 40 }: { progress: number; size?: number }) {
  const strokeWidth = 3;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-gray-100"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={cn(
            "transition-all duration-500 ease-out",
            progress === 100 ? "text-green-500" : "text-gray-900"
          )}
        />
      </svg>
      {progress === 100 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <IconCheck size={14} className="text-green-500" />
        </div>
      )}
    </div>
  );
}

export default function ChecklistPage() {
  const router = useRouter();
  const [checkedItems, setCheckedItems] = useState<Set<string>>(new Set());
  const [expandedTips, setExpandedTips] = useState<Set<string>>(new Set());
  const [recentlyChecked, setRecentlyChecked] = useState<string | null>(null);

  useEffect(() => {
    if (!authApi.isAuthenticated()) {
      router.push("/login");
    }
  }, [router]);

  const toggleItem = (itemId: string) => {
    setCheckedItems((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
        setRecentlyChecked(null);
      } else {
        newSet.add(itemId);
        setRecentlyChecked(itemId);
        setTimeout(() => setRecentlyChecked(null), 600);
      }
      return newSet;
    });
  };

  const toggleTip = (itemId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedTips((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const totalItems = MOCK_CHECKLIST_CATEGORIES.reduce(
    (acc, cat) => acc + cat.items.length,
    0
  );
  const checkedCount = checkedItems.size;
  const progress = totalItems > 0 ? (checkedCount / totalItems) * 100 : 0;

  return (
    <div className="min-h-[100dvh] bg-gray-50/50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200/80 sticky top-0 z-10">
        <div className="px-4 sm:px-5 h-14 flex items-center gap-3 max-w-2xl mx-auto">
          <Link
            href="/"
            className="w-10 h-10 flex items-center justify-center -ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-xl transition-all duration-200"
          >
            <IconArrowLeft size={18} />
          </Link>
          <div className="flex-1 min-w-0">
            <h1 className="text-base font-semibold text-gray-900 tracking-tight">
              고용계약 체크리스트
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-500">
              {checkedCount}/{totalItems}
            </span>
            <CircularProgress progress={progress} size={36} />
          </div>
        </div>
      </header>

      {/* Progress Summary */}
      <div className="bg-white border-b border-gray-100">
        <div className="px-4 sm:px-5 py-4 max-w-2xl mx-auto">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500 ease-out",
                    progress === 100
                      ? "bg-green-500"
                      : "bg-gradient-to-r from-gray-800 to-gray-600"
                  )}
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <span className={cn(
              "text-sm font-semibold tabular-nums transition-colors",
              progress === 100 ? "text-green-600" : "text-gray-900"
            )}>
              {Math.round(progress)}%
            </span>
          </div>
          {progress === 100 && (
            <div className="mt-3 flex items-center gap-2 text-green-600 animate-fadeIn">
              <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center">
                <IconCheck size={12} />
              </div>
              <p className="text-sm font-medium">모든 항목을 확인했습니다</p>
            </div>
          )}
        </div>
      </div>

      {/* Checklist */}
      <main className="px-4 sm:px-5 py-5 sm:py-6 max-w-2xl mx-auto">
        <div className="space-y-5">
          {MOCK_CHECKLIST_CATEGORIES.map((category, categoryIndex) => {
            const categoryCheckedCount = category.items.filter((item) =>
              checkedItems.has(item.id)
            ).length;
            const categoryProgress = (categoryCheckedCount / category.items.length) * 100;
            const isCategoryComplete = categoryProgress === 100;

            return (
              <div
                key={category.id}
                className="animate-fadeInUp"
                style={{ animationDelay: `${categoryIndex * 50}ms` }}
              >
                {/* Category Header */}
                <div className="flex items-center justify-between mb-3 px-1">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "w-8 h-8 rounded-xl flex items-center justify-center transition-colors duration-300",
                      isCategoryComplete ? "bg-green-100" : "bg-gray-100"
                    )}>
                      {isCategoryComplete ? (
                        <IconCheck size={16} className="text-green-600" />
                      ) : (
                        <span className="text-xs font-bold text-gray-400">
                          {categoryIndex + 1}
                        </span>
                      )}
                    </div>
                    <div>
                      <h2 className="text-sm font-semibold text-gray-900 tracking-tight">
                        {category.title}
                      </h2>
                      <p className="text-xs text-gray-500">{category.description}</p>
                    </div>
                  </div>
                  <div className={cn(
                    "text-xs font-medium px-2.5 py-1 rounded-full transition-colors",
                    isCategoryComplete
                      ? "bg-green-50 text-green-600"
                      : "bg-gray-100 text-gray-500"
                  )}>
                    {categoryCheckedCount}/{category.items.length}
                  </div>
                </div>

                {/* Items */}
                <div className="bg-white rounded-2xl border border-gray-200/80 shadow-sm overflow-hidden">
                  {category.items.map((item, itemIndex) => {
                    const isChecked = checkedItems.has(item.id);
                    const isTipExpanded = expandedTips.has(item.id);
                    const isJustChecked = recentlyChecked === item.id;

                    return (
                      <div
                        key={item.id}
                        className={cn(
                          "transition-colors duration-200",
                          itemIndex !== 0 && "border-t border-gray-100",
                          isChecked && "bg-green-50/30"
                        )}
                      >
                        <div
                          onClick={() => toggleItem(item.id)}
                          className={cn(
                            "flex items-start gap-3 p-4 cursor-pointer group",
                            "hover:bg-gray-50/50 active:bg-gray-100/50 transition-colors"
                          )}
                        >
                          {/* Checkbox */}
                          <div
                            className={cn(
                              "flex-shrink-0 w-6 h-6 rounded-[8px] border-2 flex items-center justify-center transition-all duration-200 mt-0.5",
                              isChecked
                                ? "bg-green-500 border-green-500 text-white"
                                : "border-gray-300 group-hover:border-gray-400",
                              isJustChecked && "scale-110"
                            )}
                          >
                            <IconCheck
                              size={14}
                              className={cn(
                                "transition-all duration-200",
                                isChecked ? "opacity-100 scale-100" : "opacity-0 scale-50"
                              )}
                            />
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start gap-3">
                              <p
                                className={cn(
                                  "text-sm tracking-tight leading-relaxed transition-all duration-200 flex-1",
                                  isChecked
                                    ? "text-gray-400 line-through"
                                    : "text-gray-800"
                                )}
                              >
                                {item.text}
                              </p>
                              <ImportanceBadge importance={item.importance} />
                            </div>
                          </div>
                        </div>

                        {/* Tip Section */}
                        <div className="px-4 pb-3 -mt-1">
                          <button
                            onClick={(e) => toggleTip(item.id, e)}
                            className={cn(
                              "flex items-center gap-1.5 text-xs font-medium transition-colors ml-9",
                              isTipExpanded
                                ? "text-blue-600"
                                : "text-gray-400 hover:text-gray-600"
                            )}
                          >
                            <IconInfo size={13} />
                            <span>{isTipExpanded ? "팁 숨기기" : "팁 보기"}</span>
                            <IconChevronDown
                              size={12}
                              className={cn(
                                "transition-transform duration-200",
                                isTipExpanded && "rotate-180"
                              )}
                            />
                          </button>

                          {/* Tip Content */}
                          <div
                            className={cn(
                              "overflow-hidden transition-all duration-300 ml-9",
                              isTipExpanded ? "max-h-40 opacity-100 mt-2" : "max-h-0 opacity-0"
                            )}
                          >
                            <div className="p-3 bg-blue-50/70 rounded-xl border border-blue-100/50">
                              <p className="text-xs text-blue-700 leading-relaxed">
                                {item.tip}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom Actions */}
        <div className="mt-8 pb-6">
          <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-5 text-white shadow-lg">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-white/10 backdrop-blur rounded-xl flex items-center justify-center flex-shrink-0">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14,2 14,8 20,8" />
                  <path d="M9 15l2 2 4-4" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-base font-semibold tracking-tight">
                  계약서 분석이 필요하신가요?
                </h3>
                <p className="text-sm text-gray-300 mt-1 mb-4 leading-relaxed">
                  AI가 계약서의 위험 조항을 자동으로 분석해드립니다
                </p>
                <Link
                  href="/"
                  className="inline-flex items-center gap-2 px-4 py-2.5 bg-white text-gray-900 text-sm font-semibold rounded-xl hover:bg-gray-100 active:scale-[0.98] transition-all duration-200 min-h-[44px]"
                >
                  계약서 분석하기
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
