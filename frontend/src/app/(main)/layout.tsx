"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { contractsApi, authApi, User, DocumentVersion } from "@/lib/api";

interface RecentRevision {
  id: number;
  title: string;
  change: string;
  time: string;
  contractId: number;
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
    month: "short",
    day: "numeric",
  });
}
import {
  IconUpload,
  IconDocument,
  IconCheck,
  IconLoading,
  IconLogout,
  IconSettings,
  IconPlus,
  IconUser,
  IconClose,
  IconScan,
  IconChecklist,
  IconHome,
  IconHistory,
  IconList,
  IconFileText,
  Logo,
} from "@/components/icons";
import { cn } from "@/lib/utils";

// Supported file formats
const SUPPORTED_FORMATS = {
  document: {
    extensions: [".pdf", ".hwp", ".hwpx", ".docx", ".doc", ".txt", ".rtf", ".md"],
    mimeTypes: [
      "application/pdf",
      "application/x-hwp",
      "application/haansofthwp",
      "application/vnd.hancom.hwp",
      "application/vnd.hancom.hwpx",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "text/plain",
      "text/markdown",
      "application/rtf",
    ],
  },
  image: {
    extensions: [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"],
    mimeTypes: [
      "image/png",
      "image/jpeg",
      "image/gif",
      "image/webp",
      "image/bmp",
      "image/tiff",
    ],
  },
};

const ALL_EXTENSIONS = [
  ...SUPPORTED_FORMATS.document.extensions,
  ...SUPPORTED_FORMATS.image.extensions,
];

const ACCEPT_STRING = [
  ...ALL_EXTENSIONS,
  ...SUPPORTED_FORMATS.document.mimeTypes,
  ...SUPPORTED_FORMATS.image.mimeTypes,
].join(",");

const MAX_FILE_SIZE = 50 * 1024 * 1024;

function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  return lastDot !== -1 ? filename.substring(lastDot).toLowerCase() : "";
}

function isValidFile(file: File): boolean {
  const ext = getFileExtension(file.name);
  return ALL_EXTENSIONS.includes(ext);
}

function getFileTypeLabel(extension: string): string {
  const ext = extension.toLowerCase();
  if (ext === ".pdf") return "PDF";
  if (ext === ".hwp") return "HWP";
  if (ext === ".hwpx") return "HWPX";
  if ([".docx", ".doc"].includes(ext)) return "Word";
  if ([".txt", ".md", ".rtf"].includes(ext)) return "Text";
  if (SUPPORTED_FORMATS.image.extensions.includes(ext)) return "Image";
  return "File";
}

function getFileIconColor(extension: string): string {
  const ext = extension.toLowerCase();
  if (ext === ".pdf") return "bg-red-100 text-red-500";
  if ([".hwp", ".hwpx"].includes(ext)) return "bg-blue-100 text-blue-500";
  if ([".docx", ".doc"].includes(ext)) return "bg-blue-100 text-blue-600";
  if ([".txt", ".md", ".rtf"].includes(ext)) return "bg-gray-100 text-gray-500";
  if (SUPPORTED_FORMATS.image.extensions.includes(ext)) return "bg-gray-100 text-gray-600";
  return "bg-gray-100 text-gray-500";
}

// User Menu Dropdown Component
function UserMenuDropdown({
  user,
  isOpen,
  onClose,
  onLogout,
  onOpenSettings,
}: {
  user: User | null;
  isOpen: boolean;
  onClose: () => void;
  onLogout: () => void;
  onOpenSettings: () => void;
}) {
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose();
      }
    }
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      ref={dropdownRef}
      className="absolute left-full bottom-0 ml-2 w-52 bg-white rounded-xl border border-gray-200 shadow-strong overflow-hidden animate-fadeIn z-50"
    >
      <div className="px-4 py-3 border-b border-gray-100">
        <p className="text-sm font-medium text-gray-900 tracking-tight">{user?.username || "사용자"}</p>
        <p className="text-xs text-gray-500 truncate">{user?.email || ""}</p>
      </div>
      <div className="py-1">
        <button
          onClick={() => {
            onOpenSettings();
            onClose();
          }}
          className="w-full flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
        >
          <IconSettings size={16} className="text-gray-400" />
          설정
        </button>
        <button
          onClick={() => {
            onLogout();
            onClose();
          }}
          className="w-full flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
        >
          <IconLogout size={16} className="text-gray-400" />
          로그아웃
        </button>
      </div>
    </div>
  );
}

// Settings Modal Component
function SettingsModal({
  isOpen,
  onClose,
  user,
  onLogout,
}: {
  isOpen: boolean;
  onClose: () => void;
  user: User | null;
  onLogout: () => void;
}) {
  const modalRef = useRef<HTMLDivElement>(null);
  const [activeSection, setActiveSection] = useState<"account" | "notification" | "appearance">("account");

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  useEffect(() => {
    function handleEscape(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
    }
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sections = [
    { key: "account" as const, label: "계정", icon: IconUser },
    { key: "notification" as const, label: "알림", icon: IconDocument },
    { key: "appearance" as const, label: "화면", icon: IconSettings },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop with blur */}
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-md animate-fadeIn"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        ref={modalRef}
        className="relative w-full max-w-md mx-4 max-h-[85vh] bg-white rounded-3xl shadow-2xl overflow-hidden animate-scaleIn"
        style={{ animationDuration: "0.2s" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900 tracking-tight">설정</h2>
          <button
            onClick={onClose}
            className="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-xl transition-all"
          >
            <IconClose size={20} />
          </button>
        </div>

        {/* Section Tabs */}
        <div className="flex gap-1 px-4 py-3 bg-gray-50/50 border-b border-gray-100">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.key}
                onClick={() => setActiveSection(section.key)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all",
                  activeSection === section.key
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700 hover:bg-white/50"
                )}
              >
                <Icon size={16} />
                {section.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(85vh-180px)]">
          {activeSection === "account" && (
            <div className="space-y-6 animate-fadeIn">
              {/* Profile Section */}
              <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-2xl">
                <div className="w-14 h-14 bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] rounded-2xl flex items-center justify-center text-white text-xl font-semibold">
                  {user?.username?.charAt(0).toUpperCase() || "U"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-base font-semibold text-gray-900 tracking-tight">{user?.username || "사용자"}</p>
                  <p className="text-sm text-gray-500 truncate">{user?.email || ""}</p>
                </div>
              </div>

              {/* Account Info */}
              <div className="space-y-4">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">계정 정보</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">사용자명</p>
                      <p className="text-sm text-gray-500">{user?.username || "-"}</p>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">이메일</p>
                      <p className="text-sm text-gray-500">{user?.email || "-"}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Danger Zone */}
              <div className="space-y-4 pt-4 border-t border-gray-100">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">계정 관리</h3>
                <button
                  onClick={() => {
                    onLogout();
                    onClose();
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-[#b54a45] bg-[#fdedec] hover:bg-[#fce4e3] border border-[#f5c6c4] rounded-xl transition-all"
                >
                  <IconLogout size={18} />
                  로그아웃
                </button>
              </div>
            </div>
          )}

          {activeSection === "notification" && (
            <div className="space-y-6 animate-fadeIn">
              <div className="space-y-4">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">알림 설정</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">분석 완료 알림</p>
                      <p className="text-xs text-gray-500 mt-0.5">계약서 분석이 완료되면 알림</p>
                    </div>
                    <div className="w-11 h-6 bg-[#3d5a47] rounded-full relative cursor-pointer">
                      <div className="absolute right-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow-sm" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">위험 조항 알림</p>
                      <p className="text-xs text-gray-500 mt-0.5">고위험 조항 발견 시 알림</p>
                    </div>
                    <div className="w-11 h-6 bg-[#3d5a47] rounded-full relative cursor-pointer">
                      <div className="absolute right-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow-sm" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">이메일 알림</p>
                      <p className="text-xs text-gray-500 mt-0.5">중요 알림을 이메일로 받기</p>
                    </div>
                    <div className="w-11 h-6 bg-gray-200 rounded-full relative cursor-pointer">
                      <div className="absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow-sm" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === "appearance" && (
            <div className="space-y-6 animate-fadeIn">
              <div className="space-y-4">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">화면 설정</h3>
                <div className="space-y-3">
                  <div className="p-4 bg-white border border-gray-100 rounded-xl">
                    <p className="text-sm font-medium text-gray-900 tracking-tight mb-3">테마</p>
                    <div className="flex gap-2">
                      <button className="flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium bg-gray-900 text-white rounded-xl">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="5" />
                          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                        </svg>
                        라이트
                      </button>
                      <button className="flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-gray-500 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                        </svg>
                        다크
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl">
                    <div>
                      <p className="text-sm font-medium text-gray-900 tracking-tight">컴팩트 모드</p>
                      <p className="text-xs text-gray-500 mt-0.5">더 많은 정보를 한 화면에</p>
                    </div>
                    <div className="w-11 h-6 bg-gray-200 rounded-full relative cursor-pointer">
                      <div className="absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow-sm" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Version Info */}
              <div className="pt-4 border-t border-gray-100">
                <div className="text-center">
                  <p className="text-xs text-gray-400">DocScanner.ai v1.0.0</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// Upload Sidebar Component
function UploadSidebar({
  isOpen,
  onClose,
  onUploadSuccess,
}: {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isValidFile(droppedFile) && droppedFile.size <= MAX_FILE_SIZE) {
      setFile(droppedFile);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile) && selectedFile.size <= MAX_FILE_SIZE) {
      setFile(selectedFile);
    }
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);

    try {
      await contractsApi.upload(file);
      setUploadSuccess(true);
      setTimeout(() => {
        onUploadSuccess();
        handleReset();
        onClose();
      }, 1500);
    } catch {
      // Error handling
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setUploadSuccess(false);
  };

  useEffect(() => {
    if (!isOpen) {
      setTimeout(() => {
        handleReset();
      }, 300);
    }
  }, [isOpen]);

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 bg-black/30 backdrop-blur-sm z-40 transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className={cn(
          "fixed z-50 transition-transform duration-300 ease-out overflow-hidden",
          "inset-x-0 bottom-0 h-[85vh] rounded-t-3xl",
          "sm:inset-y-0 sm:left-auto sm:right-0 sm:h-full sm:w-[440px] sm:rounded-none",
          "shadow-2xl border-l border-white/50",
          isOpen
            ? "translate-y-0 sm:translate-y-0 sm:translate-x-0"
            : "translate-y-full sm:translate-y-0 sm:translate-x-full"
        )}
      >
        {/* Gradient Background */}
        <div className="absolute inset-0 bg-[#f8f9fa]" />
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `
              radial-gradient(ellipse 80% 60% at 5% 15%, rgba(220, 235, 224, 0.95) 0%, transparent 55%),
              radial-gradient(ellipse 60% 50% at 95% 85%, rgba(254, 243, 210, 0.7) 0%, transparent 55%),
              radial-gradient(ellipse 50% 40% at 60% 5%, rgba(220, 240, 226, 0.8) 0%, transparent 45%)
            `
          }}
        />
        {/* Grain Texture */}
        <div
          className="absolute inset-0 pointer-events-none opacity-30"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='2' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`
          }}
        />

        {/* Mobile handle */}
        <div className="relative z-10 flex justify-center pt-3 pb-1 sm:hidden">
          <div className="w-12 h-1.5 bg-gray-300/60 rounded-full" />
        </div>

        {/* Header */}
        <div className="relative z-10 flex items-center justify-between px-6 sm:px-8 h-16 sm:h-20">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] flex items-center justify-center shadow-lg shadow-[#3d5a47]/20">
              <IconUpload size={22} className="text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 tracking-tight">계약서 업로드</h2>
              <p className="text-sm text-gray-500">AI가 위험 조항을 분석합니다</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-11 h-11 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-xl transition-all"
          >
            <IconClose size={22} />
          </button>
        </div>

        {/* Content */}
        <div className="relative z-10 p-6 sm:p-8 h-[calc(100%-5rem)] sm:h-[calc(100%-5rem)] overflow-y-auto pb-[env(safe-area-inset-bottom)]">
          {uploadSuccess ? (
            <div className="flex flex-col items-center justify-center h-full animate-scaleIn">
              <div className="relative">
                <div className="w-24 h-24 bg-gradient-to-br from-[#e8f5ec] to-white rounded-3xl flex items-center justify-center border border-[#c8e6cf] shadow-xl shadow-[#3d5a47]/10">
                  <IconCheck size={48} className="text-[#3d7a4a]" />
                </div>
                <div className="absolute -inset-3 rounded-[28px] border-2 border-[#3d5a47]/20 animate-pulse" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mt-8 mb-3 tracking-tight">업로드 완료</h3>
              <p className="text-base text-gray-500 text-center max-w-xs">
                AI 분석이 시작되었습니다. 잠시 후 결과를 확인할 수 있습니다.
              </p>
            </div>
          ) : !file ? (
            <div className="space-y-8">
              {/* Upload Area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={cn(
                  "relative group cursor-pointer rounded-2xl transition-all duration-300 overflow-hidden",
                  "bg-white/60 backdrop-blur-md",
                  "border border-white/60 shadow-[inset_0_1px_1px_rgba(255,255,255,0.8),0_4px_24px_rgba(0,0,0,0.04)]",
                  isDragging
                    ? "bg-[#e8f0ea]/70 scale-[1.02] shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_8px_32px_rgba(61,90,71,0.15)] border-[#c8e6cf]/80"
                    : "hover:shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_8px_32px_rgba(0,0,0,0.08)] hover:border-white/80"
                )}
              >
                <div className={cn(
                  "relative p-12 text-center transition-all duration-300",
                  isDragging && "bg-gradient-to-br from-[#e8f0ea]/60 to-transparent"
                )}>
                  <div className={cn(
                    "relative inline-flex items-center justify-center w-20 h-20 rounded-2xl mb-6 transition-all duration-300 shadow-lg",
                    isDragging
                      ? "bg-gradient-to-br from-[#3d5a47] to-[#4a6b52] text-white scale-110 shadow-[#3d5a47]/30"
                      : "bg-gradient-to-br from-[#e8f0ea] to-white text-[#3d5a47] border border-[#c8e6cf] group-hover:from-[#3d5a47] group-hover:to-[#4a6b52] group-hover:text-white group-hover:scale-105 group-hover:shadow-[#3d5a47]/30"
                  )}>
                    <IconUpload size={32} />
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 mb-2 tracking-tight">
                    {isDragging ? "여기에 놓으세요" : "파일을 드래그하세요"}
                  </h3>
                  <p className="text-base text-gray-500 mb-6">
                    또는 클릭하여 파일을 선택하세요
                  </p>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="inline-flex items-center gap-2.5 px-6 py-3 text-base font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl shadow-lg shadow-[#3d5a47]/20 hover:shadow-xl hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 transition-all"
                  >
                    <IconDocument size={18} />
                    파일 선택
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPT_STRING}
                    onChange={handleFileSelect}
                    className="hidden"
                  />

                  <p className="text-sm text-gray-400 mt-6">PDF, HWP, Word, TXT, 이미지 (최대 50MB)</p>
                </div>
              </div>

              {/* Supported Formats */}
              <div className="liquid-glass rounded-[16px] p-6 border border-white/40">
                <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-4">지원 형식</h4>
                <div className="flex flex-wrap gap-2">
                  {["PDF", "HWP", "HWPX", "DOCX", "TXT"].map((fmt) => (
                    <span key={fmt} className="px-3 py-1.5 text-sm font-medium bg-[#e8f0ea] text-[#3d5a47] rounded-[8px] border border-[#c8e6cf]">
                      {fmt}
                    </span>
                  ))}
                  {["PNG", "JPG", "WEBP"].map((fmt) => (
                    <span key={fmt} className="px-3 py-1.5 text-sm font-medium bg-gray-100 text-gray-600 rounded-[8px] border border-gray-200">
                      {fmt}
                    </span>
                  ))}
                </div>
              </div>

              {/* Process Info */}
              <div className="liquid-glass rounded-2xl p-6 border border-white/40">
                <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-4">분석 과정</h4>
                <div className="space-y-4">
                  {[
                    { step: 1, text: "문서 파싱 및 텍스트 추출", icon: IconDocument },
                    { step: 2, text: "AI가 위험 조항 식별", icon: IconScan },
                    { step: 3, text: "법률 근거 기반 분석", icon: IconCheck },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <div key={item.step} className="flex items-center gap-4 text-base text-gray-600">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#e8f0ea] to-white flex items-center justify-center text-[#3d5a47] border border-[#c8e6cf]/50 shadow-sm">
                          <Icon size={18} />
                        </div>
                        <span className="font-medium">{item.text}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-8 animate-fadeInUp">
              {/* Selected File */}
              <div className="liquid-glass rounded-2xl p-6 border border-white/40">
                <div className="flex items-center gap-5">
                  <div className={cn("w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-md", getFileIconColor(getFileExtension(file.name)))}>
                    <IconDocument size={28} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="text-lg font-semibold text-gray-900 truncate tracking-tight">{file.name}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="px-2.5 py-1 text-sm font-medium bg-gray-100 text-gray-600 rounded-lg">
                        {getFileTypeLabel(getFileExtension(file.name))}
                      </span>
                      <span className="text-sm text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={handleReset}
                    disabled={uploading}
                    className="w-11 h-11 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-xl transition-all disabled:opacity-50"
                  >
                    <IconClose size={20} />
                  </button>
                </div>
              </div>

              {/* Upload Info */}
              <div className="bg-gradient-to-r from-[#e8f0ea] to-[#d8e8dc] rounded-2xl p-5 border border-[#c8e6cf]">
                <div className="flex items-start gap-4">
                  <div className="w-11 h-11 rounded-xl bg-[#3d5a47] flex items-center justify-center flex-shrink-0">
                    <IconScan size={20} className="text-white" />
                  </div>
                  <div>
                    <p className="text-base font-semibold text-[#3d5a47] mb-1">AI 분석 준비 완료</p>
                    <p className="text-sm text-[#3d7a4a] leading-relaxed">버튼을 클릭하면 AI가 계약서의 위험 조항을 분석합니다.</p>
                  </div>
                </div>
              </div>

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="w-full flex items-center justify-center gap-3 px-6 py-4 text-lg font-semibold text-white bg-gradient-to-r from-[#3d5a47] to-[#4a6b52] rounded-xl shadow-lg shadow-[#3d5a47]/20 hover:shadow-xl hover:shadow-[#3d5a47]/30 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 transition-all"
              >
                {uploading ? (
                  <>
                    <IconLoading size={22} />
                    분석 중...
                  </>
                ) : (
                  <>
                    <IconUpload size={22} />
                    AI 분석 시작
                  </>
                )}
              </button>

              {/* Change File */}
              <label className="block text-center">
                <span className="text-base text-gray-500 hover:text-[#3d5a47] cursor-pointer transition-colors font-medium">
                  다른 파일 선택
                </span>
                <input
                  type="file"
                  accept={ACCEPT_STRING}
                  onChange={handleFileSelect}
                  className="hidden"
                  disabled={uploading}
                />
              </label>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showUploadSidebar, setShowUploadSidebar] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [recentRevisions, setRecentRevisions] = useState<RecentRevision[]>([]);

  useEffect(() => {
    if (!authApi.isAuthenticated()) {
      router.push("/login");
      return;
    }
    setIsAuthenticated(true);
    loadUser();
    loadRecentRevisions();
    setIsLoading(false);
  }, [router]);

  // Listen for custom event to open upload sidebar from child pages
  useEffect(() => {
    function handleOpenUpload() {
      setShowUploadSidebar(true);
    }
    window.addEventListener("openUploadSidebar", handleOpenUpload);
    return () => window.removeEventListener("openUploadSidebar", handleOpenUpload);
  }, []);

  async function loadUser() {
    try {
      const userData = await authApi.getMe();
      setUser(userData);
    } catch {
      // Silently fail
    }
  }

  async function loadRecentRevisions() {
    try {
      const contractData = await contractsApi.list(0, 20);
      interface RawRevision {
        id: number;
        title: string;
        change: string;
        time: string;
        contractId: number;
        createdAt: string;
      }
      const rawRevisions: RawRevision[] = [];

      for (const contract of contractData.items) {
        try {
          const versionData = await contractsApi.getVersions(contract.id);
          for (const version of versionData.versions) {
            rawRevisions.push({
              id: version.id,
              title: contract.title,
              change: version.change_summary || "원본 저장",
              time: formatRelativeTime(version.created_at),
              createdAt: version.created_at,
              contractId: contract.id,
            });
          }
        } catch {
          // Skip contracts without versions
        }
      }

      // Sort by most recent and take top 3
      rawRevisions.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
      const top3 = rawRevisions.slice(0, 3);

      // Check for duplicate titles and add date suffix
      const titleCounts: Record<string, number> = {};
      top3.forEach((r) => {
        titleCounts[r.title] = (titleCounts[r.title] || 0) + 1;
      });

      const finalRevisions: RecentRevision[] = top3.map((r) => {
        let displayTitle = r.title;
        if (titleCounts[r.title] > 1) {
          const date = new Date(r.createdAt);
          const dateStr = `${date.getMonth() + 1}/${date.getDate()}`;
          displayTitle = `${r.title} (${dateStr})`;
        }
        return {
          id: r.id,
          title: displayTitle,
          change: r.change,
          time: r.time,
          contractId: r.contractId,
        };
      });

      setRecentRevisions(finalRevisions);
    } catch {
      // Silently fail
    }
  }

  function handleLogout() {
    authApi.logout();
    router.push("/login");
  }

  function handleUploadSuccess() {
    // Refresh the page or notify child components
    window.location.reload();
  }


  const navItems = [
    { href: "/", icon: IconHome, label: "대시보드" },
    { href: "/history", icon: IconList, label: "분석 기록" },
    { href: "/certification", icon: IconFileText, label: "내용증명 작성" },
    { href: "/contract-revisions", icon: IconHistory, label: "수정 기록" },
  ];


  if (isLoading || !isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#fafafa]">
        <div className="flex flex-col items-center gap-3 animate-fadeIn">
          <IconLoading size={32} className="text-gray-400" />
          <p className="text-sm text-gray-500">불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen gradient-blob-bg">
      {/* Icon Dock Sidebar - Desktop Only */}
      <nav className="hidden lg:flex lg:flex-col fixed left-8 top-1/2 -translate-y-1/2 z-30 icon-dock">
        {/* Logo */}
        <Link href="/" className="dock-icon mb-2">
          <Logo size={24} color="#1a1a1a" />
        </Link>

        {/* Divider */}
        <div className="w-6 h-px bg-gray-200/50 my-1" />

        {/* Navigation */}
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn("dock-icon", isActive && "active")}
              title={item.label}
            >
              <Icon size={22} />
            </Link>
          );
        })}

        {/* Divider */}
        <div className="w-6 h-px bg-gray-200/50 my-1" />

        {/* Upload Button */}
        <button
          onClick={() => setShowUploadSidebar(true)}
          className="dock-icon hover:text-[#3d5a47]"
          title="업로드"
        >
          <IconPlus size={22} />
        </button>

        {/* Settings */}
        <button
          onClick={() => setShowSettingsModal(true)}
          className="dock-icon"
          title="설정"
        >
          <IconSettings size={22} />
        </button>

        {/* User */}
        <button
          onClick={() => setShowUserMenu(!showUserMenu)}
          className="dock-icon relative"
          title={user?.username || "내 계정"}
        >
          <div className="w-7 h-7 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center text-white text-xs font-semibold">
            {user?.username?.charAt(0).toUpperCase() || <IconUser size={14} />}
          </div>
        </button>

        {/* User Menu Dropdown */}
        {showUserMenu && (
          <div className="absolute left-full bottom-0 ml-3">
            <UserMenuDropdown
              user={user}
              isOpen={showUserMenu}
              onClose={() => setShowUserMenu(false)}
              onLogout={handleLogout}
              onOpenSettings={() => setShowSettingsModal(true)}
            />
          </div>
        )}
      </nav>

      {/* Mobile Header */}
      <header className="fixed top-0 left-0 right-0 z-20 px-4 sm:px-6 py-4 lg:hidden">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Logo size={36} color="#111827" />
            <span className="text-lg font-semibold text-gray-900 tracking-tight">DocScanner AI</span>
          </div>
          {/* Hamburger Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(true)}
            className="w-10 h-10 flex items-center justify-center rounded-xl bg-white/80 backdrop-blur-sm border border-gray-200/50 shadow-sm hover:bg-white transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        </div>
      </header>

      {/* Mobile Slide-in Menu */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-sm animate-fadeIn"
            onClick={() => setMobileMenuOpen(false)}
          />
          {/* Sidebar */}
          <div className="absolute left-0 top-0 bottom-0 w-72 bg-[#e8f0ea] shadow-2xl animate-slideInLeft">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-[#c8e6cf]/50">
              <div className="flex items-center gap-2.5">
                <Logo size={28} color="#111827" />
                <span className="text-sm font-semibold text-gray-900 tracking-tight">DocScanner</span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="w-9 h-9 flex items-center justify-center rounded-xl hover:bg-white/50 transition-colors"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Navigation */}
            <nav className="p-3 space-y-1">
              {navItems.map((item) => {
                const isActive = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={cn(
                      "flex items-center gap-3 h-12 px-3 rounded-xl transition-all duration-200",
                      isActive
                        ? "bg-white/70 text-gray-900 shadow-sm"
                        : "text-gray-600 hover:text-gray-900 hover:bg-white/50"
                    )}
                  >
                    <Icon size={20} />
                    <span className="text-sm font-medium tracking-tight">{item.label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* Divider */}
            <div className="mx-4 my-2 border-t border-[#c8e6cf]/50" />

            {/* Recent Revisions Preview */}
            <div className="px-4 py-2">
              <p className="text-[11px] font-medium text-gray-500 uppercase tracking-wider mb-2">최근 수정</p>
              <div className="space-y-2">
                {recentRevisions.length > 0 ? (
                  recentRevisions.map((rev) => (
                    <Link
                      key={rev.id}
                      href={`/contract-revisions?contract=${rev.contractId}`}
                      onClick={() => setMobileMenuOpen(false)}
                      className="block p-2.5 bg-white/50 rounded-xl hover:bg-white/70 transition-colors"
                    >
                      <p className="text-xs font-medium text-gray-800 truncate">{rev.title}</p>
                      <p className="text-[10px] text-gray-500 mt-0.5">{rev.change}</p>
                      <p className="text-[10px] text-gray-400 mt-0.5">{rev.time}</p>
                    </Link>
                  ))
                ) : (
                  <p className="text-xs text-gray-400 text-center py-2">수정 기록이 없습니다</p>
                )}
              </div>
            </div>

            {/* User Section */}
            <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-[#c8e6cf]/50 bg-white/30">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gray-900 rounded-xl flex items-center justify-center text-white text-sm font-medium">
                  {user?.username?.[0]?.toUpperCase() || "U"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate tracking-tight">
                    {user?.username || "User"}
                  </p>
                  <p className="text-[11px] text-gray-500 truncate">{user?.email}</p>
                </div>
                <button
                  onClick={handleLogout}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-white/50 rounded-lg transition-colors"
                  title="로그아웃"
                >
                  <IconLogout size={18} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mobile Floating Dock - Only visible on mobile */}
      <div className="lg:hidden fixed bottom-6 left-1/2 -translate-x-1/2 z-30">
        <div className="flex items-center gap-1 px-2 py-2 bg-white/90 backdrop-blur-xl border border-black/[0.04] rounded-2xl shadow-lg">
          <Link
            href="/scan"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/scan"
                ? "bg-gray-100 text-gray-900"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconScan size={22} />
            <span className="text-[10px] font-medium mt-1 tracking-tight">스캔</span>
          </Link>

          <button
            onClick={() => setShowUploadSidebar(true)}
            className="group flex flex-col items-center justify-center w-16 h-16 -my-1 bg-[#3d5a47] text-white rounded-2xl shadow-lg hover:bg-[#4a6b52] hover:scale-105 active:scale-95 transition-all duration-200"
          >
            <IconPlus size={24} className="group-hover:rotate-90 transition-transform duration-300" />
            <span className="text-[10px] font-medium mt-0.5 tracking-tight">업로드</span>
          </button>

          <Link
            href="/checklist"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/checklist"
                ? "bg-gray-100 text-gray-900"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconChecklist size={22} />
            <span className="text-[10px] font-medium mt-1 tracking-tight">체크리스트</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <main className="relative z-10 lg:ml-24 max-w-[1800px] mx-auto px-4 sm:px-6 lg:px-12 pt-20 lg:pt-10 pb-32 lg:pb-10">
        {children}
      </main>

      {/* Upload Sidebar */}
      <UploadSidebar
        isOpen={showUploadSidebar}
        onClose={() => setShowUploadSidebar(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        user={user}
        onLogout={handleLogout}
      />
    </div>
  );
}
