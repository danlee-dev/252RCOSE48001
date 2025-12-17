"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { authApi, User, contractsApi } from "@/lib/api";
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
  if (SUPPORTED_FORMATS.image.extensions.includes(ext)) return "bg-purple-100 text-purple-500";
  return "bg-gray-100 text-gray-500";
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
          "fixed inset-0 bg-black/20 backdrop-blur-[2px] z-40 transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className={cn(
          "fixed bg-white shadow-2xl z-50 transition-transform duration-300 ease-out",
          "inset-x-0 bottom-0 h-[85vh] rounded-t-3xl",
          "sm:inset-y-0 sm:left-auto sm:right-0 sm:h-full sm:w-[420px] sm:rounded-none",
          isOpen
            ? "translate-y-0 sm:translate-y-0 sm:translate-x-0"
            : "translate-y-full sm:translate-y-0 sm:translate-x-full"
        )}
      >
        {/* Mobile handle */}
        <div className="flex justify-center pt-3 pb-1 sm:hidden">
          <div className="w-10 h-1 bg-gray-300 rounded-full" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-4 sm:px-6 h-14 sm:h-16 border-b border-gray-100">
          <h2 className="text-base font-semibold text-gray-900 tracking-tight">계약서 업로드</h2>
          <button
            onClick={onClose}
            className="w-10 h-10 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all"
          >
            <IconClose size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-6 h-[calc(100%-4rem)] sm:h-[calc(100%-4rem)] overflow-y-auto pb-[env(safe-area-inset-bottom)]">
          {uploadSuccess ? (
            <div className="flex flex-col items-center justify-center h-full animate-scaleIn">
              <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mb-4">
                <IconCheck size={32} className="text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">업로드 완료</h3>
              <p className="text-sm text-gray-500 text-center">
                AI 분석이 시작되었습니다
              </p>
            </div>
          ) : !file ? (
            <div className="space-y-6">
              {/* Upload Area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={cn(
                  "relative group cursor-pointer rounded-2xl transition-all duration-300 overflow-hidden",
                  isDragging
                    ? "ring-2 ring-gray-900 ring-offset-4"
                    : "hover:ring-2 hover:ring-gray-200 hover:ring-offset-2"
                )}
              >
                <div className={cn(
                  "relative p-10 text-center bg-gradient-to-br from-gray-50 via-white to-gray-50 transition-all duration-300",
                  isDragging && "from-gray-100 via-gray-50 to-gray-100"
                )}>
                  <div className="absolute inset-0 opacity-[0.03]">
                    <div className="absolute top-4 left-4 w-24 h-24 border border-gray-900 rounded-2xl" />
                    <div className="absolute bottom-4 right-4 w-16 h-16 border border-gray-900 rounded-xl" />
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-gray-900 rounded-3xl rotate-12" />
                  </div>

                  <div className={cn(
                    "relative inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-5 transition-all duration-300",
                    isDragging
                      ? "bg-gray-900 text-white scale-110"
                      : "bg-gray-100 text-gray-400 group-hover:bg-gray-900 group-hover:text-white group-hover:scale-105"
                  )}>
                    <IconUpload size={28} />
                  </div>

                  <h3 className="text-base font-semibold text-gray-900 mb-2">
                    {isDragging ? "여기에 놓으세요" : "계약서 업로드"}
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    드래그하여 놓거나 클릭하여 선택
                  </p>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-200 rounded-xl shadow-sm hover:bg-gray-50 hover:border-gray-300 transition-all"
                  >
                    <IconDocument size={16} />
                    파일 선택
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPT_STRING}
                    onChange={handleFileSelect}
                    className="hidden"
                  />

                  <p className="text-xs text-gray-400 mt-4">PDF, HWP, Word, TXT, 이미지 (최대 50MB)</p>
                </div>
              </div>

              {/* Supported Formats */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">지원 형식</h4>
                <div className="flex flex-wrap gap-1.5">
                  {["PDF", "HWP", "DOCX", "TXT"].map((fmt) => (
                    <span key={fmt} className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                      {fmt}
                    </span>
                  ))}
                  {["PNG", "JPG"].map((fmt) => (
                    <span key={fmt} className="px-2 py-1 text-xs bg-purple-50 text-purple-600 rounded">
                      {fmt}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 animate-fadeInUp">
              {/* Selected File */}
              <div className="p-5 bg-gray-50 rounded-2xl">
                <div className="flex items-center gap-4">
                  <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0", getFileIconColor(getFileExtension(file.name)))}>
                    <IconDocument size={24} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                      <span className="flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium bg-gray-200 text-gray-600 rounded">
                        {getFileTypeLabel(getFileExtension(file.name))}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={handleReset}
                    disabled={uploading}
                    className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-lg transition-all disabled:opacity-50"
                  >
                    <IconClose size={18} />
                  </button>
                </div>
              </div>

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="w-full flex items-center justify-center gap-2 px-5 py-3.5 text-sm font-medium text-white bg-gray-900 rounded-xl shadow-sm hover:bg-gray-800 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {uploading ? (
                  <>
                    <IconLoading size={18} />
                    분석 중...
                  </>
                ) : (
                  <>
                    <IconUpload size={18} />
                    AI 분석 시작
                  </>
                )}
              </button>

              {/* Change File */}
              <label className="block text-center">
                <span className="text-sm text-gray-500 hover:text-gray-700 cursor-pointer transition-colors">
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

// User Menu Dropdown Component
function UserMenuDropdown({
  user,
  isOpen,
  onClose,
  onLogout,
}: {
  user: User | null;
  isOpen: boolean;
  onClose: () => void;
  onLogout: () => void;
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
      className="absolute right-0 top-full mt-2 w-52 xs:w-56 bg-white rounded-xl border border-gray-200 shadow-strong overflow-hidden animate-fadeInDown z-50"
    >
      <div className="px-4 py-3 border-b border-gray-100">
        <p className="text-sm font-medium text-gray-900 tracking-tight">{user?.username || "사용자"}</p>
        <p className="text-xs text-gray-500 truncate">{user?.email || ""}</p>
      </div>
      <div className="py-1">
        <Link
          href="/settings"
          onClick={onClose}
          className="flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 active:bg-gray-100 transition-colors"
        >
          <IconSettings size={16} className="text-gray-400" />
          설정
        </Link>
        <button
          onClick={() => {
            onLogout();
            onClose();
          }}
          className="w-full flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 active:bg-gray-100 transition-colors"
        >
          <IconLogout size={16} className="text-gray-400" />
          로그아웃
        </button>
      </div>
    </div>
  );
}

interface MainLayoutProps {
  children: React.ReactNode;
  user: User | null;
  onUploadSuccess?: () => void;
  stats?: {
    total: number;
    completed: number;
  };
}

export function MainLayout({ children, user, onUploadSuccess, stats }: MainLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showUploadSidebar, setShowUploadSidebar] = useState(false);
  const [sidebarExpanded, setSidebarExpanded] = useState(false);

  function handleLogout() {
    authApi.logout();
    router.push("/login");
  }

  const navItems = [
    { href: "/", icon: IconHome, label: "대시보드" },
    { href: "/history", icon: IconList, label: "분석 기록" },
    { href: "/certification", icon: IconFileText, label: "내용증명 작성" },
    { href: "/contract-revisions", icon: IconHistory, label: "수정 기록" },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Floating Island Sidebar */}
      <div
        className="hidden lg:block fixed left-4 top-1/2 -translate-y-1/2 z-30"
        onMouseEnter={() => setSidebarExpanded(true)}
        onMouseLeave={() => setSidebarExpanded(false)}
      >
        <div
          className={cn(
            "relative bg-white/70 backdrop-blur-2xl border border-white/50 shadow-2xl overflow-hidden transition-all duration-500 ease-out",
            sidebarExpanded
              ? "w-56 rounded-3xl"
              : "w-16 rounded-2xl"
          )}
          style={{
            boxShadow: sidebarExpanded
              ? '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.5) inset'
              : '0 10px 40px -10px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.5) inset'
          }}
        >
          {/* Logo Area */}
          <div className={cn(
            "flex items-center gap-3 p-4 border-b border-gray-100/50 transition-all duration-300",
            sidebarExpanded ? "justify-start" : "justify-center"
          )}>
            <Logo size={28} color="#111827" className="flex-shrink-0" />
            <span className={cn(
              "text-sm font-semibold text-gray-900 tracking-tight whitespace-nowrap transition-all duration-300",
              sidebarExpanded ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4 absolute"
            )}>
              DocScanner
            </span>
          </div>

          {/* Navigation */}
          <nav className="p-2 space-y-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "group flex items-center h-11 rounded-xl transition-all duration-200 overflow-hidden",
                    sidebarExpanded ? "px-3 gap-3" : "justify-center",
                    isActive
                      ? "bg-gray-900/5 text-gray-900"
                      : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/50"
                  )}
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    <item.icon size={20} className={cn(!isActive && "group-hover:scale-110 transition-transform")} />
                  </div>
                  <span className={cn(
                    "text-sm font-medium tracking-tight whitespace-nowrap transition-all duration-300",
                    sidebarExpanded ? "opacity-100" : "opacity-0 w-0 overflow-hidden"
                  )}>
                    {item.label}
                  </span>
                </Link>
              );
            })}
          </nav>

          {/* Divider */}
          <div className="mx-3 h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent" />

          {/* Quick Stats */}
          {stats && (
            <div className={cn(
              "p-3 transition-all duration-300 overflow-hidden",
              sidebarExpanded ? "opacity-100 max-h-40" : "opacity-0 max-h-0 p-0"
            )}>
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2 px-1">Quick Stats</p>
              <div className="space-y-2">
                <div className="flex items-center justify-between px-2 py-1.5 bg-gray-50/50 rounded-lg">
                  <span className="text-xs text-gray-500">전체</span>
                  <span className="text-sm font-semibold text-gray-900">{stats.total}</span>
                </div>
                <div className="flex items-center justify-between px-2 py-1.5 bg-green-50/50 rounded-lg">
                  <span className="text-xs text-green-600">완료</span>
                  <span className="text-sm font-semibold text-green-700">{stats.completed}</span>
                </div>
              </div>
            </div>
          )}

          {/* User Account */}
          <div className="p-2 border-t border-gray-100/50">
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className={cn(
                  "group flex items-center h-11 w-full rounded-xl text-gray-600 hover:text-gray-900 hover:bg-gray-100/50 transition-all duration-200 overflow-hidden",
                  sidebarExpanded ? "px-3 gap-3" : "justify-center",
                  showUserMenu && "bg-gray-100/50 text-gray-900"
                )}
              >
                <div className="w-7 h-7 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
                  {user?.username?.charAt(0).toUpperCase() || <IconUser size={14} />}
                </div>
                <span className={cn(
                  "text-sm font-medium tracking-tight whitespace-nowrap transition-all duration-300 truncate",
                  sidebarExpanded ? "opacity-100" : "opacity-0 w-0 overflow-hidden"
                )}>
                  {user?.username || "내 계정"}
                </span>
              </button>
              {/* User Menu */}
              {showUserMenu && sidebarExpanded && (
                <div className="absolute left-full bottom-0 ml-2 w-52 bg-white rounded-xl border border-gray-200 shadow-strong overflow-hidden animate-fadeIn z-50">
                  <div className="px-4 py-3 border-b border-gray-100">
                    <p className="text-sm font-medium text-gray-900 tracking-tight">{user?.username || "사용자"}</p>
                    <p className="text-xs text-gray-500 truncate">{user?.email || ""}</p>
                  </div>
                  <div className="py-1">
                    <Link
                      href="/settings"
                      onClick={() => setShowUserMenu(false)}
                      className="flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <IconSettings size={16} className="text-gray-400" />
                      설정
                    </Link>
                    <button
                      onClick={() => {
                        handleLogout();
                        setShowUserMenu(false);
                      }}
                      className="w-full flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <IconLogout size={16} className="text-gray-400" />
                      로그아웃
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Expand Indicator */}
          <div className={cn(
            "absolute right-0 top-1/2 -translate-y-1/2 w-1 h-12 bg-gradient-to-b from-gray-300 via-gray-400 to-gray-300 rounded-full transition-all duration-300",
            sidebarExpanded ? "opacity-0 scale-0" : "opacity-50"
          )} />
        </div>
      </div>

      {/* Minimal Header */}
      <header className={cn(
        "fixed top-0 left-0 right-0 z-20 px-4 sm:px-6 py-4 transition-all duration-500 ease-out",
        sidebarExpanded ? "lg:pl-64" : "lg:pl-24"
      )}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5 lg:hidden">
            <Logo size={32} color="#111827" />
            <span className="text-base font-semibold text-gray-900 tracking-tight">DocScanner AI</span>
          </div>
          <div className="hidden lg:block" />

          {/* User Avatar - Mobile only */}
          <div className="relative lg:hidden">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className={cn(
                "w-9 h-9 rounded-full transition-all duration-200 shadow-sm",
                showUserMenu ? "ring-2 ring-gray-900 ring-offset-2" : "hover:ring-2 hover:ring-gray-200 hover:ring-offset-2"
              )}
            >
              <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-900 rounded-full flex items-center justify-center text-white text-sm font-semibold">
                {user?.username?.charAt(0).toUpperCase() || <IconUser size={16} />}
              </div>
            </button>
            <UserMenuDropdown
              user={user}
              isOpen={showUserMenu}
              onClose={() => setShowUserMenu(false)}
              onLogout={handleLogout}
            />
          </div>
        </div>
      </header>

      {/* Floating Dock */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30">
        <div className="flex items-center gap-1 px-2 py-2 bg-white/80 backdrop-blur-xl border border-gray-200/50 rounded-2xl shadow-lg">
          {/* Scan */}
          <Link
            href="/scan"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/scan"
                ? "text-gray-900 bg-gray-100/80"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconScan size={22} className="group-hover:scale-110 transition-transform" />
            <span className="text-[10px] font-medium mt-1 tracking-tight">스캔</span>
          </Link>

          {/* Checklist */}
          <Link
            href="/checklist"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/checklist"
                ? "text-gray-900 bg-gray-100/80"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconChecklist size={22} className="group-hover:scale-110 transition-transform" />
            <span className="text-[10px] font-medium mt-1 tracking-tight">체크리스트</span>
          </Link>

          {/* Upload - Primary Action */}
          <button
            onClick={() => setShowUploadSidebar(true)}
            className="group flex flex-col items-center justify-center w-16 h-16 -my-1 bg-gray-900 text-white rounded-2xl shadow-lg hover:bg-gray-800 hover:scale-105 active:scale-95 transition-all duration-200"
          >
            <IconPlus size={24} className="group-hover:rotate-90 transition-transform duration-300" />
            <span className="text-[10px] font-medium mt-0.5 tracking-tight">업로드</span>
          </button>

          {/* History */}
          <Link
            href="/history"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/history"
                ? "text-gray-900 bg-gray-100/80"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconHistory size={22} className="group-hover:scale-110 transition-transform" />
            <span className="text-[10px] font-medium mt-1 tracking-tight">기록</span>
          </Link>

          {/* Settings */}
          <Link
            href="/settings"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/settings"
                ? "text-gray-900 bg-gray-100/80"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconSettings size={22} className="group-hover:rotate-45 transition-transform duration-300" />
            <span className="text-[10px] font-medium mt-1 tracking-tight">설정</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <main className={cn(
        "max-w-5xl mx-auto px-4 sm:px-6 pt-20 pb-32 transition-all duration-500 ease-out",
        sidebarExpanded && "lg:pl-56"
      )}>
        {children}
      </main>

      {/* Upload Sidebar */}
      <UploadSidebar
        isOpen={showUploadSidebar}
        onClose={() => setShowUploadSidebar(false)}
        onUploadSuccess={onUploadSuccess || (() => {})}
      />
    </div>
  );
}
