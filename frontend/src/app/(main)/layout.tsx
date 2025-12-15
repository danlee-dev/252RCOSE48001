"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { contractsApi, authApi, User } from "@/lib/api";
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
      className="absolute left-full bottom-0 ml-2 w-52 bg-white rounded-xl border border-gray-200 shadow-strong overflow-hidden animate-fadeIn z-50"
    >
      <div className="px-4 py-3 border-b border-gray-100">
        <p className="text-sm font-medium text-gray-900 tracking-tight">{user?.username || "사용자"}</p>
        <p className="text-xs text-gray-500 truncate">{user?.email || ""}</p>
      </div>
      <div className="py-1">
        <Link
          href="/settings"
          onClick={onClose}
          className="flex items-center gap-2 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
        >
          <IconSettings size={16} className="text-gray-400" />
          설정
        </Link>
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

              {/* Process Info */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">분석 과정</h4>
                <div className="space-y-2">
                  {[
                    "문서 파싱 및 텍스트 추출",
                    "AI가 위험 조항 식별",
                    "법률 근거 기반 분석",
                  ].map((text, i) => (
                    <div key={i} className="flex items-center gap-3 text-sm text-gray-500">
                      <span className="w-5 h-5 rounded-[4px] bg-gray-100 flex items-center justify-center text-xs font-medium text-gray-600">
                        {i + 1}
                      </span>
                      {text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 animate-fadeInUp">
              {/* Selected File */}
              <div className="p-5 bg-gray-50 rounded-2xl">
                <div className="flex items-center gap-4">
                  <div className={cn("w-12 h-12 rounded-[12px] flex items-center justify-center flex-shrink-0", getFileIconColor(getFileExtension(file.name)))}>
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
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    if (!authApi.isAuthenticated()) {
      router.push("/login");
      return;
    }
    setIsAuthenticated(true);
    loadUser();
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

  // Mock recent revisions for sidebar preview
  const recentRevisions = [
    { id: 1, title: "근로계약서_2024", change: "위약금 조항 수정", time: "30분 전" },
    { id: 2, title: "임대차계약서", change: "원상복구 조항 추가", time: "1일 전" },
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
    <div className="min-h-screen bg-[#fafafa]">
      {/* Floating Island Sidebar */}
      <div
        className="hidden lg:block fixed left-4 top-1/2 -translate-y-1/2 z-30"
        onMouseEnter={() => setSidebarExpanded(true)}
        onMouseLeave={() => setSidebarExpanded(false)}
      >
        <div
          className={cn(
            "relative bg-[#e8f0ea]/90 backdrop-blur-2xl border border-[#c8e6cf]/50 shadow-2xl overflow-hidden transition-all duration-500 ease-out",
            sidebarExpanded ? "w-56 rounded-3xl" : "w-16 rounded-2xl"
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
              const Icon = item.icon;
              const isRevisionItem = item.href === "/contract-revisions";
              return (
                <div key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      "group flex items-center h-11 rounded-xl transition-all duration-200 overflow-hidden",
                      isActive
                        ? "bg-gray-900/5 text-gray-900"
                        : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/50",
                      sidebarExpanded ? "px-3 gap-3" : "justify-center"
                    )}
                  >
                    <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                      <Icon size={20} className={!isActive ? "group-hover:scale-110 transition-transform" : ""} />
                    </div>
                    <span className={cn(
                      "text-sm font-medium tracking-tight whitespace-nowrap transition-all duration-300",
                      sidebarExpanded ? "opacity-100" : "opacity-0 w-0 overflow-hidden"
                    )}>
                      {item.label}
                    </span>
                  </Link>
                  {/* Revision Preview - only show when sidebar is expanded */}
                  {isRevisionItem && (
                    <div
                      className={cn(
                        "overflow-hidden transition-all duration-300 ease-out",
                        sidebarExpanded
                          ? "max-h-24 opacity-100 mt-1"
                          : "max-h-0 opacity-0 mt-0"
                      )}
                    >
                      <div className="ml-8 space-y-1">
                        {recentRevisions.map((rev) => (
                          <Link
                            key={rev.id}
                            href="/contract-revisions"
                            className="block px-2 py-1.5 rounded-lg hover:bg-gray-100/50 transition-colors"
                          >
                            <p className="text-[11px] font-medium text-gray-700 truncate tracking-tight">
                              {rev.title}
                            </p>
                            <p className="text-[10px] text-gray-400 truncate">
                              {rev.change} · {rev.time}
                            </p>
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </nav>

          {/* Divider */}
          <div className="mx-3 h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent" />

          {/* User Account */}
          <div className="p-2">
            <div className="relative">
              <div className={cn(
                "flex items-center h-11 w-full rounded-xl transition-all duration-200 overflow-hidden",
                sidebarExpanded ? "px-3 justify-between" : "justify-center"
              )}>
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className={cn(
                    "group flex items-center gap-3 text-gray-600 hover:text-gray-900 transition-all duration-200 flex-shrink-0",
                    showUserMenu && "text-gray-900"
                  )}
                >
                  <div className="w-7 h-7 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
                    {user?.username?.charAt(0).toUpperCase() || <IconUser size={14} />}
                  </div>
                  {sidebarExpanded && (
                    <span className="text-sm font-medium tracking-tight whitespace-nowrap truncate">
                      {user?.username || "내 계정"}
                    </span>
                  )}
                </button>
                {/* Settings Icon */}
                {sidebarExpanded && (
                  <Link
                    href="/settings"
                    className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all flex-shrink-0"
                  >
                    <IconSettings size={16} />
                  </Link>
                )}
              </div>
              {/* User Menu */}
              {showUserMenu && sidebarExpanded && (
                <UserMenuDropdown
                  user={user}
                  isOpen={showUserMenu}
                  onClose={() => setShowUserMenu(false)}
                  onLogout={handleLogout}
                />
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
                {recentRevisions.map((rev) => (
                  <div key={rev.id} className="p-2.5 bg-white/50 rounded-xl">
                    <p className="text-xs font-medium text-gray-800 truncate">{rev.title}</p>
                    <p className="text-[10px] text-gray-500 mt-0.5">{rev.change}</p>
                    <p className="text-[10px] text-gray-400 mt-0.5">{rev.time}</p>
                  </div>
                ))}
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

      {/* Floating Dock */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30">
        <div className="flex items-center gap-1 px-2 py-2 bg-[#e8f0ea]/90 backdrop-blur-xl border border-[#c8e6cf]/50 rounded-2xl shadow-lg">
          <Link
            href="/scan"
            className={cn(
              "group flex flex-col items-center justify-center w-14 h-14 rounded-xl transition-all duration-200",
              pathname === "/scan"
                ? "bg-gray-100 text-gray-900"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/80"
            )}
          >
            <IconScan size={22} className={pathname !== "/scan" ? "group-hover:scale-110 transition-transform" : ""} />
            <span className="text-[10px] font-medium mt-1 tracking-tight">스캔</span>
          </Link>

          <button
            onClick={() => setShowUploadSidebar(true)}
            className="group flex flex-col items-center justify-center w-16 h-16 -my-1 bg-gray-900 text-white rounded-2xl shadow-lg hover:bg-gray-800 hover:scale-105 active:scale-95 transition-all duration-200"
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
            <IconChecklist size={22} className={pathname !== "/checklist" ? "group-hover:scale-110 transition-transform" : ""} />
            <span className="text-[10px] font-medium mt-1 tracking-tight">체크리스트</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <main className={cn(
        "max-w-6xl mx-auto px-4 sm:px-6 pt-20 lg:pt-16 pb-32 lg:pb-6 transition-all duration-500 ease-out",
        sidebarExpanded && "lg:pl-56"
      )}>
        {children}
      </main>

      {/* Upload Sidebar */}
      <UploadSidebar
        isOpen={showUploadSidebar}
        onClose={() => setShowUploadSidebar(false)}
        onUploadSuccess={handleUploadSuccess}
      />

    </div>
  );
}
