"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import {
  IconCamera,
  IconHome,
  IconLightbulb,
  IconLoading,
  IconRefresh,
  IconScan,
  IconShield,
  IconWarning,
  IconDanger,
} from "@/components/icons";
import { cn } from "@/lib/utils";

// Quick Scan result interface
interface QuickScanResult {
  risk_level: "HIGH" | "MEDIUM" | "LOW" | "SAFE";
  detected_clauses: DetectedClause[];
  summary: string;
  scan_time_ms: number;
}

interface DetectedClause {
  text: string;
  risk_level: "HIGH" | "MEDIUM" | "LOW";
  keyword: string;
  bbox?: { x: number; y: number; width: number; height: number };
}

// Excuse templates for users
const EXCUSE_TEMPLATES = [
  {
    id: 1,
    title: "부모님 핑계",
    text: "부모님이 계약서는 꼭 사진 찍어서 보내라고 하셨어요. 잠깐만요!",
    icon: "family",
  },
  {
    id: 2,
    title: "기록용",
    text: "기록용으로 사진 한 장만 찍어둘게요. 나중에 헷갈릴까 봐요.",
    icon: "note",
  },
  {
    id: 3,
    title: "변호사 지인",
    text: "변호사 친구가 계약서는 항상 사진 찍어두라고 해서요.",
    icon: "lawyer",
  },
  {
    id: 4,
    title: "회사 규정",
    text: "회사에서 계약 전에 꼭 검토하라고 해서요. 잠시만요.",
    icon: "company",
  },
];

// Quick tips for users
const QUICK_TIPS = [
  "급여, 수당 관련 조항을 주의깊게 확인하세요",
  "계약 해지 조건과 위약금을 반드시 확인하세요",
  "근무 시간과 초과 근무 규정을 체크하세요",
  "서명 전 모든 빈칸이 채워졌는지 확인하세요",
];

export default function ScanPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<QuickScanResult | null>(null);
  const [showTips, setShowTips] = useState(true);
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [showExcuses, setShowExcuses] = useState(false);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [stream, setStream] = useState<MediaStream | null>(null);

  // Auth check
  useEffect(() => {
    if (!authApi.isAuthenticated()) {
      router.push("/login");
    }
  }, [router]);

  // Initialize camera
  const initCamera = useCallback(async (mode: "environment" | "user") => {
    try {
      setIsLoading(true);

      // Stop existing stream first
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }

      // Small delay to ensure previous stream is fully stopped
      await new Promise(resolve => setTimeout(resolve, 100));

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: mode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });

      setStream(mediaStream);
      setHasPermission(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        // Handle play() promise properly to avoid AbortError
        try {
          await videoRef.current.play();
        } catch (playError) {
          // Ignore AbortError - it happens when stream changes quickly
          if ((playError as Error).name !== "AbortError") {
            throw playError;
          }
        }
      }
    } catch (error) {
      console.error(">>> Camera access error:", error);
      if ((error as Error).name !== "AbortError") {
        setHasPermission(false);
      }
    } finally {
      setIsLoading(false);
    }
  }, [stream]);

  // Initial camera setup
  useEffect(() => {
    initCamera(facingMode);

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Switch camera
  const switchCamera = useCallback(() => {
    const newMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(newMode);
    initCamera(newMode);
  }, [facingMode, initCamera]);

  // Rotate tips
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTipIndex((prev) => (prev + 1) % QUICK_TIPS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Perform quick scan
  const performQuickScan = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsScanning(true);
    setShowTips(false);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      // Capture frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Convert to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), "image/jpeg", 0.9);
      });

      // Send to quick scan API
      const formData = new FormData();
      formData.append("image", blob, "scan.jpg");

      const token = localStorage.getItem("access_token");
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/scan/quick`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Scan failed");
      }

      const result: QuickScanResult = await response.json();
      setScanResult(result);

      // Draw overlays on detected clauses
      drawOverlays(result.detected_clauses);
    } catch (error) {
      console.error(">>> Quick scan error:", error);
      // 에러 발생 시 사용자에게 알림
      const errorResult: QuickScanResult = {
        risk_level: "SAFE",
        detected_clauses: [],
        summary: "스캔 중 오류가 발생했습니다. 네트워크 연결을 확인하고 다시 시도해주세요.",
        scan_time_ms: 0,
      };
      setScanResult(errorResult);
    } finally {
      setIsScanning(false);
    }
  }, []);

  // Draw AR overlays
  const drawOverlays = (clauses: DetectedClause[]) => {
    if (!overlayCanvasRef.current || !videoRef.current) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    clauses.forEach((clause) => {
      if (clause.bbox) {
        const { x, y, width, height } = clause.bbox;

        // Draw warning box
        ctx.strokeStyle = clause.risk_level === "HIGH" ? "#ef4444" : "#f59e0b";
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(x, y, width, height);

        // Draw warning icon area
        ctx.fillStyle =
          clause.risk_level === "HIGH"
            ? "rgba(239, 68, 68, 0.3)"
            : "rgba(245, 158, 11, 0.3)";
        ctx.fillRect(x, y, width, height);
      }
    });
  };

  // Reset scan
  const resetScan = useCallback(() => {
    setScanResult(null);
    setShowTips(true);

    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
  }, []);

  // Render risk level badge
  const getRiskBadge = (level: string) => {
    switch (level) {
      case "HIGH":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
            <IconDanger size={12} />
            위험
          </span>
        );
      case "MEDIUM":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
            <IconWarning size={12} />
            주의
          </span>
        );
      case "LOW":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
            <IconShield size={12} />
            낮음
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
            <IconShield size={12} />
            안전
          </span>
        );
    }
  };

  // Permission denied view
  if (hasPermission === false) {
    return (
      <div className="min-h-[100dvh] bg-gray-900 flex flex-col items-center justify-center p-6 text-center safe-area-inset">
        <div className="w-20 h-20 bg-gray-800 rounded-2xl flex items-center justify-center mb-6">
          <IconCamera size={40} className="text-gray-500" />
        </div>
        <h1 className="text-xl font-semibold text-white mb-2 tracking-tight">카메라 접근 권한 필요</h1>
        <p className="text-gray-400 mb-6 max-w-xs text-sm leading-relaxed">
          실시간 계약서 스캔을 위해 카메라 접근 권한이 필요합니다.
          브라우저 설정에서 카메라 권한을 허용해주세요.
        </p>
        <button
          onClick={() => initCamera(facingMode)}
          className="inline-flex items-center gap-2 px-6 py-3 bg-white text-gray-900 rounded-full font-medium hover:bg-gray-100 active:scale-95 transition-all min-h-[48px]"
        >
          <IconRefresh size={18} />
          다시 시도
        </button>
        <Link
          href="/"
          className="mt-4 text-gray-400 hover:text-white transition-colors text-sm py-2"
        >
          홈으로 돌아가기
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-[100dvh] bg-black relative overflow-hidden">
      {/* Camera Preview */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Overlay Canvas for AR markers */}
      <canvas
        ref={overlayCanvasRef}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
      />

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-3">
            <IconLoading size={32} className="text-white" />
            <p className="text-white text-sm tracking-tight">카메라 초기화 중...</p>
          </div>
        </div>
      )}

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 z-20 pt-[env(safe-area-inset-top)]">
        <div className="flex items-center justify-between p-4">
          <Link
            href="/"
            className="w-11 h-11 bg-black/40 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/60 active:scale-95 transition-all"
          >
            <IconHome size={20} />
          </Link>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowExcuses(!showExcuses)}
              className={cn(
                "px-4 py-2.5 rounded-full text-xs font-medium flex items-center gap-1.5 transition-all active:scale-95 min-h-[44px]",
                showExcuses
                  ? "bg-white text-gray-900"
                  : "bg-black/40 backdrop-blur-sm text-white hover:bg-black/60"
              )}
            >
              <IconLightbulb size={14} />
              <span className="hidden xs:inline">핑계 도우미</span>
              <span className="xs:hidden">핑계</span>
            </button>

            <button
              onClick={switchCamera}
              className="w-11 h-11 bg-black/40 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/60 active:scale-95 transition-all"
            >
              <IconRefresh size={18} />
            </button>
          </div>
        </div>

        {/* Excuse Templates Dropdown */}
        {showExcuses && (
          <div className="mx-4 mb-2 bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl overflow-hidden animate-fadeInDown sm:max-w-md sm:mx-auto">
            <div className="p-3 sm:p-4 border-b border-gray-100">
              <h3 className="text-sm font-semibold text-gray-900 tracking-tight">
                자연스럽게 촬영하는 핑계
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                탭하면 복사됩니다
              </p>
            </div>
            <div className="divide-y divide-gray-100">
              {EXCUSE_TEMPLATES.map((excuse) => (
                <button
                  key={excuse.id}
                  onClick={() => {
                    navigator.clipboard.writeText(excuse.text);
                    setShowExcuses(false);
                  }}
                  className="w-full px-4 py-3.5 text-left hover:bg-gray-50 active:bg-gray-100 transition-colors"
                >
                  <p className="text-sm font-medium text-gray-900 tracking-tight">{excuse.title}</p>
                  <p className="text-xs text-gray-600 mt-0.5 leading-relaxed">{excuse.text}</p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Scan Frame Guide */}
      {!scanResult && (
        <div className="absolute inset-0 pointer-events-none z-10">
          {/* Corner guides */}
          <div className="absolute inset-8 sm:inset-16 md:inset-24">
            {/* Top-left corner */}
            <div className="absolute top-0 left-0 w-12 h-12">
              <div className="absolute top-0 left-0 w-full h-1 bg-white rounded-full" />
              <div className="absolute top-0 left-0 w-1 h-full bg-white rounded-full" />
            </div>
            {/* Top-right corner */}
            <div className="absolute top-0 right-0 w-12 h-12">
              <div className="absolute top-0 right-0 w-full h-1 bg-white rounded-full" />
              <div className="absolute top-0 right-0 w-1 h-full bg-white rounded-full" />
            </div>
            {/* Bottom-left corner */}
            <div className="absolute bottom-0 left-0 w-12 h-12">
              <div className="absolute bottom-0 left-0 w-full h-1 bg-white rounded-full" />
              <div className="absolute bottom-0 left-0 w-1 h-full bg-white rounded-full" />
            </div>
            {/* Bottom-right corner */}
            <div className="absolute bottom-0 right-0 w-12 h-12">
              <div className="absolute bottom-0 right-0 w-full h-1 bg-white rounded-full" />
              <div className="absolute bottom-0 right-0 w-1 h-full bg-white rounded-full" />
            </div>
          </div>

          {/* Scan line animation */}
          {isScanning && (
            <div className="absolute left-8 right-8 sm:left-16 sm:right-16 md:left-24 md:right-24 h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-scan-line" />
          )}
        </div>
      )}

      {/* Tips Banner */}
      {showTips && !scanResult && (
        <div className="absolute bottom-32 sm:bottom-36 left-4 right-4 z-20">
          <div className="bg-black/60 backdrop-blur-sm rounded-xl px-4 py-3 max-w-md mx-auto">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                <IconLightbulb size={16} className="text-cyan-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-400 mb-0.5">Quick Tip</p>
                <p className="text-sm text-white leading-snug tracking-tight">
                  {QUICK_TIPS[currentTipIndex]}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Scan Result Panel */}
      {scanResult && (
        <div className="absolute bottom-0 left-0 right-0 z-30 animate-slideUp pb-[env(safe-area-inset-bottom)]">
          <div className="bg-white rounded-t-3xl shadow-2xl max-h-[70vh] sm:max-h-[60vh] overflow-y-auto">
            {/* Handle */}
            <div className="flex justify-center pt-3 pb-2 sticky top-0 bg-white">
              <div className="w-10 h-1 bg-gray-300 rounded-full" />
            </div>

            {/* Header */}
            <div className="px-4 sm:px-5 pb-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900 tracking-tight">스캔 결과</h2>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {(scanResult.scan_time_ms / 1000).toFixed(1)}초 소요
                  </p>
                </div>
                {getRiskBadge(scanResult.risk_level)}
              </div>
            </div>

            {/* Detected Clauses */}
            <div className="p-4 sm:p-5 space-y-3">
              {scanResult.detected_clauses.length > 0 ? (
                <>
                  <p className="text-sm text-gray-600 mb-4 tracking-tight">{scanResult.summary}</p>

                  {scanResult.detected_clauses.map((clause, index) => (
                    <div
                      key={index}
                      className={cn(
                        "p-3 sm:p-4 rounded-xl border-l-4",
                        clause.risk_level === "HIGH"
                          ? "bg-red-50 border-red-500"
                          : clause.risk_level === "MEDIUM"
                          ? "bg-amber-50 border-amber-500"
                          : "bg-blue-50 border-blue-500"
                      )}
                    >
                      <div className="flex items-start justify-between gap-2 sm:gap-3">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 tracking-tight">{clause.text}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            키워드: {clause.keyword}
                          </p>
                        </div>
                        {getRiskBadge(clause.risk_level)}
                      </div>
                    </div>
                  ))}

                  <div className="pt-4 border-t border-gray-100 mt-4">
                    <p className="text-xs text-gray-500 text-center mb-3">
                      정밀 분석이 필요하신가요?
                    </p>
                    <Link
                      href="/"
                      className="w-full flex items-center justify-center gap-2 px-4 py-3.5 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-gray-800 active:scale-[0.98] transition-all min-h-[48px]"
                    >
                      <IconScan size={16} />
                      정밀 분석 시작하기
                    </Link>
                  </div>
                </>
              ) : (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <IconShield size={32} className="text-green-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-1 tracking-tight">
                    위험 조항이 발견되지 않았습니다
                  </h3>
                  <p className="text-sm text-gray-500">
                    기본적인 스캔에서는 문제가 없어 보입니다.
                  </p>
                </div>
              )}
            </div>

            {/* Close button */}
            <div className="p-4 sm:p-5 pt-0">
              <button
                onClick={resetScan}
                className="w-full py-3.5 text-gray-600 text-sm font-medium hover:text-gray-900 active:bg-gray-100 rounded-xl transition-all min-h-[48px]"
              >
                다시 스캔하기
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Bottom Controls */}
      {!scanResult && (
        <div className="absolute bottom-0 left-0 right-0 z-20 pb-8 sm:pb-10 pt-6 bg-gradient-to-t from-black/60 to-transparent" style={{ paddingBottom: 'max(2rem, env(safe-area-inset-bottom))' }}>
          <div className="flex flex-col items-center gap-3 sm:gap-4">
            {/* Scan Button */}
            <button
              onClick={performQuickScan}
              disabled={isScanning || isLoading}
              className={cn(
                "w-[72px] h-[72px] sm:w-20 sm:h-20 rounded-full flex items-center justify-center transition-all shadow-lg",
                isScanning
                  ? "bg-cyan-500 animate-pulse"
                  : "bg-white hover:bg-gray-100 active:scale-95"
              )}
            >
              {isScanning ? (
                <IconLoading size={32} className="text-white" />
              ) : (
                <IconScan size={32} className="text-gray-900 sm:w-9 sm:h-9" />
              )}
            </button>

            {/* Label */}
            <p className="text-white/80 text-sm text-center px-4 tracking-tight">
              {isScanning ? "스캔 중..." : "계약서를 프레임 안에 맞추고 스캔하세요"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
