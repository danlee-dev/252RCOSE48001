"use client";

import { cn } from "@/lib/utils";

// ============================================
// Bar Chart Component
// ============================================

interface BarChartData {
  label: string;
  value: number;
  color?: "default" | "success" | "warning" | "danger";
}

interface BarChartProps {
  data: BarChartData[];
  height?: number;
  showLabels?: boolean;
  showValues?: boolean;
  className?: string;
}

export function BarChart({
  data,
  height = 120,
  showLabels = true,
  showValues = true,
  className,
}: BarChartProps) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);

  const getBarGradient = (color: BarChartData["color"], isActive: boolean) => {
    if (!isActive) {
      return "bg-gradient-to-t from-gray-200 to-gray-100";
    }
    switch (color) {
      case "success":
        return "bg-gradient-to-t from-green-600 to-green-500";
      case "warning":
        return "bg-gradient-to-t from-amber-500 to-amber-400";
      case "danger":
        return "bg-gradient-to-t from-red-600 to-red-500";
      default:
        return "bg-gradient-to-t from-gray-900 to-gray-700";
    }
  };

  const getBarShadow = (color: BarChartData["color"]) => {
    switch (color) {
      case "success":
        return "shadow-[0_-4px_14px_-4px_rgba(34,197,94,0.35)]";
      case "warning":
        return "shadow-[0_-4px_14px_-4px_rgba(245,158,11,0.35)]";
      case "danger":
        return "shadow-[0_-4px_14px_-4px_rgba(239,68,68,0.35)]";
      default:
        return "shadow-[0_-4px_14px_-4px_rgba(23,23,23,0.25)]";
    }
  };

  // 값 레이블 높이를 고려한 바 영역 높이
  const valueHeight = showValues ? 20 : 0;
  const barAreaHeight = height - valueHeight;

  return (
    <div className={cn("w-full", className)}>
      <div
        className="flex items-end justify-between gap-2"
        style={{ height: `${height}px` }}
      >
        {data.map((item, index) => {
          // 픽셀 단위로 바 높이 계산
          const barHeightPx = Math.max((item.value / maxValue) * barAreaHeight, 4);
          const isActive = item.value > 0;

          return (
            <div
              key={index}
              className="flex-1 flex flex-col items-center justify-end gap-1"
              style={{ height: `${height}px` }}
            >
              {showValues && item.value > 0 && (
                <span className="text-xs font-semibold text-gray-700 tracking-tight">
                  {item.value}
                </span>
              )}
              <div
                className={cn(
                  "w-full transition-all duration-300 ease-out",
                  getBarGradient(item.color, isActive),
                  isActive && getBarShadow(item.color)
                )}
                style={{
                  height: `${barHeightPx}px`,
                  borderRadius: "6px 6px 0 0",
                }}
              />
            </div>
          );
        })}
      </div>
      {showLabels && (
        <div className="flex justify-between gap-2 mt-3">
          {data.map((item, index) => (
            <span
              key={index}
              className="flex-1 text-center text-[11px] font-medium text-gray-400 tracking-tight truncate"
            >
              {item.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// Donut Chart Component
// ============================================

interface DonutChartSegment {
  value: number;
  color: string;
  label?: string;
}

interface DonutChartProps {
  segments: DonutChartSegment[];
  size?: number;
  strokeWidth?: number;
  centerLabel?: string;
  centerValue?: string | number;
  className?: string;
}

export function DonutChart({
  segments,
  size = 160,
  strokeWidth = 14,
  centerLabel,
  centerValue,
  className,
}: DonutChartProps) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;

  let currentOffset = 0;

  return (
    <div className={cn("relative inline-flex", className)}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background track */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(0,0,0,0.04)"
          strokeWidth={strokeWidth}
        />

        {/* Segments */}
        {segments.map((segment, index) => {
          const percentage = segment.value / total;
          const strokeDasharray = `${percentage * circumference} ${circumference}`;
          const strokeDashoffset = -currentOffset * circumference;

          currentOffset += percentage;

          return (
            <circle
              key={index}
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke={segment.color}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              className="transition-all duration-500 ease-out"
              style={{
                filter: `drop-shadow(0 2px 4px ${segment.color}40)`,
              }}
            />
          );
        })}
      </svg>

      {/* Center content */}
      {(centerLabel || centerValue) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {centerValue !== undefined && (
            <span className="text-2xl font-bold text-gray-900 tracking-tight">
              {centerValue}
            </span>
          )}
          {centerLabel && (
            <span className="text-xs font-medium text-gray-500 mt-0.5">
              {centerLabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================
// Ring Progress Component (Single Value)
// ============================================

interface RingProgressProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  trackColor?: string;
  label?: string;
  showPercentage?: boolean;
  className?: string;
}

export function RingProgress({
  value,
  max = 100,
  size = 80,
  strokeWidth = 8,
  color = "#171717",
  trackColor = "rgba(0,0,0,0.06)",
  label,
  showPercentage = true,
  className,
}: RingProgressProps) {
  const percentage = Math.min((value / max) * 100, 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;
  const center = size / 2;

  return (
    <div className={cn("relative inline-flex", className)}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Track */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={trackColor}
          strokeWidth={strokeWidth}
        />

        {/* Progress */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-700 ease-out"
          style={{
            filter: `drop-shadow(0 2px 6px ${color}50)`,
          }}
        />
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {showPercentage && (
          <span className="text-sm font-bold text-gray-900 tracking-tight">
            {Math.round(percentage)}%
          </span>
        )}
        {label && (
          <span className="text-[10px] font-medium text-gray-500 mt-0.5">
            {label}
          </span>
        )}
      </div>
    </div>
  );
}

// ============================================
// Sparkline Component (Mini Line Chart)
// ============================================

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
  className?: string;
}

export function Sparkline({
  data,
  width = 100,
  height = 32,
  color = "#171717",
  showArea = true,
  className,
}: SparklineProps) {
  if (data.length < 2) return null;

  const padding = 2;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const minVal = Math.min(...data);
  const maxVal = Math.max(...data);
  const range = maxVal - minVal || 1;

  const points = data.map((val, i) => {
    const x = padding + (i / (data.length - 1)) * chartWidth;
    const y = padding + chartHeight - ((val - minVal) / range) * chartHeight;
    return `${x},${y}`;
  });

  const linePath = `M ${points.join(" L ")}`;
  const areaPath = `${linePath} L ${width - padding},${height - padding} L ${padding},${height - padding} Z`;

  return (
    <svg
      width={width}
      height={height}
      className={cn("overflow-visible", className)}
    >
      {/* Gradient definition */}
      <defs>
        <linearGradient id={`sparkline-gradient-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Area fill */}
      {showArea && (
        <path
          d={areaPath}
          fill={`url(#sparkline-gradient-${color})`}
          className="transition-all duration-300"
        />
      )}

      {/* Line */}
      <path
        d={linePath}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-all duration-300"
      />

      {/* End dot */}
      <circle
        cx={width - padding}
        cy={padding + chartHeight - ((data[data.length - 1] - minVal) / range) * chartHeight}
        r={3}
        fill={color}
        className="transition-all duration-300"
      />
    </svg>
  );
}

// ============================================
// Horizontal Progress Bar
// ============================================

interface ProgressBarProps {
  value: number;
  max?: number;
  height?: number;
  color?: "default" | "success" | "warning" | "danger";
  showLabel?: boolean;
  label?: string;
  className?: string;
}

export function ProgressBar({
  value,
  max = 100,
  height = 8,
  color = "default",
  showLabel = false,
  label,
  className,
}: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100);

  const getBarColor = () => {
    switch (color) {
      case "success":
        return "bg-gradient-to-r from-green-500 to-green-400";
      case "warning":
        return "bg-gradient-to-r from-amber-500 to-amber-400";
      case "danger":
        return "bg-gradient-to-r from-red-500 to-red-400";
      default:
        return "bg-gradient-to-r from-gray-800 to-gray-600";
    }
  };

  const getShadow = () => {
    switch (color) {
      case "success":
        return "shadow-[0_0_10px_-2px_rgba(34,197,94,0.4)]";
      case "warning":
        return "shadow-[0_0_10px_-2px_rgba(245,158,11,0.4)]";
      case "danger":
        return "shadow-[0_0_10px_-2px_rgba(239,68,68,0.4)]";
      default:
        return "shadow-[0_0_10px_-2px_rgba(23,23,23,0.3)]";
    }
  };

  return (
    <div className={cn("w-full", className)}>
      {showLabel && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-medium text-gray-600 tracking-tight">
            {label}
          </span>
          <span className="text-xs font-semibold text-gray-900 tracking-tight">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
      <div
        className="w-full bg-gray-100 rounded-full overflow-hidden"
        style={{ height: `${height}px` }}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            getBarColor(),
            percentage > 0 && getShadow()
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// ============================================
// Stats Card with Trend
// ============================================

interface StatTrendCardProps {
  title: string;
  value: string | number;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  sparklineData?: number[];
  icon?: React.ReactNode;
  iconBg?: string;
  className?: string;
}

export function StatTrendCard({
  title,
  value,
  trend,
  sparklineData,
  icon,
  iconBg = "bg-gray-100",
  className,
}: StatTrendCardProps) {
  return (
    <div className={cn("liquid-glass-card p-5", className)}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-gray-500 mb-2 tracking-tight">
            {title}
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-gray-900 tracking-tight">
              {value}
            </span>
            {trend && (
              <span
                className={cn(
                  "text-xs font-semibold tracking-tight",
                  trend.isPositive ? "text-green-600" : "text-red-600"
                )}
              >
                {trend.isPositive ? "+" : ""}{trend.value}%
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          {icon && (
            <div
              className={cn(
                "w-10 h-10 rounded-xl flex items-center justify-center",
                iconBg
              )}
            >
              {icon}
            </div>
          )}
          {sparklineData && sparklineData.length > 0 && (
            <Sparkline
              data={sparklineData}
              width={64}
              height={24}
              color={trend?.isPositive ? "#22c55e" : trend?.isPositive === false ? "#ef4444" : "#171717"}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================
// Legend Component
// ============================================

interface LegendItem {
  label: string;
  color: string;
  value?: number | string;
}

interface LegendProps {
  items: LegendItem[];
  direction?: "horizontal" | "vertical";
  className?: string;
}

export function ChartLegend({
  items,
  direction = "horizontal",
  className,
}: LegendProps) {
  return (
    <div
      className={cn(
        "flex gap-4",
        direction === "vertical" ? "flex-col" : "flex-wrap",
        className
      )}
    >
      {items.map((item, index) => (
        <div key={index} className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{
              backgroundColor: item.color,
              boxShadow: `0 0 6px -1px ${item.color}60`,
            }}
          />
          <span className="text-xs font-medium text-gray-600 tracking-tight">
            {item.label}
          </span>
          {item.value !== undefined && (
            <span className="text-xs font-semibold text-gray-900 tracking-tight">
              {item.value}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
