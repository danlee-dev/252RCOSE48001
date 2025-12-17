"use client";

import { cn } from "@/lib/utils";

interface SwirlOrbProps {
  size?: number;
  className?: string;
}

export function SwirlOrb({ size = 120, className }: SwirlOrbProps) {
  return (
    <div
      className={cn("relative", className)}
      style={{ width: size, height: size }}
    >
      {/* Outer glow */}
      <div
        className="absolute inset-0 rounded-full opacity-40 blur-xl"
        style={{
          background: "radial-gradient(circle, #f59e0b 0%, #ea580c 50%, transparent 70%)",
        }}
      />

      {/* Main orb container */}
      <div
        className="absolute inset-0 rounded-full overflow-hidden"
        style={{
          background: "linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)",
          boxShadow: `
            inset 0 0 ${size * 0.3}px rgba(245, 158, 11, 0.3),
            0 0 ${size * 0.2}px rgba(234, 88, 12, 0.4),
            0 0 ${size * 0.4}px rgba(245, 158, 11, 0.2)
          `,
        }}
      >
        {/* Swirl layer 1 - Main orange swirl */}
        <div
          className="absolute inset-0 animate-swirl-slow"
          style={{
            background: `
              conic-gradient(
                from 0deg at 50% 50%,
                transparent 0deg,
                #f59e0b 60deg,
                #ea580c 120deg,
                transparent 180deg,
                #d97706 240deg,
                #f59e0b 300deg,
                transparent 360deg
              )
            `,
            filter: "blur(8px)",
            opacity: 0.8,
          }}
        />

        {/* Swirl layer 2 - Counter rotation */}
        <div
          className="absolute inset-0 animate-swirl-reverse"
          style={{
            background: `
              conic-gradient(
                from 180deg at 40% 60%,
                transparent 0deg,
                #1a1a1a 45deg,
                #374151 90deg,
                transparent 135deg,
                #1f2937 180deg,
                transparent 225deg,
                #111827 270deg,
                transparent 315deg,
                transparent 360deg
              )
            `,
            filter: "blur(12px)",
            opacity: 0.7,
          }}
        />

        {/* Swirl layer 3 - Accent swirl */}
        <div
          className="absolute inset-0 animate-swirl-medium"
          style={{
            background: `
              conic-gradient(
                from 90deg at 60% 40%,
                transparent 0deg,
                #fbbf24 30deg,
                transparent 60deg,
                #f97316 150deg,
                transparent 180deg,
                #fb923c 270deg,
                transparent 300deg,
                transparent 360deg
              )
            `,
            filter: "blur(6px)",
            opacity: 0.6,
          }}
        />

        {/* Inner dark core */}
        <div
          className="absolute animate-swirl-core"
          style={{
            top: "25%",
            left: "25%",
            width: "50%",
            height: "50%",
            borderRadius: "50%",
            background: `
              radial-gradient(
                ellipse at 30% 40%,
                #1f2937 0%,
                #111827 40%,
                transparent 70%
              )
            `,
            filter: "blur(10px)",
            opacity: 0.8,
          }}
        />

        {/* Highlight rim */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: `
              linear-gradient(
                135deg,
                rgba(251, 191, 36, 0.4) 0%,
                transparent 30%,
                transparent 70%,
                rgba(234, 88, 12, 0.2) 100%
              )
            `,
          }}
        />

        {/* Glass reflection */}
        <div
          className="absolute rounded-full"
          style={{
            top: "8%",
            left: "15%",
            width: "35%",
            height: "20%",
            background: "linear-gradient(180deg, rgba(255,255,255,0.15) 0%, transparent 100%)",
            filter: "blur(2px)",
            transform: "rotate(-20deg)",
          }}
        />
      </div>

      {/* Outer rim glow */}
      <div
        className="absolute inset-0 rounded-full pointer-events-none"
        style={{
          border: "1px solid rgba(251, 191, 36, 0.3)",
          boxShadow: "inset 0 0 20px rgba(245, 158, 11, 0.1)",
        }}
      />
    </div>
  );
}

export default SwirlOrb;
