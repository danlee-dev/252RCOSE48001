"use client";

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface AIAvatarProps {
  size?: number;
  isThinking?: boolean;
  isSpeaking?: boolean;
  className?: string;
}

export function AIAvatar({
  size = 44,
  isThinking = false,
  isSpeaking = false,
  className,
}: AIAvatarProps) {
  const [isBlinking, setIsBlinking] = useState(false);
  const [eyeDirection, setEyeDirection] = useState<"center" | "left" | "right" | "up" | "down">("center");
  const [leftEyebrowOffset, setLeftEyebrowOffset] = useState(0);
  const [rightEyebrowOffset, setRightEyebrowOffset] = useState(0);
  const [mouthState, setMouthState] = useState<"neutral" | "open" | "smile" | "smirk-left" | "smirk-right" | "frown" | "surprised">("neutral");
  const [logoOffsets, setLogoOffsets] = useState({
    topLeft: { x: 0, y: 0, rotate: 0, scale: 1 },
    topRight: { x: 0, y: 0, rotate: 0, scale: 1 },
    bottomLeft: { x: 0, y: 0, rotate: 0, scale: 1 },
    bottomRight: { x: 0, y: 0, rotate: 0, scale: 1 },
  });
  const [expression, setExpression] = useState<"neutral" | "happy" | "thinking" | "surprised" | "concerned">("neutral");

  // Sync logo with expression
  const updateLogoForExpression = useCallback((expr: string) => {
    switch (expr) {
      case "happy":
        setLogoOffsets({
          topLeft: { x: -0.5, y: -1, rotate: -3, scale: 1.02 },
          topRight: { x: 0.5, y: -1, rotate: 3, scale: 1.02 },
          bottomLeft: { x: -0.3, y: 0.5, rotate: -2, scale: 1 },
          bottomRight: { x: 0.3, y: 0.5, rotate: 2, scale: 1 },
        });
        break;
      case "thinking":
        setLogoOffsets({
          topLeft: { x: -1, y: -0.5, rotate: -5, scale: 1 },
          topRight: { x: 1.5, y: -1.5, rotate: 8, scale: 1.05 },
          bottomLeft: { x: 0.5, y: 0.3, rotate: 2, scale: 0.98 },
          bottomRight: { x: -0.5, y: 0, rotate: -3, scale: 1 },
        });
        break;
      case "surprised":
        setLogoOffsets({
          topLeft: { x: -1, y: -2, rotate: -5, scale: 1.08 },
          topRight: { x: 1, y: -2, rotate: 5, scale: 1.08 },
          bottomLeft: { x: -0.5, y: 1, rotate: -3, scale: 1.02 },
          bottomRight: { x: 0.5, y: 1, rotate: 3, scale: 1.02 },
        });
        break;
      case "concerned":
        setLogoOffsets({
          topLeft: { x: 0, y: 0.5, rotate: 3, scale: 0.98 },
          topRight: { x: -0.5, y: -1, rotate: -5, scale: 1.02 },
          bottomLeft: { x: 0.3, y: -0.3, rotate: 2, scale: 1 },
          bottomRight: { x: -0.3, y: 0.5, rotate: -2, scale: 0.98 },
        });
        break;
      default:
        setLogoOffsets({
          topLeft: { x: (Math.random() - 0.5) * 1.5, y: (Math.random() - 0.5) * 1, rotate: (Math.random() - 0.5) * 4, scale: 1 },
          topRight: { x: (Math.random() - 0.5) * 1.5, y: (Math.random() - 0.5) * 1, rotate: (Math.random() - 0.5) * 4, scale: 1 },
          bottomLeft: { x: (Math.random() - 0.5) * 1, y: (Math.random() - 0.5) * 0.8, rotate: (Math.random() - 0.5) * 3, scale: 1 },
          bottomRight: { x: (Math.random() - 0.5) * 1, y: (Math.random() - 0.5) * 0.8, rotate: (Math.random() - 0.5) * 3, scale: 1 },
        });
    }
  }, []);

  // Blink animation - more frequent
  useEffect(() => {
    const blink = () => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 100);
    };

    const scheduleNextBlink = () => {
      const delay = 1500 + Math.random() * 2000;
      return setTimeout(() => {
        blink();
        scheduleNextBlink();
      }, delay);
    };

    const timeout = scheduleNextBlink();
    return () => clearTimeout(timeout);
  }, []);

  // Eye direction animation - more frequent movement
  useEffect(() => {
    const changeDirection = () => {
      const directions: ("center" | "left" | "right" | "up" | "down")[] = ["center", "left", "right", "up", "down", "center", "center"];
      const newDir = directions[Math.floor(Math.random() * directions.length)];
      setEyeDirection(newDir);
    };

    const interval = setInterval(changeDirection, 800 + Math.random() * 1200);
    return () => clearInterval(interval);
  }, []);

  // Expression changes - drive everything
  useEffect(() => {
    if (isThinking) {
      setExpression("thinking");
      return;
    }

    const expressions: ("neutral" | "happy" | "thinking" | "surprised" | "concerned")[] =
      ["neutral", "neutral", "happy", "neutral", "surprised", "neutral", "concerned", "neutral", "happy"];

    const changeExpression = () => {
      const newExpr = expressions[Math.floor(Math.random() * expressions.length)];
      setExpression(newExpr);
      updateLogoForExpression(newExpr);

      // Sync face with expression
      switch (newExpr) {
        case "happy":
          setMouthState("smile");
          setLeftEyebrowOffset(-1);
          setRightEyebrowOffset(-1);
          break;
        case "surprised":
          setMouthState("surprised");
          setLeftEyebrowOffset(-2.5);
          setRightEyebrowOffset(-2.5);
          break;
        case "concerned":
          setMouthState("frown");
          setLeftEyebrowOffset(1);
          setRightEyebrowOffset(-1.5);
          break;
        case "thinking":
          setMouthState("smirk-left");
          setLeftEyebrowOffset(-1.5);
          setRightEyebrowOffset(0.5);
          break;
        default:
          setMouthState("neutral");
          setLeftEyebrowOffset(0);
          setRightEyebrowOffset(0);
      }
    };

    const interval = setInterval(changeExpression, 1500 + Math.random() * 1500);
    return () => clearInterval(interval);
  }, [isThinking, updateLogoForExpression]);

  // Enhanced thinking animation
  useEffect(() => {
    if (!isThinking) return;

    const thinkingAnimation = () => {
      setLeftEyebrowOffset(-2);
      setRightEyebrowOffset(0.5);
      updateLogoForExpression("thinking");
      setTimeout(() => {
        setLeftEyebrowOffset(0.5);
        setRightEyebrowOffset(-2);
        setLogoOffsets(prev => ({
          ...prev,
          topRight: { ...prev.topRight, rotate: prev.topRight.rotate + 5 },
        }));
      }, 400);
    };

    const interval = setInterval(thinkingAnimation, 800);
    thinkingAnimation();
    return () => clearInterval(interval);
  }, [isThinking, updateLogoForExpression]);

  // Speaking animation
  useEffect(() => {
    if (!isSpeaking) return;

    const speakingStates: ("neutral" | "open" | "smile" | "smirk-left" | "smirk-right")[] =
      ["open", "neutral", "open", "smile", "open", "smirk-left", "open"];
    let index = 0;

    const interval = setInterval(() => {
      setMouthState(speakingStates[index % speakingStates.length]);
      // Subtle logo movement while speaking
      setLogoOffsets(prev => ({
        topLeft: { ...prev.topLeft, y: prev.topLeft.y + (Math.random() - 0.5) * 0.5 },
        topRight: { ...prev.topRight, y: prev.topRight.y + (Math.random() - 0.5) * 0.5 },
        bottomLeft: { ...prev.bottomLeft, x: prev.bottomLeft.x + (Math.random() - 0.5) * 0.3 },
        bottomRight: { ...prev.bottomRight, x: prev.bottomRight.x + (Math.random() - 0.5) * 0.3 },
      }));
      index++;
    }, 120);

    return () => clearInterval(interval);
  }, [isSpeaking]);

  // Continuous subtle logo breathing animation
  useEffect(() => {
    if (isThinking || isSpeaking) return;

    const breathe = () => {
      setLogoOffsets(prev => ({
        topLeft: {
          x: prev.topLeft.x + (Math.random() - 0.5) * 0.8,
          y: prev.topLeft.y + (Math.random() - 0.5) * 0.6,
          rotate: prev.topLeft.rotate + (Math.random() - 0.5) * 2,
          scale: 1 + (Math.random() - 0.5) * 0.03,
        },
        topRight: {
          x: prev.topRight.x + (Math.random() - 0.5) * 0.8,
          y: prev.topRight.y + (Math.random() - 0.5) * 0.6,
          rotate: prev.topRight.rotate + (Math.random() - 0.5) * 2,
          scale: 1 + (Math.random() - 0.5) * 0.03,
        },
        bottomLeft: {
          x: prev.bottomLeft.x + (Math.random() - 0.5) * 0.6,
          y: prev.bottomLeft.y + (Math.random() - 0.5) * 0.5,
          rotate: prev.bottomLeft.rotate + (Math.random() - 0.5) * 1.5,
          scale: 1 + (Math.random() - 0.5) * 0.02,
        },
        bottomRight: {
          x: prev.bottomRight.x + (Math.random() - 0.5) * 0.6,
          y: prev.bottomRight.y + (Math.random() - 0.5) * 0.5,
          rotate: prev.bottomRight.rotate + (Math.random() - 0.5) * 1.5,
          scale: 1 + (Math.random() - 0.5) * 0.02,
        },
      }));
    };

    const interval = setInterval(breathe, 500);
    return () => clearInterval(interval);
  }, [isThinking, isSpeaking]);

  const getEyeOffset = () => {
    switch (eyeDirection) {
      case "left": return { x: -1.5, y: 0 };
      case "right": return { x: 1.5, y: 0 };
      case "up": return { x: 0, y: -1 };
      case "down": return { x: 0, y: 1 };
      default: return { x: 0, y: 0 };
    }
  };

  const eyeOffset = getEyeOffset();

  const getMouthPath = () => {
    switch (mouthState) {
      case "open":
        return "M-2 1 Q0 3.5 2 1 Q0 3 -2 1";
      case "smile":
        return "M-3 0 Q0 3 3 0";
      case "smirk-left":
        return "M-3 0.5 Q-1 2 2 0";
      case "smirk-right":
        return "M-2 0 Q1 2 3 0.5";
      case "frown":
        return "M-2 2 Q0 0.5 2 2";
      case "surprised":
        return "M-1.5 1 Q0 2.5 1.5 1 Q0 2 -1.5 1";
      default:
        return "M-2 0.5 Q0 1.8 2 0.5";
    }
  };

  // Clamp values to prevent extreme movements
  const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));

  return (
    <div
      className={cn("relative flex-shrink-0", className)}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 44 44"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* DocScanner Logo - 4 parallelograms with independent movement */}
        <g transform="translate(6, 4) scale(0.152)">
          {/* Top left parallelogram */}
          <path
            d="M43.9767 5.81418C44.7712 2.40898 47.807 0 51.3037 0H141.185C146.031 0 149.613 4.51423 148.512 9.23344L125.69 107.043C124.895 110.448 121.86 112.857 118.363 112.857H28.4815C23.6355 112.857 20.0533 108.343 21.1545 103.624L43.9767 5.81418Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.topLeft.x, -3, 3)}px, ${clamp(logoOffsets.topLeft.y, -3, 3)}px) rotate(${clamp(logoOffsets.topLeft.rotate, -8, 8)}deg) scale(${logoOffsets.topLeft.scale})`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-300 ease-out"
          />
          {/* Top right parallelogram */}
          <path
            d="M146.242 54.6245C147.072 51.2653 150.086 48.9053 153.547 48.9053H201.247C206.133 48.9053 209.723 53.4901 208.551 58.2336L196.468 107.138C195.639 110.498 192.624 112.858 189.164 112.858H141.464C136.578 112.858 132.988 108.273 134.16 103.529L146.242 54.6245Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.topRight.x, -3, 3)}px, ${clamp(logoOffsets.topRight.y, -3, 3)}px) rotate(${clamp(logoOffsets.topRight.rotate, -8, 8)}deg) scale(${logoOffsets.topRight.scale})`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-300 ease-out"
          />
          {/* Bottom left parallelogram */}
          <path
            d="M14.387 137.862C15.2169 134.503 18.231 132.143 21.6912 132.143H69.3911C74.2773 132.143 77.8673 136.727 76.6953 141.471L64.613 190.376C63.7831 193.735 60.769 196.095 57.3088 196.095H9.60886C4.72269 196.095 1.13272 191.51 2.30466 186.767L14.387 137.862Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.bottomLeft.x, -2, 2)}px, ${clamp(logoOffsets.bottomLeft.y, -2, 2)}px) rotate(${clamp(logoOffsets.bottomLeft.rotate, -6, 6)}deg) scale(${logoOffsets.bottomLeft.scale})`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200 ease-out"
          />
          {/* Bottom right parallelogram */}
          <path
            d="M85.1681 137.957C85.9626 134.552 88.9984 132.143 92.4951 132.143H182.377C187.223 132.143 190.805 136.657 189.704 141.376L166.881 239.186C166.087 242.591 163.051 245 159.554 245H69.6729C64.8269 245 61.2447 240.485 62.3459 235.766L85.1681 137.957Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.bottomRight.x, -2, 2)}px, ${clamp(logoOffsets.bottomRight.y, -2, 2)}px) rotate(${clamp(logoOffsets.bottomRight.rotate, -6, 6)}deg) scale(${logoOffsets.bottomRight.scale})`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200 ease-out"
          />
        </g>

        {/* Left eye with eyebrow */}
        <g transform="translate(14, 12)">
          {/* Left eyebrow */}
          <path
            d={`M-2 ${-2.5 + leftEyebrowOffset} Q1 ${-4 + leftEyebrowOffset} 4 ${-2.5 + leftEyebrowOffset * 0.6}`}
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            fill="none"
            className="transition-all duration-200"
          />
          {/* Left eye */}
          {isBlinking ? (
            <path
              d="M-1 2 Q1.5 3 4 2"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              fill="none"
            />
          ) : (
            <ellipse
              cx={1.5 + eyeOffset.x}
              cy={2 + eyeOffset.y}
              rx={expression === "surprised" ? 3.2 : 2.8}
              ry={expression === "surprised" ? 3 : 2.5}
              fill="white"
              className="transition-all duration-150"
            />
          )}
        </g>

        {/* Right eye with eyebrow */}
        <g transform="translate(28, 14)">
          {/* Right eyebrow */}
          <path
            d={`M-2 ${-2.5 + rightEyebrowOffset * 0.6} Q1 ${-4 + rightEyebrowOffset} 4 ${-2.5 + rightEyebrowOffset}`}
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            fill="none"
            className="transition-all duration-200"
          />
          {/* Right eye */}
          {isBlinking ? (
            <path
              d="M-1 2 Q1.5 3 4 2"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              fill="none"
            />
          ) : (
            <ellipse
              cx={1.5 + eyeOffset.x}
              cy={2 + eyeOffset.y}
              rx={expression === "surprised" ? 3.2 : 2.8}
              ry={expression === "surprised" ? 3 : 2.5}
              fill="white"
              className="transition-all duration-150"
            />
          )}
        </g>

        {/* Mouth */}
        <g transform="translate(22, 32)">
          <path
            d={getMouthPath()}
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            fill={mouthState === "open" || mouthState === "surprised" ? "white" : "none"}
            className="transition-all duration-100"
          />
        </g>

        {/* Thinking indicator */}
        {isThinking && (
          <g className="animate-pulse">
            <circle cx="38" cy="6" r="2.5" fill="black" opacity="0.7" />
            <circle cx="42" cy="10" r="1.8" fill="black" opacity="0.5" />
            <circle cx="44" cy="14" r="1.2" fill="black" opacity="0.3" />
          </g>
        )}
      </svg>
    </div>
  );
}

// Small avatar for messages - with eyebrows and more animation
export function AIAvatarSmall({
  size = 28,
  isThinking = false,
  className,
}: {
  size?: number;
  isThinking?: boolean;
  className?: string;
}) {
  const [isBlinking, setIsBlinking] = useState(false);
  const [eyeDirection, setEyeDirection] = useState<"center" | "left" | "right">("center");
  const [mouthState, setMouthState] = useState<"neutral" | "smile" | "open" | "surprised" | "smirk">("neutral");
  const [leftEyebrowOffset, setLeftEyebrowOffset] = useState(0);
  const [rightEyebrowOffset, setRightEyebrowOffset] = useState(0);
  const [expression, setExpression] = useState<"neutral" | "happy" | "surprised" | "thinking" | "curious">("neutral");
  const [logoOffsets, setLogoOffsets] = useState({
    topLeft: { x: 0, y: 0, rotate: 0 },
    topRight: { x: 0, y: 0, rotate: 0 },
    bottomLeft: { x: 0, y: 0, rotate: 0 },
    bottomRight: { x: 0, y: 0, rotate: 0 },
  });

  // Blink - frequent
  useEffect(() => {
    const blink = () => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 100);
    };

    const scheduleNextBlink = () => {
      const delay = 1500 + Math.random() * 1800;
      return setTimeout(() => {
        blink();
        scheduleNextBlink();
      }, delay);
    };

    const timeout = scheduleNextBlink();
    return () => clearTimeout(timeout);
  }, []);

  // Eye direction - more frequent
  useEffect(() => {
    const changeDirection = () => {
      const directions: ("center" | "left" | "right")[] = ["center", "left", "right", "center", "center"];
      setEyeDirection(directions[Math.floor(Math.random() * directions.length)]);
    };

    const interval = setInterval(changeDirection, 700 + Math.random() * 1000);
    return () => clearInterval(interval);
  }, []);

  // Expression changes - drives face and logo
  useEffect(() => {
    if (isThinking) {
      setExpression("thinking");
      return;
    }

    const expressions: ("neutral" | "happy" | "surprised" | "curious")[] =
      ["neutral", "happy", "neutral", "surprised", "neutral", "curious", "happy", "neutral"];

    const changeExpression = () => {
      const newExpr = expressions[Math.floor(Math.random() * expressions.length)];
      setExpression(newExpr);

      // Sync face with expression
      switch (newExpr) {
        case "happy":
          setMouthState("smile");
          setLeftEyebrowOffset(-1);
          setRightEyebrowOffset(-1);
          setLogoOffsets({
            topLeft: { x: -0.4, y: -0.8, rotate: -2 },
            topRight: { x: 0.4, y: -0.8, rotate: 2 },
            bottomLeft: { x: -0.2, y: 0.3, rotate: -1 },
            bottomRight: { x: 0.2, y: 0.3, rotate: 1 },
          });
          break;
        case "surprised":
          setMouthState("surprised");
          setLeftEyebrowOffset(-2);
          setRightEyebrowOffset(-2);
          setLogoOffsets({
            topLeft: { x: -0.8, y: -1.2, rotate: -4 },
            topRight: { x: 0.8, y: -1.2, rotate: 4 },
            bottomLeft: { x: -0.4, y: 0.6, rotate: -2 },
            bottomRight: { x: 0.4, y: 0.6, rotate: 2 },
          });
          break;
        case "curious":
          setMouthState("smirk");
          setLeftEyebrowOffset(-1.5);
          setRightEyebrowOffset(0.5);
          setLogoOffsets({
            topLeft: { x: -0.6, y: -0.3, rotate: -3 },
            topRight: { x: 1, y: -1, rotate: 5 },
            bottomLeft: { x: 0.3, y: 0.2, rotate: 1 },
            bottomRight: { x: -0.3, y: 0, rotate: -2 },
          });
          break;
        default:
          setMouthState("neutral");
          setLeftEyebrowOffset(0);
          setRightEyebrowOffset(0);
          setLogoOffsets({
            topLeft: { x: (Math.random() - 0.5) * 0.6, y: (Math.random() - 0.5) * 0.4, rotate: (Math.random() - 0.5) * 2 },
            topRight: { x: (Math.random() - 0.5) * 0.6, y: (Math.random() - 0.5) * 0.4, rotate: (Math.random() - 0.5) * 2 },
            bottomLeft: { x: (Math.random() - 0.5) * 0.4, y: (Math.random() - 0.5) * 0.3, rotate: (Math.random() - 0.5) * 1.5 },
            bottomRight: { x: (Math.random() - 0.5) * 0.4, y: (Math.random() - 0.5) * 0.3, rotate: (Math.random() - 0.5) * 1.5 },
          });
      }
    };

    const interval = setInterval(changeExpression, 1200 + Math.random() * 1200);
    return () => clearInterval(interval);
  }, [isThinking]);

  // Continuous subtle logo breathing
  useEffect(() => {
    if (isThinking) return;

    const breathe = () => {
      setLogoOffsets(prev => ({
        topLeft: {
          x: prev.topLeft.x + (Math.random() - 0.5) * 0.5,
          y: prev.topLeft.y + (Math.random() - 0.5) * 0.4,
          rotate: prev.topLeft.rotate + (Math.random() - 0.5) * 1.5,
        },
        topRight: {
          x: prev.topRight.x + (Math.random() - 0.5) * 0.5,
          y: prev.topRight.y + (Math.random() - 0.5) * 0.4,
          rotate: prev.topRight.rotate + (Math.random() - 0.5) * 1.5,
        },
        bottomLeft: {
          x: prev.bottomLeft.x + (Math.random() - 0.5) * 0.4,
          y: prev.bottomLeft.y + (Math.random() - 0.5) * 0.3,
          rotate: prev.bottomLeft.rotate + (Math.random() - 0.5) * 1,
        },
        bottomRight: {
          x: prev.bottomRight.x + (Math.random() - 0.5) * 0.4,
          y: prev.bottomRight.y + (Math.random() - 0.5) * 0.3,
          rotate: prev.bottomRight.rotate + (Math.random() - 0.5) * 1,
        },
      }));
    };

    const interval = setInterval(breathe, 400);
    return () => clearInterval(interval);
  }, [isThinking]);

  // Thinking animation
  useEffect(() => {
    if (!isThinking) return;

    const thinkingAnim = () => {
      setExpression("thinking");
      setMouthState("smirk");
      setLeftEyebrowOffset(-1.5);
      setRightEyebrowOffset(0.5);
      setLogoOffsets({
        topLeft: { x: -0.5, y: -0.3, rotate: -3 },
        topRight: { x: 1.2, y: -1, rotate: 6 },
        bottomLeft: { x: 0.4, y: 0.3, rotate: 2 },
        bottomRight: { x: -0.3, y: 0.4, rotate: -2 },
      });
      setTimeout(() => {
        setLeftEyebrowOffset(0.5);
        setRightEyebrowOffset(-1.5);
        setLogoOffsets({
          topLeft: { x: 0.6, y: -0.5, rotate: 4 },
          topRight: { x: -0.4, y: -0.2, rotate: -2 },
          bottomLeft: { x: -0.3, y: 0.5, rotate: -1 },
          bottomRight: { x: 0.5, y: 0.2, rotate: 3 },
        });
      }, 400);
    };

    const interval = setInterval(thinkingAnim, 800);
    thinkingAnim();
    return () => clearInterval(interval);
  }, [isThinking]);

  const eyeOffset = eyeDirection === "left" ? -1 : eyeDirection === "right" ? 1 : 0;

  // Clamp function for safety
  const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));

  const getMouthPath = () => {
    switch (mouthState) {
      case "smile":
        return "M-2 0 Q0 2 2 0";
      case "open":
        return "M-1.5 0.5 Q0 2 1.5 0.5 Q0 1.5 -1.5 0.5";
      case "surprised":
        return "M-1 0.5 Q0 2 1 0.5 Q0 1.5 -1 0.5";
      case "smirk":
        return "M-1.5 0.2 Q0.5 1.5 2 0";
      default:
        return "M-1.5 0 Q0 1.2 1.5 0";
    }
  };

  return (
    <div
      className={cn("relative flex-shrink-0", className)}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 28 28"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* DocScanner Logo - 4 parallelograms with independent movement */}
        <g transform="translate(4, 3) scale(0.095)">
          {/* Top left parallelogram */}
          <path
            d="M43.9767 5.81418C44.7712 2.40898 47.807 0 51.3037 0H141.185C146.031 0 149.613 4.51423 148.512 9.23344L125.69 107.043C124.895 110.448 121.86 112.857 118.363 112.857H28.4815C23.6355 112.857 20.0533 108.343 21.1545 103.624L43.9767 5.81418Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.topLeft.x, -2, 2)}px, ${clamp(logoOffsets.topLeft.y, -2, 2)}px) rotate(${clamp(logoOffsets.topLeft.rotate, -6, 6)}deg)`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200"
          />
          {/* Bottom right parallelogram */}
          <path
            d="M85.1681 137.957C85.9626 134.552 88.9984 132.143 92.4951 132.143H182.377C187.223 132.143 190.805 136.657 189.704 141.376L166.881 239.186C166.087 242.591 163.051 245 159.554 245H69.6729C64.8269 245 61.2447 240.485 62.3459 235.766L85.1681 137.957Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.bottomRight.x, -2, 2)}px, ${clamp(logoOffsets.bottomRight.y, -2, 2)}px) rotate(${clamp(logoOffsets.bottomRight.rotate, -6, 6)}deg)`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200"
          />
          {/* Top right parallelogram */}
          <path
            d="M146.242 54.6245C147.072 51.2653 150.086 48.9053 153.547 48.9053H201.247C206.133 48.9053 209.723 53.4901 208.551 58.2336L196.468 107.138C195.639 110.498 192.624 112.858 189.164 112.858H141.464C136.578 112.858 132.988 108.273 134.16 103.529L146.242 54.6245Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.topRight.x, -2, 2)}px, ${clamp(logoOffsets.topRight.y, -2, 2)}px) rotate(${clamp(logoOffsets.topRight.rotate, -6, 6)}deg)`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200"
          />
          {/* Bottom left parallelogram */}
          <path
            d="M14.387 137.862C15.2169 134.503 18.231 132.143 21.6912 132.143H69.3911C74.2773 132.143 77.8673 136.727 76.6953 141.471L64.613 190.376C63.7831 193.735 60.769 196.095 57.3088 196.095H9.60886C4.72269 196.095 1.13272 191.51 2.30466 186.767L14.387 137.862Z"
            fill="black"
            style={{
              transform: `translate(${clamp(logoOffsets.bottomLeft.x, -2, 2)}px, ${clamp(logoOffsets.bottomLeft.y, -2, 2)}px) rotate(${clamp(logoOffsets.bottomLeft.rotate, -6, 6)}deg)`,
              transformOrigin: "center",
            }}
            className="transition-transform duration-200"
          />
        </g>

        {/* Left eye with eyebrow */}
        <g transform="translate(9, 8)">
          {/* Left eyebrow */}
          <path
            d={`M-0.5 ${-1.5 + leftEyebrowOffset} Q1.5 ${-2.5 + leftEyebrowOffset} 3.5 ${-1.5 + leftEyebrowOffset * 0.5}`}
            stroke="white"
            strokeWidth="1.5"
            strokeLinecap="round"
            fill="none"
            className="transition-all duration-200"
          />
          {isBlinking ? (
            <path d="M0 1 Q1.5 2 3 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" fill="none" />
          ) : (
            <ellipse cx={1.5 + eyeOffset} cy="1" rx={expression === "surprised" ? 2.2 : 2} ry={expression === "surprised" ? 2 : 1.8} fill="white" className="transition-all duration-150" />
          )}
        </g>

        {/* Right eye with eyebrow */}
        <g transform="translate(17, 9)">
          {/* Right eyebrow */}
          <path
            d={`M-0.5 ${-1.5 + rightEyebrowOffset * 0.5} Q1.5 ${-2.5 + rightEyebrowOffset} 3.5 ${-1.5 + rightEyebrowOffset}`}
            stroke="white"
            strokeWidth="1.5"
            strokeLinecap="round"
            fill="none"
            className="transition-all duration-200"
          />
          {isBlinking ? (
            <path d="M0 1 Q1.5 2 3 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" fill="none" />
          ) : (
            <ellipse cx={1.5 + eyeOffset} cy="1" rx={expression === "surprised" ? 2.2 : 2} ry={expression === "surprised" ? 2 : 1.8} fill="white" className="transition-all duration-150" />
          )}
        </g>

        {/* Mouth */}
        <g transform="translate(14, 20)">
          <path
            d={getMouthPath()}
            stroke="white"
            strokeWidth="1.4"
            strokeLinecap="round"
            fill={mouthState === "open" || mouthState === "surprised" ? "white" : "none"}
            className="transition-all duration-150"
          />
        </g>

        {/* Thinking dots */}
        {isThinking && (
          <g className="animate-pulse">
            <circle cx="24" cy="4" r="1.8" fill="black" opacity="0.7" />
            <circle cx="26" cy="7" r="1.2" fill="black" opacity="0.4" />
          </g>
        )}
      </svg>
    </div>
  );
}

export default AIAvatar;
