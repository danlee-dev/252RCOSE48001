# DocScanner.ai Design System Guide

이 문서는 DocScanner.ai 프론트엔드의 디자인 시스템을 정의합니다.

---

## 1. 디자인 철학

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| Trustworthy & Professional | 법률 서비스 플랫폼으로서 신뢰감을 주는 정제된 디자인 |
| Non-AI Aesthetic | 과도한 네온, 사이버펑크 스타일 지양 |
| Clean & Refined | 충분한 여백, 복잡한 장식 배제, 정보 가독성 최우선 |
| Micro-Interactions | 과하지 않은 부드러운 애니메이션으로 피드백 제공 |

---

## 2. 색상 팔레트 (Color Palette)

### 2.1 Primary Colors

| 용도 | 색상 | HEX | 사용처 |
|------|------|-----|--------|
| Primary Green (Dark) | ![#3d5a47](https://via.placeholder.com/20/3d5a47/3d5a47.png) | `#3d5a47` | 버튼, 액티브 상태, 진행률 바, 아이콘 강조 |
| Primary Green (Light) | ![#4a6b52](https://via.placeholder.com/20/4a6b52/4a6b52.png) | `#4a6b52` | 호버 상태 |
| Primary Foreground | ![#ffffff](https://via.placeholder.com/20/ffffff/ffffff.png) | `#ffffff` | Primary 위의 텍스트 |

### 2.2 Background Colors

| 용도 | 색상 | HEX | 사용처 |
|------|------|-----|--------|
| Page Background | ![#f8f9fa](https://via.placeholder.com/20/f8f9fa/f8f9fa.png) | `#f8f9fa` | 전체 페이지 배경 |
| Card Background | ![#ffffff](https://via.placeholder.com/20/ffffff/ffffff.png) | `#ffffff` | 카드 배경 |
| Icon Container | ![#e8f0ea](https://via.placeholder.com/20/e8f0ea/e8f0ea.png) | `#e8f0ea` | 아이콘 컨테이너 배경 |
| Muted Background | HSL(0 0% 96%) | `#f5f5f5` | 비활성 영역, 보조 배경 |

### 2.3 Badge Colors (Toned-down)

위험도/상태 표시에 사용되는 절제된 배지 색상입니다.

| 상태 | Background | Text | Border |
|------|------------|------|--------|
| Success (Completed/LOW) | `#e8f5ec` | `#3d7a4a` | `#c8e6cf` |
| Warning (Pending/MEDIUM) | `#fef7e0` | `#9a7b2d` | `#f5e6b8` |
| Danger (Failed/HIGH) | `#fdedec` | `#b54a45` | `#f5c6c4` |
| Neutral (SAFE) | `#e8f0ea` | `#3d5a47` | `#c8e6cf` |

**CSS 클래스:**
```css
.badge-success { background: #e8f5ec; color: #3d7a4a; border: 1px solid #c8e6cf; }
.badge-warning { background: #fef7e0; color: #9a7b2d; border: 1px solid #f5e6b8; }
.badge-danger  { background: #fdedec; color: #b54a45; border: 1px solid #f5c6c4; }
.badge-neutral { background: #e8f0ea; color: #3d5a47; border: 1px solid #c8e6cf; }
```

### 2.4 Chart Colors

차트/그래프에서 위험도 시각화에 사용됩니다.

| 위험도 | 색상 | HEX |
|--------|------|-----|
| Safe/Low | ![#4a9a5b](https://via.placeholder.com/20/4a9a5b/4a9a5b.png) | `#4a9a5b` |
| Medium | ![#d4a84d](https://via.placeholder.com/20/d4a84d/d4a84d.png) | `#d4a84d` |
| High/Danger | ![#c94b45](https://via.placeholder.com/20/c94b45/c94b45.png) | `#c94b45` |

### 2.5 Text Colors

| 용도 | HSL | 사용처 |
|------|-----|--------|
| Primary Text | `hsl(0 0% 9%)` / `#171717` | 헤딩, 중요 텍스트 |
| Secondary Text | `#6b7280` | 부가 설명 |
| Muted Text | `#9ca3af` | 비활성 텍스트, 플레이스홀더 |

### 2.6 Highlight Colors (인라인 형광펜)

문서 내 위험 조항 하이라이팅에 사용됩니다.

| 위험도 | 기본 | 호버 | 활성 |
|--------|------|------|------|
| High | `rgba(201, 75, 69, 0.2)` | `rgba(201, 75, 69, 0.3)` | `rgba(201, 75, 69, 0.5)` |
| Medium | `rgba(212, 168, 77, 0.2)` | `rgba(212, 168, 77, 0.3)` | `rgba(212, 168, 77, 0.5)` |
| Low | `rgba(74, 154, 91, 0.2)` | `rgba(74, 154, 91, 0.3)` | `rgba(74, 154, 91, 0.5)` |

---

## 3. 타이포그래피 (Typography)

### 3.1 폰트 패밀리

| 용도 | 폰트 | 설명 |
|------|------|------|
| 기본 (한글/영문) | **Pretendard Variable** | 모든 UI 텍스트의 기본 폰트 |
| 숫자/영문 강조 | **Inter** | 통계 수치, 숫자 표시에 사용 |

```css
/* 기본 설정 */
font-family: 'Pretendard Variable', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;

/* 숫자/영문 전용 */
.font-inter {
  font-family: 'Inter', 'Pretendard Variable', sans-serif;
  font-feature-settings: 'tnum' on, 'lnum' on;
}

/* 통계 숫자 (대형) */
.stat-number {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-feature-settings: 'tnum' on;
}
```

### 3.2 자간 (Letter Spacing)

| 용도 | 값 | Tailwind |
|------|-----|----------|
| 기본 자간 | `-0.025em` | `tracking-[-0.025em]` 또는 `tracking-tight` |
| 숫자 자간 | `-0.02em` | `tracking-[-0.02em]` |

> Pretendard 폰트가 가장 정돈되어 보이는 자간 수치입니다. 모든 텍스트에 일괄 적용합니다.

### 3.3 폰트 크기 및 굵기

| 요소 | 크기 | 굵기 | 예시 |
|------|------|------|------|
| 페이지 제목 | `1.5rem` (24px) | 700 (Bold) | 대시보드, 분석 결과 |
| 섹션 제목 | `1.125rem` (18px) | 600 (Semibold) | 카드 타이틀 |
| 본문 | `0.875rem` (14px) | 400 (Regular) | 일반 텍스트 |
| 보조 텍스트 | `0.75rem` (12px) | 500 (Medium) | 라벨, 캡션 |
| 작은 텍스트 | `0.6875rem` (11px) | 500 (Medium) | 배지, 태그 |

### 3.4 행간 (Line Height)

| 용도 | 값 |
|------|-----|
| 기본 | `1.5` |
| 본문/산문 | `1.6` |
| 계약서 문서 | `1.8` |
| 제목 | `1.3` |

---

## 4. 테두리 반경 (Border Radius)

### 4.1 기본 반경 시스템

| 토큰 | 값 | 사용처 |
|------|-----|--------|
| `--radius-sm` | `4px` | 작은 요소, 태그 |
| `--radius` | `8px` | 버튼, 입력 필드 |
| `--radius-md` | `12px` | 일반 카드 |
| `--radius-lg` | `16px` | 모바일 카드 |
| `--radius-xl` | `18px` | Apple 스타일 카드 |
| `--radius-2xl` | `20px` | V2 카드 |
| `--radius-3xl` | `24px` | 인증 카드, Dock |
| `--radius-4xl` | `28px` | Liquid Glass 카드 |

### 4.2 컴포넌트별 반경

| 컴포넌트 | 반경 | 비고 |
|----------|------|------|
| 버튼 (기본) | `8px` | rounded-md |
| 버튼 (Auth) | `12px` | |
| 입력 필드 | `8px` ~ `14px` | Liquid: 14px |
| 배지 | `9999px` | rounded-full (완전 원형) |
| 일반 카드 | `12px` | .card |
| Apple 카드 | `18px` | .card-apple |
| V2 카드 | `20px` | .card-v2 |
| Auth 카드 | `24px` | .auth-card |
| Liquid Glass 카드 | `28px` | .liquid-glass-card |
| Dock 아이콘 | `14px` | |
| Dock 컨테이너 | `24px` | |
| 차트 바 | `6px 6px 0 0` | 상단만 둥글게 |

---

## 5. 그림자 (Shadows)

### 5.1 기본 그림자 시스템

```css
/* Soft - 기본 카드 */
.shadow-soft {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06);
}

/* Medium - 호버 상태 */
.shadow-medium {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04);
}

/* Strong - 강조 요소 */
.shadow-strong {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.06);
}

/* Inner Soft - 입력 필드 등 */
.shadow-inner-soft {
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.04);
}
```

### 5.2 컴포넌트별 그림자

| 컴포넌트 | 기본 상태 | 호버 상태 |
|----------|-----------|-----------|
| 일반 카드 | `0 2px 8px rgba(0,0,0,0.04)` | `0 4px 16px rgba(0,0,0,0.08)` |
| Apple 카드 | `0 1px 3px, 0 4px 12px` + inset | `0 4px 8px, 0 8px 24px` + inset |
| V2 카드 | `0 1px 3px, 0 4px 12px` | `0 4px 12px, 0 8px 24px` |
| Auth 카드 | `0 4px 6px, 0 10px 15px, 0 20px 25px` | - |
| Dock | `0 8px 32px, 0 2px 8px` + inset | `0 12px 40px` + inset |

### 5.3 색상별 글로우 그림자 (차트)

```css
/* 차트 바/원형 그림자 */
.shadow-success { box-shadow: 0 0 10px -2px rgba(34, 197, 94, 0.4); }
.shadow-warning { box-shadow: 0 0 10px -2px rgba(245, 158, 11, 0.4); }
.shadow-danger  { box-shadow: 0 0 10px -2px rgba(239, 68, 68, 0.4); }
.shadow-default { box-shadow: 0 0 10px -2px rgba(23, 23, 23, 0.3); }
```

---

## 6. 간격 시스템 (Spacing)

Tailwind CSS 기본 간격 시스템을 사용합니다.

| 토큰 | 값 | 사용처 |
|------|-----|--------|
| `1` | `4px` | 아이콘과 텍스트 사이 |
| `2` | `8px` | 인라인 요소 간격 |
| `3` | `12px` | 작은 패딩 |
| `4` | `16px` | 기본 패딩, 카드 내부 |
| `5` | `20px` | 섹션 간격 |
| `6` | `24px` | 카드 패딩 |
| `8` | `32px` | 큰 섹션 간격 |
| `10` | `40px` | 페이지 여백 |
| `12` | `48px` | 대형 컴포넌트 간격 |

---

## 7. 컴포넌트 스타일

### 7.1 카드 (Cards)

#### 일반 카드 (.card)
```css
.card {
  background: white;
  border-radius: 12px;
  border: 1px solid hsl(var(--border));
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  transition: all 0.2s ease;
}

.card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  border-color: hsl(0 0% 85%);
}
```

#### Apple 스타일 카드 (.card-apple)
```css
.card-apple {
  background: white;
  border-radius: 18px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.04),
    0 4px 12px rgba(0, 0, 0, 0.03),
    inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.card-apple:hover {
  transform: translateY(-2px);
  box-shadow:
    0 4px 8px rgba(0, 0, 0, 0.06),
    0 8px 24px rgba(0, 0, 0, 0.08),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
}
```

#### V2 카드 (.card-v2)
```css
.card-v2 {
  background: white;
  border-radius: 20px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.02),
    0 4px 12px rgba(0, 0, 0, 0.03);
}
```

### 7.2 버튼 (Buttons)

#### 버튼 사이즈

| 사이즈 | 높이 | 패딩 | 용도 |
|--------|------|------|------|
| sm | `36px` | `px-3` | 작은 액션 |
| default | `40px` | `px-4 py-2` | 일반 |
| lg | `44px` | `px-8` | 주요 CTA |
| icon | `40x40px` | - | 아이콘 버튼 |

#### 버튼 변형

| 변형 | 배경 | 테두리 | 용도 |
|------|------|--------|------|
| default | `hsl(var(--primary))` | - | 주요 액션 |
| secondary | `hsl(var(--secondary))` | - | 보조 액션 |
| outline | 투명 | `border-input` | 선택적 액션 |
| ghost | 투명 | - | 미묘한 액션 |
| destructive | `hsl(var(--destructive))` | - | 삭제, 위험 액션 |

#### Auth 버튼 (.auth-button)
```css
.auth-button {
  width: 100%;
  padding: 1rem 1.5rem;
  font-size: 0.9375rem;
  font-weight: 600;
  color: white;
  background: hsl(0 0% 10%);
  border-radius: 12px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.auth-button:hover:not(:disabled) {
  background: hsl(0 0% 5%);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.4);
}
```

### 7.3 입력 필드 (Input Fields)

#### 기본 입력 필드 (.input-field)
```css
.input-field {
  width: 100%;
  padding: 0.625rem 0.875rem;
  border: 1px solid hsl(var(--border));
  border-radius: 8px;
  background: white;
  transition: all 0.2s ease;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

.input-field:hover {
  border-color: hsl(0 0% 80%);
}

.input-field:focus {
  outline: none;
  border-color: hsl(var(--primary));
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.05);
}
```

#### Floating 라벨 입력 필드
```css
.floating-input {
  padding: 1.25rem 1rem 0.75rem 1rem;
  border: 1.5px solid hsl(0 0% 88%);
  border-radius: 12px;
}

.floating-input:focus {
  border-color: hsl(0 0% 20%);
  box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.04);
}
```

#### Liquid Glass 입력 필드
```css
.liquid-input {
  background: linear-gradient(135deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.35) 100%);
  backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.5);
  border-radius: 14px;
}
```

### 7.4 배지 (Badges)

```css
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.625rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
  transition: all 0.15s ease;
}
```

---

## 8. Dock 사이드바 (Icon Dock)

macOS Dock 스타일의 Liquid Glass 효과 사이드바입니다.

```css
.icon-dock {
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 12px 8px;
  background: linear-gradient(165deg,
    rgba(255, 255, 255, 0.75) 0%,
    rgba(255, 255, 255, 0.5) 30%,
    rgba(255, 255, 255, 0.4) 100%
  );
  backdrop-filter: blur(24px) saturate(180%);
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.08),
    0 2px 8px rgba(0, 0, 0, 0.04),
    inset 0 1px 1px rgba(255, 255, 255, 0.8);
}

.dock-icon {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  color: #9ca3af;
}

.dock-icon.active {
  color: #1a1a1a;
  background: rgba(0, 0, 0, 0.06);
}
```

---

## 9. 글래스모피즘 (Glassmorphism)

### 9.1 기본 Glass

```css
.glass {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.5);
}

.glass-subtle {
  background: rgba(255, 255, 255, 0.4);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.glass-card {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.03),
    0 10px 15px -3px rgba(0, 0, 0, 0.05),
    inset 0 1px 0 0 rgba(255, 255, 255, 0.6);
}
```

### 9.2 Liquid Glass

Apple Vision Pro 스타일의 고급 유리 효과입니다.

```css
.liquid-glass-card {
  background: linear-gradient(145deg,
    rgba(255, 255, 255, 0.4) 0%,
    rgba(255, 255, 255, 0.15) 100%
  );
  backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 28px;
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.04),
    0 12px 48px rgba(0, 0, 0, 0.06),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
}

/* 상단 하이라이트 (::before) */
.liquid-glass-card::before {
  background: linear-gradient(180deg,
    rgba(255, 255, 255, 0.35) 0%,
    rgba(255, 255, 255, 0) 100%
  );
  height: 40%;
}
```

---

## 10. 애니메이션 (Animations)

### 10.1 Easing Functions

| 이름 | 값 | 용도 |
|------|-----|------|
| ease | `ease` | 기본 |
| ease-out | `ease-out` | 등장 |
| smooth | `cubic-bezier(0.4, 0, 0.2, 1)` | 부드러운 전환 |
| bounce | `cubic-bezier(0.34, 1.56, 0.64, 1)` | 탄성 효과 |

### 10.2 주요 애니메이션

```css
/* 페이드 인 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* 위로 슬라이드 */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 스케일 인 */
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}

/* 펄스 */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 플로트 */
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

/* 스캔 라인 (AR 스캐너) */
@keyframes scanLine {
  0% { top: 8%; }
  50% { top: 85%; }
  100% { top: 8%; }
}

/* Liquid Blob (배경) */
@keyframes liquidMove {
  0%, 100% { transform: translate(0, 0) scale(1); border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%; }
  25% { transform: translate(10px, -20px) scale(1.05); }
  50% { transform: translate(-10px, 10px) scale(0.95); }
  75% { transform: translate(5px, -10px) scale(1.02); }
}
```

### 10.3 애니메이션 클래스

| 클래스 | 지속시간 | 용도 |
|--------|----------|------|
| `.animate-fadeIn` | 0.2s | 기본 등장 |
| `.animate-fadeInUp` | 0.3s | 카드, 모달 등장 |
| `.animate-slideUp` | 0.2s | 바텀시트 |
| `.animate-scaleIn` | 0.2s | 팝업 |
| `.animate-pulse` | 2s infinite | 로딩 상태 |
| `.animate-float` | 3s infinite | AI 아바타 |
| `.animate-scan-line` | 2s infinite | 스캐너 |

### 10.4 호버 효과

```css
/* 리프트 */
.hover-lift:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 스케일 */
.hover-scale:hover {
  transform: scale(1.02);
}

/* 글로우 */
.hover-glow:hover {
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.05);
}
```

---

## 11. 그라데이션 배경

### 11.1 페이지 배경 (Gradient Blob)

```css
.gradient-blob-bg {
  background: #f8f9fa;
}

.gradient-blob-bg::before {
  background:
    radial-gradient(ellipse 80% 60% at 5% 15%, rgba(220, 235, 224, 0.95) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 95% 85%, rgba(254, 243, 210, 0.7) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 60% 5%, rgba(220, 240, 226, 0.8) 0%, transparent 45%),
    radial-gradient(ellipse 40% 30% at 80% 30%, rgba(250, 235, 215, 0.6) 0%, transparent 40%);
}
```

### 11.2 노이즈 텍스처 오버레이

```css
.gradient-blob-bg::after {
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512'..."); /* fractalNoise */
  opacity: 0.35;
}
```

---

## 12. 반응형 브레이크포인트

| 브레이크포인트 | 값 | 주요 변화 |
|----------------|-----|-----------|
| xs | `480px` | 작은 모바일 |
| sm | `640px` | 모바일 |
| md | `768px` | 태블릿 |
| lg | `1024px` | 데스크톱 (Dock 표시) |
| xl | `1280px` | 대형 데스크톱 |
| 2xl | `1536px` | 초대형 |

### 모바일 우선 고려사항

```css
/* 모바일에서 호버 효과 비활성화 */
@media (max-width: 640px) {
  .card:hover {
    transform: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  }
}

/* 터치 타겟 최소 크기 */
.touch-target {
  min-height: 44px;
  min-width: 44px;
}

/* Safe Area (노치 대응) */
.safe-area-bottom {
  padding-bottom: env(safe-area-inset-bottom);
}
```

---

## 13. 스크롤바 스타일

```css
/* 기본: 투명 */
::-webkit-scrollbar-thumb {
  background: transparent;
}

/* 호버 시 표시 */
*:hover::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
}

*:hover::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.25);
}

/* 스크롤바 숨김 */
.scrollbar-hide {
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
```

---

## 14. 접근성

### 14.1 포커스 스타일

```css
:focus-visible {
  outline: 2px solid hsl(var(--ring));
  outline-offset: 2px;
}
```

### 14.2 색상 대비

모든 텍스트-배경 조합은 WCAG 2.1 AA 기준을 충족합니다.

| 조합 | 대비 비율 |
|------|-----------|
| Primary Text (#171717) on White | 16.1:1 |
| Secondary Text (#6b7280) on White | 5.9:1 |
| Badge Danger Text (#b54a45) on #fdedec | 4.6:1 |
| Badge Success Text (#3d7a4a) on #e8f5ec | 4.8:1 |

---

## 15. 디자인 토큰 요약

### CSS 변수 (globals.css :root)

```css
:root {
  --background: 0 0% 100%;
  --foreground: 0 0% 9%;
  --muted: 0 0% 96%;
  --muted-foreground: 0 0% 45%;
  --border: 0 0% 90%;
  --input: 0 0% 90%;
  --primary: 0 0% 9%;
  --primary-foreground: 0 0% 100%;
  --secondary: 0 0% 96%;
  --secondary-foreground: 0 0% 9%;
  --destructive: 0 84% 60%;
  --success: 142 76% 36%;
  --warning: 38 92% 50%;
  --ring: 0 0% 9%;
  --radius: 0.5rem;
}
```

---

*마지막 업데이트: 2025-12-17*
