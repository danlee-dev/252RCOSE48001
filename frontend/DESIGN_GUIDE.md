# DocScanner.ai Design System Guide

이 문서는 DocScanner.ai 프론트엔드의 디자인 시스템을 정의합니다. 모든 UI 구현은 이 가이드를 따라야 합니다.

---

## 1. Design Philosophy

### Core Principles

**Trustworthy & Professional**
- 법률 서비스 플랫폼답게 신뢰감을 주는 정제된 디자인
- 과도한 장식보다 정보의 가독성을 최우선

**Non-AI Aesthetic**
- 네온, 사이버펑크, 과도한 그라데이션 지양
- 깔끔하고 절제된 컬러 사용

**Subtle & Refined**
- 미세한 그림자와 부드러운 곡선
- 과하지 않은 마이크로 인터랙션

---

## 2. Color System

### Page Background

```css
/* Warm beige background - subtle and professional */
background-color: #f2f1ee;
```

### Primary Colors (Neutrals)

| Token | Value | Usage |
|-------|-------|-------|
| `gray-900` | `#171717` | Primary text, headings |
| `gray-700` | `#404040` | Secondary text |
| `gray-500` | `#737373` | Tertiary text, labels |
| `gray-400` | `#a3a3a3` | Placeholder, disabled |
| `gray-200` | `#e5e5e5` | Borders, dividers |
| `gray-100` | `#f5f5f5` | Card backgrounds, surfaces |
| `white` | `#ffffff` | Card backgrounds |

### Semantic Colors

| Token | Value | Usage |
|-------|-------|-------|
| Success | `#22c55e` | Low risk, completed |
| Warning | `#f59e0b` | Medium risk, caution |
| Danger | `#ef4444` | High risk, errors |
| Info | `#3b82f6` | Processing, links |

### Semantic Color Backgrounds (Subtle)

항상 투명도를 사용하여 배경색을 적용합니다.

```css
/* Success */
background: rgba(34, 197, 94, 0.08);
border-left: 3px solid #22c55e;

/* Warning */
background: rgba(245, 158, 11, 0.08);
border-left: 3px solid #f59e0b;

/* Danger */
background: rgba(239, 68, 68, 0.08);
border-left: 3px solid #ef4444;
```

### Color-Matched Shadows

요소의 색상에 맞는 그림자를 사용합니다.

```css
/* Success shadow */
box-shadow: 0 4px 14px -3px rgba(34, 197, 94, 0.25);

/* Warning shadow */
box-shadow: 0 4px 14px -3px rgba(245, 158, 11, 0.25);

/* Danger shadow */
box-shadow: 0 4px 14px -3px rgba(239, 68, 68, 0.25);
```

---

## 3. Typography

### Font Family

```css
font-family: 'Pretendard Variable', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
```

### Letter Spacing (Critical)

모든 텍스트에 `-0.025em` (tracking-tight) 적용

```css
letter-spacing: -0.025em;
```

### Font Scale

| Level | Size | Weight | Usage |
|-------|------|--------|-------|
| Display | 32px | 700 | Hero sections |
| H1 | 24px | 700 | Page titles |
| H2 | 20px | 600 | Section headers |
| H3 | 16px | 600 | Card titles |
| Body | 14px | 400 | Default text |
| Small | 13px | 400 | Secondary info |
| Caption | 12px | 500 | Labels, badges |
| Tiny | 11px | 500 | Timestamps |

### Font Weight

| Weight | Value | Usage |
|--------|-------|-------|
| Regular | 400 | Body text |
| Medium | 500 | Labels, captions |
| Semibold | 600 | Headings, emphasis |
| Bold | 700 | Display, H1 |

---

## 4. Spacing System

8px 기반의 일관된 간격 시스템

| Token | Value | Usage |
|-------|-------|-------|
| `space-1` | 4px | Inline icon gaps |
| `space-2` | 8px | Tight spacing |
| `space-3` | 12px | Default gaps |
| `space-4` | 16px | Component padding |
| `space-5` | 20px | Card padding |
| `space-6` | 24px | Section spacing |
| `space-8` | 32px | Large gaps |
| `space-10` | 40px | Section margins |

### Component Spacing

```css
/* Card padding */
padding: 20px; /* p-5 */

/* Section margin */
margin-bottom: 32px; /* mb-8 */

/* Grid gap */
gap: 16px; /* gap-4 */
```

---

## 5. Border Radius

Apple 스타일의 부드러운 연속 곡선(Squircle) 느낌을 주기 위해 적절한 라운드를 사용합니다.

| Token | Value | Usage |
|-------|-------|-------|
| `rounded` | 8px | Badges, small elements |
| `rounded-lg` | 12px | Buttons, inputs |
| `rounded-xl` | 16px | Icon containers |
| `rounded-[18px]` | 18px | Cards, containers |
| `rounded-full` | 9999px | Pills, avatars |

### Card Radius

```css
/* Standard card - 18px radius (Apple-style) */
.card {
  border-radius: 18px; /* rounded-[18px] */
}

/* Icon containers - 16px */
.icon-container {
  border-radius: 16px; /* rounded-xl */
}

/* Small elements (badges) - 8px */
.badge {
  border-radius: 8px; /* rounded */
}
```

---

## 6. Shadows

### Standard Shadows

```css
/* Soft - Cards at rest */
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04),
            0 1px 2px rgba(0, 0, 0, 0.06);

/* Medium - Cards on hover */
box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08),
            0 2px 4px rgba(0, 0, 0, 0.04);

/* Strong - Elevated elements */
box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12),
            0 4px 8px rgba(0, 0, 0, 0.06);
```

### Glass Shadows (Liquid Glass 컴포넌트)

```css
box-shadow:
  0 4px 24px rgba(0, 0, 0, 0.04),
  0 12px 48px rgba(0, 0, 0, 0.06),
  inset 0 1px 0 rgba(255, 255, 255, 0.5),
  inset 0 -1px 0 rgba(0, 0, 0, 0.02);
```

---

## 7. Components

### Apple-Style Card (card-apple)

기본 카드 스타일입니다. 모든 콘텐츠 컨테이너에 사용합니다. Hover 시 살짝 떠오르는 효과와 inset shadow로 입체감을 줍니다.

```css
/* Tailwind: card-apple p-4 sm:p-5 */
.card-apple {
  background: white;
  border-radius: 18px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.04),
    0 4px 12px rgba(0, 0, 0, 0.03),
    inset 0 1px 0 rgba(255, 255, 255, 0.8);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.card-apple:hover {
  transform: translateY(-2px);
  box-shadow:
    0 4px 8px rgba(0, 0, 0, 0.06),
    0 8px 24px rgba(0, 0, 0, 0.08),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
  border-color: rgba(0, 0, 0, 0.08);
}

.card-apple:active {
  transform: translateY(0);
  box-shadow:
    0 1px 2px rgba(0, 0, 0, 0.05),
    0 2px 8px rgba(0, 0, 0, 0.04),
    inset 0 1px 0 rgba(255, 255, 255, 0.7);
}
```

### Usage

```tsx
{/* Basic card */}
<div className="card-apple p-5">
  Card content
</div>

{/* Card with active press effect */}
<div className="card-apple p-4 active:scale-[0.99] cursor-pointer">
  Clickable card
</div>
```

### Stat Card

통계 숫자를 표시하는 카드입니다.

```tsx
<div className="card-apple p-4 sm:p-5">
  <div className="flex items-start justify-between mb-2">
    <div>
      <p className="text-xs font-medium text-gray-500 mb-1">Label</p>
      <p className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
        {value}
      </p>
    </div>
    {/* Optional: Sparkline or Icon */}
  </div>
  <p className="text-[11px] text-gray-400">Description</p>
</div>
```

### Badge

상태를 나타내는 배지입니다.

```css
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 500;
}

/* Variants */
.badge-success { background: hsl(142 76% 95%); color: hsl(142 60% 30%); }
.badge-warning { background: hsl(38 92% 95%); color: hsl(30 80% 40%); }
.badge-danger { background: hsl(0 84% 95%); color: hsl(0 70% 45%); }
.badge-neutral { background: hsl(0 0% 96%); color: hsl(0 0% 40%); }
```

### Button

```css
/* Primary */
.liquid-button {
  background: linear-gradient(135deg, hsl(0 0% 12%) 0%, hsl(0 0% 18%) 100%);
  border-radius: 14px;
  color: white;
  font-weight: 600;
  box-shadow:
    0 2px 8px rgba(0, 0, 0, 0.15),
    0 8px 24px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

/* Secondary */
.btn-secondary {
  background: white;
  border: 1px solid hsl(0 0% 90%);
  border-radius: 12px;
  color: hsl(0 0% 9%);
  font-weight: 500;
}
```

### Input

```css
.liquid-input {
  background: rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.4);
  border-radius: 14px;
  padding: 10px 14px;
  transition: all 0.3s ease;
}

.liquid-input:focus {
  background: rgba(255, 255, 255, 0.7);
  border-color: rgba(0, 0, 0, 0.15);
  box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.03);
}
```

---

## 8. Charts Design

차트는 패키지 없이 직접 구현합니다. 다음 스타일을 따릅니다.

### Chart Container

```css
.chart-container {
  position: relative;
  padding: 20px;
  background: rgba(255, 255, 255, 0.5);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.3);
}
```

### Bar Chart Style

```css
.bar {
  border-radius: 6px 6px 0 0;
  background: linear-gradient(180deg, #e5e5e5 0%, #d4d4d4 100%);
  transition: all 0.3s ease;
}

.bar.active,
.bar:hover {
  background: linear-gradient(180deg, #404040 0%, #171717 100%);
  box-shadow: 0 -4px 16px -4px rgba(23, 23, 23, 0.3);
}
```

### Donut/Ring Chart Style

```css
.ring-chart {
  /* Using SVG stroke */
  stroke-linecap: round;
  stroke-width: 12;
  fill: none;
}

/* Track */
.ring-track {
  stroke: rgba(0, 0, 0, 0.06);
}

/* Progress segments with gradient */
.ring-segment {
  stroke: url(#gradient);
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
}
```

### Chart Colors (Gradient)

차트에 사용할 그라데이션 색상

```css
/* Primary gradient (grayscale) */
--chart-primary: linear-gradient(180deg, #404040 0%, #171717 100%);

/* Secondary gradient */
--chart-secondary: linear-gradient(180deg, #737373 0%, #525252 100%);

/* Tertiary */
--chart-tertiary: linear-gradient(180deg, #a3a3a3 0%, #737373 100%);

/* Background/inactive */
--chart-bg: linear-gradient(180deg, #f5f5f5 0%, #e5e5e5 100%);
```

### Chart Typography

- 축 라벨: 11px, `gray-400`, medium
- 값 라벨: 12px, `gray-700`, semibold
- 차트 제목: 14px, `gray-900`, semibold

---

## 9. Layout Patterns

### Dashboard Grid

```tsx
{/* Main stats - 2x2 on mobile, 4 columns on desktop */}
<div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
  <StatCard />
  <StatCard />
  <StatCard />
  <StatCard />
</div>

{/* Content sections - 2 columns on desktop */}
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
  <ChartSection />
  <ListSection />
</div>
```

### Section Header

```tsx
<div className="flex items-center justify-between mb-4">
  <h2 className="text-lg font-semibold text-gray-900 tracking-tight">
    Section Title
  </h2>
  <Link className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1">
    View All
    <IconChevronRight size={16} />
  </Link>
</div>
```

### Card Grouping (Visual Hierarchy)

관련 항목은 시각적으로 그룹화합니다.

```tsx
{/* Group container with subtle background */}
<div className="bg-gray-50/50 rounded-2xl p-4 space-y-3">
  <h3 className="text-sm font-medium text-gray-500 mb-3">Group Title</h3>
  <ItemCard />
  <ItemCard />
</div>
```

---

## 10. Animation & Transitions

### Duration

| Type | Duration | Usage |
|------|----------|-------|
| Fast | 150ms | Hover states |
| Normal | 200ms | Default transitions |
| Smooth | 300ms | Page transitions, modals |

### Easing

```css
/* Standard */
transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);

/* Bounce (for buttons) */
transition-timing-function: cubic-bezier(0.34, 1.56, 0.64, 1);
```

### Hover Effects

```css
/* Card hover */
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

/* Subtle scale */
.card:hover {
  transform: scale(1.01);
}

/* Active press */
.card:active {
  transform: scale(0.99);
}
```

---

## 11. Responsive Breakpoints

| Breakpoint | Value | Usage |
|------------|-------|-------|
| `sm` | 640px | Mobile landscape |
| `md` | 768px | Tablet |
| `lg` | 1024px | Desktop |
| `xl` | 1280px | Large desktop |

### Mobile-First Approach

항상 모바일 우선으로 스타일을 작성합니다.

```tsx
<div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
```

---

## 12. Do's and Don'ts

### DO

- 여백을 충분히 사용하여 콘텐츠에 숨 쉴 공간 제공
- 정보의 위계를 크기와 굵기로 명확히 구분
- 색상 일관성 유지 (시맨틱 컬러 사용)
- 미세한 그림자와 블러로 깊이감 표현
- 부드러운 곡선(border-radius) 사용

### DON'T

- 네온 컬러, 과도한 그라데이션 사용
- 지나치게 작은 폰트 사용 (최소 11px)
- 불필요한 장식 요소 추가
- 급격한 애니메이션 사용
- 색상만으로 상태 구분 (아이콘 병행)

---

## 13. Icon Guidelines

### Size

| Context | Size | Usage |
|---------|------|-------|
| Inline | 14-16px | Text alongside |
| Button | 18-20px | Inside buttons |
| Card | 20-24px | Card indicators |
| Empty State | 28-32px | Placeholder |
| Feature | 40-48px | Feature highlights |

### Color

- Primary actions: `gray-900`
- Secondary: `gray-500`
- Disabled: `gray-300`
- Success: `#22c55e`
- Warning: `#f59e0b`
- Danger: `#ef4444`

---

## Example Implementation

```tsx
// Dashboard Stat Card
<div className="card-apple p-5 group">
  <div className="flex items-center justify-between mb-3">
    <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center
                    group-hover:scale-110 transition-transform">
      <IconCheck size={20} className="text-green-600" />
    </div>
    <span className="text-xs text-green-600 font-medium">Completed</span>
  </div>
  <p className="text-3xl font-bold text-gray-900 tracking-tight">24</p>
  <p className="text-xs text-gray-500 mt-1">Analysis completed</p>
</div>
```

---

*이 가이드는 지속적으로 업데이트됩니다.*
