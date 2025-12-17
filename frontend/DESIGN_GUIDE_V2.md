# DocScanner.ai Design System V2

Reference: Modern Property Dashboard Style

---

## 1. Design Philosophy

### Core Principles

**Spacious & Balanced**
- 화면을 효율적으로 채우되 비어 보이지 않게
- 요소 간 적절한 간격으로 시각적 안정감 제공

**Soft & Organic**
- 단색 배경이 아닌 부드러운 그라데이션 블롭
- 자연스러운 색상 전환으로 입체감 표현

**Minimal Icon Dock**
- 왼쪽 사이드바는 아이콘만 있는 세로 독 형태
- 플로팅 스타일로 화면에서 분리된 느낌

---

## 2. Color System

### Background with Gradient Blobs

```css
/* Base background */
background-color: #f8f9fa;

/* Gradient blob overlay - subtle and soft */
.gradient-blob-1 {
  background: radial-gradient(ellipse at 20% 30%, rgba(232, 240, 234, 0.6) 0%, transparent 50%);
}

.gradient-blob-2 {
  background: radial-gradient(ellipse at 80% 70%, rgba(254, 247, 224, 0.4) 0%, transparent 50%);
}

.gradient-blob-3 {
  background: radial-gradient(ellipse at 60% 20%, rgba(232, 245, 236, 0.5) 0%, transparent 40%);
}
```

### Primary Accent (Sage Green)

| Token | Value | Usage |
|-------|-------|-------|
| `primary-dark` | `#3d5a47` | Buttons, active states |
| `primary` | `#4a6b52` | Hover states |
| `primary-light` | `#e8f0ea` | Icon backgrounds |
| `primary-muted` | `#c8e6cf` | Borders, subtle accents |

### Neutral Colors

| Token | Value | Usage |
|-------|-------|-------|
| `text-primary` | `#1a1a1a` | Headings, important text |
| `text-secondary` | `#6b7280` | Body text, descriptions |
| `text-muted` | `#9ca3af` | Placeholder, disabled |
| `border` | `rgba(0, 0, 0, 0.06)` | Card borders |
| `surface` | `#ffffff` | Cards, containers |

### Semantic Colors (Toned Down)

| State | Background | Text | Icon | Border |
|-------|------------|------|------|--------|
| Success | `#e8f5ec` | `#3d7a4a` | `#4a9a5b` | `#c8e6cf` |
| Warning | `#fef7e0` | `#9a7b2d` | `#d4a84d` | `#f5e6b8` |
| Danger | `#fdedec` | `#b54a45` | `#c94b45` | `#f5c6c4` |

---

## 3. Typography

### Font Family

```css
/* Korean text */
font-family: 'Pretendard Variable', sans-serif;
letter-spacing: -0.025em;

/* English, numbers - Use Inter for cleaner look */
.font-display {
  font-family: 'Inter', 'Pretendard Variable', sans-serif;
  font-feature-settings: 'tnum' on, 'lnum' on;
}
```

### Font Scale

| Level | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| Display | 28-32px | 700 | 1.2 | Hero numbers, titles |
| H1 | 22-24px | 600 | 1.3 | Page titles |
| H2 | 18-20px | 600 | 1.4 | Section headers |
| H3 | 15-16px | 600 | 1.4 | Card titles |
| Body | 14px | 400 | 1.5 | Default text |
| Small | 13px | 400 | 1.5 | Secondary info |
| Caption | 11-12px | 500 | 1.4 | Labels, uppercase |
| Tiny | 10-11px | 500 | 1.4 | Timestamps |

### Number Display Style

```css
/* Large stat numbers */
.stat-number {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 2rem;
  letter-spacing: -0.02em;
  font-feature-settings: 'tnum' on;
}

/* Currency/percentage */
.stat-unit {
  font-size: 0.875rem;
  font-weight: 500;
  color: #6b7280;
}
```

---

## 4. Layout - Icon Dock Sidebar

### Dock Container

```css
.icon-dock {
  position: fixed;
  left: 24px;
  top: 50%;
  transform: translateY(-50%);

  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;

  padding: 12px 8px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.06),
    0 1px 2px rgba(0, 0, 0, 0.04);
}
```

### Dock Icon Button

```css
.dock-icon {
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  color: #9ca3af;
  transition: all 0.2s ease;
}

.dock-icon:hover {
  color: #6b7280;
  background: rgba(0, 0, 0, 0.04);
}

.dock-icon.active {
  color: #1a1a1a;
  background: rgba(0, 0, 0, 0.06);
}
```

---

## 5. Icons - Minimal Thin Line Style

### Icon Properties

```css
/* Base icon style */
.icon {
  stroke-width: 1.5;  /* Thin line */
  stroke-linecap: round;
  stroke-linejoin: round;
  fill: none;
}

/* Slightly thicker for emphasis */
.icon-medium {
  stroke-width: 1.75;
}

/* Very minimal */
.icon-light {
  stroke-width: 1.25;
}
```

### Icon Sizes

| Context | Size | Stroke |
|---------|------|--------|
| Dock nav | 22-24px | 1.5px |
| Card icon | 18-20px | 1.5px |
| Inline text | 14-16px | 1.5px |
| Button | 18-20px | 1.75px |
| Feature | 28-32px | 1.5px |

### Recommended Icon Set

- Lucide Icons (thin, consistent stroke)
- Heroicons Outline (24px, 1.5px stroke)
- Custom SVG with strokeWidth={1.5}

---

## 6. Card Components

### Standard Card

```css
.card {
  background: white;
  border-radius: 20px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.02),
    0 4px 12px rgba(0, 0, 0, 0.03);
  padding: 24px;
}
```

### Stat Card

```css
.stat-card {
  background: white;
  border-radius: 20px;
  padding: 20px 24px;
  border: 1px solid rgba(0, 0, 0, 0.04);
}

.stat-card .label {
  font-size: 12px;
  font-weight: 500;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.stat-card .value {
  font-family: 'Inter', sans-serif;
  font-size: 28px;
  font-weight: 700;
  color: #1a1a1a;
  letter-spacing: -0.02em;
}
```

### Feature Card (with accent)

```css
.feature-card {
  background: linear-gradient(135deg, #e8f0ea 0%, #f8f9fa 100%);
  border-radius: 20px;
  padding: 24px;
  border: 1px solid rgba(61, 90, 71, 0.1);
}

.feature-card .badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: white;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  color: #1a1a1a;
}
```

---

## 7. Main Content Layout

### Page Grid

```css
.page-layout {
  display: grid;
  grid-template-columns: 80px 1fr;  /* dock + content */
  min-height: 100vh;
}

.main-content {
  padding: 32px 40px;
  max-width: 1400px;
  margin: 0 auto;
}
```

### Dashboard Grid

```css
/* Hero section - 3 columns */
.dashboard-hero {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 24px;
}

/* Stats row */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

/* Content sections */
.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
}
```

---

## 8. Gradient Background Implementation

```css
.page-background {
  position: relative;
  background: #f8f9fa;
  min-height: 100vh;
}

.page-background::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 10% 20%, rgba(232, 240, 234, 0.7) 0%, transparent 50%),
    radial-gradient(ellipse 60% 50% at 90% 80%, rgba(254, 247, 224, 0.5) 0%, transparent 50%),
    radial-gradient(ellipse 50% 40% at 50% 10%, rgba(232, 245, 236, 0.4) 0%, transparent 40%);
  pointer-events: none;
}
```

---

## 9. Interactive Elements

### Button Primary

```css
.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: #e8f0ea;
  color: #3d5a47;
  border-radius: 12px;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-primary:hover {
  background: #d8e8dc;
}
```

### Dropdown

```css
.dropdown {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 10px;
  font-size: 13px;
  font-weight: 500;
  color: #6b7280;
}
```

---

## 10. Spacing System

| Token | Value | Usage |
|-------|-------|-------|
| `xs` | 4px | Icon gaps |
| `sm` | 8px | Tight spacing |
| `md` | 12px | Default gaps |
| `lg` | 16px | Component padding |
| `xl` | 20px | Card padding |
| `2xl` | 24px | Section padding |
| `3xl` | 32px | Large gaps |
| `4xl` | 40px | Page margins |

---

## 11. Shadows

```css
/* Subtle - Default cards */
--shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.03);

/* Medium - Hover, elevated */
--shadow-md: 0 4px 12px rgba(0, 0, 0, 0.05), 0 8px 24px rgba(0, 0, 0, 0.04);

/* Dock shadow */
--shadow-dock: 0 4px 24px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
```

---

## 12. Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `sm` | 8px | Small badges |
| `md` | 12px | Buttons, inputs |
| `lg` | 16px | Small cards |
| `xl` | 20px | Standard cards |
| `2xl` | 24px | Dock, large elements |
| `full` | 9999px | Pills, avatars |

---

## 13. Component Examples

### Dock Navigation

```tsx
<nav className="fixed left-6 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1 p-3 bg-white/90 backdrop-blur-xl rounded-3xl border border-black/[0.04] shadow-dock">
  <DockIcon icon={Grid} active />
  <DockIcon icon={Home} />
  <DockIcon icon={Calendar} />
  <DockIcon icon={FileText} />
  <DockIcon icon={Clock} />
  <DockIcon icon={Settings} />
  <DockIcon icon={Mail} />
</nav>
```

### Stat Card

```tsx
<div className="bg-white rounded-[20px] p-6 border border-black/[0.04] shadow-sm">
  <div className="flex items-center gap-2 text-gray-400 mb-3">
    <DollarSign className="w-5 h-5" strokeWidth={1.5} />
    <span className="text-xs font-medium uppercase tracking-wider">Total Revenue</span>
  </div>
  <p className="font-inter text-3xl font-bold text-gray-900 tracking-tight">
    $ 8,492,000
  </p>
  <span className="text-sm text-red-500 font-medium">-20%</span>
</div>
```

---

*Design System V2 - Modern Dashboard Style*
