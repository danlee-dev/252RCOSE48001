# Frontend Troubleshooting

프론트엔드 개발 중 발생한 이슈와 해결 방법을 정리합니다.

---

## [CSS] globals.css 전역 스타일이 UI 컴포넌트에 의도치 않게 적용되는 문제

**발생 날짜**: 2025-01-03
**작성자**: 이성민

### 문제 상황

사이드바의 브랜드 영역에서 `DocScanner.ai`와 `AI 계약서 분석` 두 텍스트 사이에 24px의 예상치 못한 여백이 발생했습니다.

```tsx
<div className="flex flex-col gap-1">
  <h1>DocScanner.ai</h1>
  <p>AI 계약서 분석</p>  {/* 여기에 24px margin이 자동 추가됨 */}
</div>
```

**증상**:
- 개발자 도구에서 확인 시 `p` 태그에 `margin-top: 24px` 자동 적용
- `gap-1` (4px)을 설정했지만 실제로는 28px 간격 발생
- `m-0`, `!mt-0` 등으로 오버라이드 시도했지만 여전히 적용됨

### 원인

`src/app/globals.css`에 정의된 전역 스타일이 원인:

```css
@layer base {
  p {
    @apply leading-7 [&:not(:first-child)]:mt-6;
  }
}
```

`[&:not(:first-child)]:mt-6`는 "첫 번째 자식이 아닌 모든 `p` 태그에 `mt-6` (24px) margin을 추가"를 의미합니다.

**적용 메커니즘**:
```html
<div>
  <h1>...</h1>  <!-- 첫 번째 자식 -->
  <p>...</p>    <!-- 두 번째 자식 → :not(:first-child)에 해당 → mt-6 자동 적용 -->
</div>
```

이 전역 스타일은 원래 블로그 글이나 마크다운 콘텐츠에서 단락 간 자동 간격을 주기 위한 것이지만, **모든 `p` 태그에 적용되어 UI 컴포넌트에도 영향**을 미칩니다.

### 해결 방법

#### 방법 1: 시맨틱 태그를 중립적인 태그로 변경 (채택)

```tsx
// Before
<div className="flex flex-col gap-1">
  <h1 className="...">DocScanner.ai</h1>
  <p className="...">AI 계약서 분석</p>
</div>

// After
<div className="flex flex-col gap-1">
  <span className="...">DocScanner.ai</span>
  <span className="...">AI 계약서 분석</span>
</div>
```

**장점**:
- 전역 스타일 영향 없음
- 추가 CSS 불필요
- UI 컴포넌트에서는 의미론보다 스타일 제어가 중요

**단점**:
- 의미론적으로는 덜 명확 (하지만 UI 컴포넌트에서는 무관)

#### 방법 2: `!important`로 오버라이드

```tsx
<p className="text-xs ... !mt-0">AI 계약서 분석</p>
```

**단점**:
- specificity 전쟁 시작
- 유지보수 어려움

#### 방법 3: 전역 스타일 범위 제한 (장기 해결책)

```css
@layer base {
  /* Before: 모든 p 태그에 적용 */
  p {
    @apply leading-7 [&:not(:first-child)]:mt-6;
  }

  /* After: .prose 클래스 안에서만 적용 */
  .prose p {
    @apply leading-7 [&:not(:first-child)]:mt-6;
  }
}
```

그리고 콘텐츠 영역에만 `.prose` 적용:
```tsx
<div className="prose">
  {/* 블로그 글, 마크다운 콘텐츠 */}
</div>
```

### 언제 이런 문제가 발생하는가?

1. **Tailwind Typography plugin 사용 시**
   - 블로그, 문서 사이트에서 콘텐츠 스타일링 자동화

2. **전역 스타일에서 시맨틱 태그에 기본 스타일 적용**
   - `h1`, `h2`, `p`, `ul` 등에 자동 margin/padding
   - 문서 콘텐츠에는 유용하지만 UI 컴포넌트에서는 문제

3. **`:not(:first-child)` 같은 선택자**
   - 형제 요소 간 자동 간격
   - 의도치 않은 곳에도 적용

### 교훈 및 베스트 프랙티스

#### 1. UI 컴포넌트에서는 중립적인 태그 사용

```tsx
// Good: UI 컴포넌트
<div className="sidebar">
  <span className="title">제목</span>
  <span className="subtitle">부제목</span>
</div>

// Bad: 전역 스타일 영향 받을 수 있음
<div className="sidebar">
  <h1>제목</h1>
  <p>부제목</p>
</div>
```

#### 2. 전역 스타일은 범위 제한

```css
/* Bad: 전역 적용 */
p { ... }

/* Good: 특정 클래스 내에서만 */
.prose p { ... }
.article p { ... }
```

#### 3. 컴포넌트별 스타일 격리

```tsx
// Tailwind 유틸리티로 명시적 스타일 정의
<div className="flex flex-col gap-1">
  <span className="text-xl font-bold">제목</span>
  <span className="text-xs text-gray-600">부제목</span>
</div>
```

#### 4. 디버깅 팁

예상치 못한 여백 발견 시:
1. 개발자 도구에서 Computed 탭 확인
2. Styles 패널에서 어느 CSS 파일에서 왔는지 확인
3. `@layer base` 의 전역 스타일 의심
4. `:not()`, `:first-child`, `:last-child` 같은 선택자 체크

### 관련 파일

- `frontend/src/app/globals.css` - 전역 스타일 정의
- `frontend/src/components/ui/sidebar.tsx` - 수정된 컴포넌트

### 참고 링크

- [Tailwind CSS @layer directive](https://tailwindcss.com/docs/functions-and-directives#layer)
- [CSS :not() pseudo-class](https://developer.mozilla.org/en-US/docs/Web/CSS/:not)
