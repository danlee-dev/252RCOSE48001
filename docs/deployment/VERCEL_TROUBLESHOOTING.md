# Vercel Deployment Troubleshooting Guide

이 문서는 Vercel 프론트엔드 배포 과정에서 발생한 에러와 해결 방법을 정리합니다.

---

## Vercel URL 구조 및 Preview 배포

### URL 종류

Vercel은 배포 시 3가지 유형의 URL을 생성한다:

| URL 패턴 | 유형 | 설명 | 안정성 |
|----------|------|------|--------|
| `docscannerai-mason-lees-...` | Production | Production 브랜치의 최신 코드 | 고정 (브랜치 기반) |
| `docscannerai-git-feature-xxx-...` | Preview (브랜치) | 특정 브랜치의 최신 코드 | 고정 (브랜치 기반) |
| `docscannerai-ozs2p7jja-...` | Preview (배포) | 특정 배포의 스냅샷 | 고정 (커밋 기반) |

### Production vs Preview

```
Production 환경
- 설정된 Production 브랜치 (예: develop)의 코드만 배포
- 단일 고정 URL 제공
- 실제 서비스용

Preview 환경
- Production 브랜치를 제외한 모든 브랜치에 자동 배포
- 브랜치당 고정 URL + 배포당 고유 URL 생성
- 개발/테스트/PR 리뷰용
```

### 배포 흐름 예시

```
1. feature/login-fix 브랜치에서 작업 후 push

   생성되는 URL:
   - docscannerai-git-feature-login-fix-xxx.vercel.app  (브랜치 URL, 업데이트됨)
   - docscannerai-abc123xyz-xxx.vercel.app              (배포 URL, 이 커밋 전용)

2. 같은 브랜치에 추가 커밋 후 push

   URL 변화:
   - 브랜치 URL: 새 코드로 업데이트
   - 배포 URL: 새로운 고유 URL 생성 (이전 URL은 그대로 유지)

3. develop에 PR 머지

   URL 변화:
   - Production URL: develop 최신 코드로 업데이트
   - Preview URL들: 더 이상 업데이트 안 됨 (삭제 전까지 유지)
```

### 언제 어떤 URL을 사용해야 하나?

| 상황 | 사용할 URL |
|------|-----------|
| 실제 서비스 접속 | Production URL |
| PR 리뷰 시 동작 확인 | 브랜치 기반 Preview URL |
| "어제 배포한 버전 확인" | 배포 고유 URL |
| Railway CORS 등록 | Production URL만 |

### Preview 배포 비활성화 (선택)

불필요한 빌드를 줄이려면 Vercel Settings에서 특정 브랜치만 배포하도록 설정 가능:
1. Project Settings > Git > Ignored Build Step
2. 또는 vercel.json에서 설정

---

## 1. ESLint prefer-const 에러

### 증상
```
Error: 'targetVersion' is never reassigned. Use 'const' instead.  prefer-const
```
빌드 시 ESLint 검사에서 실패.

### 원인
`let`으로 선언했지만 실제로 재할당되지 않는 변수 존재.

### 해결
해당 변수를 `const`로 변경:
```typescript
// Before
let targetVersion = condition ? valueA : valueB;

// After
const targetVersion = condition ? valueA : valueB;
```

### 예방
로컬에서 빌드 전 lint 검사 실행:
```bash
cd frontend
npm run lint
```

---

## 2. 빌드 캐시 문제

### 증상
코드 수정 후 PR 머지했는데 동일한 에러가 계속 발생.

### 원인
Vercel이 이전 빌드 캐시를 사용하거나, webhook이 새 커밋을 감지하지 못함.

### 해결
1. Vercel Dashboard > Deployments 탭
2. 최근 배포 옆 **...** 메뉴 클릭
3. **Redeploy** > **Redeploy without cache** 선택

또는 빌드 로그에서 커밋 SHA 확인하여 올바른 커밋이 빌드되고 있는지 검증.

---

## Vercel 환경변수 설정

### 필수 환경변수
```
NEXT_PUBLIC_API_URL=https://your-railway-backend.up.railway.app
```

### Production vs Preview

Vercel은 두 가지 배포 환경을 제공:

| 환경 | 브랜치 | 용도 |
|------|--------|------|
| Production | main | 실제 서비스 URL |
| Preview | develop, feature/* | 개발/테스트용 URL |

환경변수 설정 시 해당 환경에만 적용할지 선택 가능.

---

## Railway CORS 설정

Vercel 프론트엔드가 Railway 백엔드에 요청하려면 CORS 설정 필요:

### Railway Backend 환경변수
```
CORS_ORIGINS=https://your-production.vercel.app,https://your-develop.vercel.app
```

여러 도메인은 쉼표로 구분.

---

## 배포 체크리스트

### 배포 전
- [ ] `npm run lint` 로컬 실행하여 ESLint 에러 확인
- [ ] `npm run build` 로컬 빌드 테스트
- [ ] 환경변수 확인 (NEXT_PUBLIC_* 변수는 빌드 시점에 포함됨)

### 배포 후
- [ ] Vercel 빌드 로그에서 커밋 SHA 확인
- [ ] 브라우저 개발자 도구에서 API 요청 확인
- [ ] CORS 에러 발생 시 Railway CORS_ORIGINS 확인
