# 협업 가이드

## 팀 구성

| 이름 | 역할 | 담당 영역 |
|------|------|-----------|
| 이성민 | 프론트엔드 + AI | Next.js, AI 데이터 파이프라인 |
| 강민선 | 백엔드 + AI | NestJS, FastAPI, AI 모델 서빙 |

## Git 브랜치 전략

### 브랜치 구조

```
main (배포용)
  ↑
develop (개발 통합)
  ↑
feature/frontend-* (이성민)
feature/backend-* (강민선)
feature/ai-* (공동)
```

### 브랜치 설명

**main**
- 배포 가능한 안정 버전
- 직접 커밋 금지
- develop에서 PR을 통해서만 병합
- 태그를 통한 버전 관리

**develop**
- 개발 통합 브랜치
- 모든 feature 브랜치는 develop에서 분기
- 기능 완성 후 develop으로 PR
- 주기적으로 main에 병합

**feature 브랜치**
- develop에서 분기
- 기능 개발 완료 후 develop으로 PR
- 병합 후 삭제

### 브랜치 네이밍 규칙

```
feature/영역-기능명
예시:
  feature/frontend-dashboard
  feature/frontend-contract-upload
  feature/backend-auth
  feature/backend-contract-api
  feature/ai-rag-engine
  feature/ai-elasticsearch

bugfix/영역-버그명
예시:
  bugfix/frontend-upload-error
  bugfix/backend-db-connection

hotfix/긴급수정명
예시:
  hotfix/security-patch
```

## 작업 흐름

### 1. 새 기능 시작

```bash
# develop 최신 상태로 업데이트
git checkout develop
git pull origin develop

# 새 기능 브랜치 생성
git checkout -b feature/frontend-dashboard

# 작업 진행...
```

### 2. 작업 중

```bash
# 자주 커밋 (의미 있는 단위로)
git add .
git commit -m "feat: add dashboard layout component"

# 원격 저장소에 push
git push origin feature/frontend-dashboard
```

### 3. PR 생성

```bash
# GitHub에서 PR 생성
# feature/frontend-dashboard → develop

# PR 템플릿:
# - 작업 내용 설명
# - 스크린샷 (프론트엔드의 경우)
# - 테스트 결과
# - 주의사항
```

### 4. 코드 리뷰 및 병합

- 상대방이 코드 리뷰
- 피드백 반영 후 승인
- develop에 병합
- 브랜치 삭제

### 5. 병합 후 정리

```bash
# develop 최신화
git checkout develop
git pull origin develop

# 작업 완료된 브랜치 삭제
git branch -d feature/frontend-dashboard
```

## 커밋 메시지 규칙

### 형식

```
type: subject

body (선택)
```

### Type 종류

- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅 (기능 변경 없음)
- `refactor`: 리팩토링
- `test`: 테스트 코드
- `chore`: 빌드, 설정 변경

### 예시

```bash
feat: add contract upload component
fix: resolve database connection timeout
docs: update API documentation
refactor: optimize embedding generation
```

### 주의사항

- 이모지 사용 금지
- Claude 작성 표기 금지
- 한 줄로 간결하게
- 영어로 작성

## AI 작업 시 충돌 방지

AI 폴더는 두 사람 모두 작업하므로 충돌 주의가 필요합니다.

### 방법 1: 세부 모듈로 분리

```
이성민:
  - ai/preprocessing/ (데이터 전처리)
  - ai/data/ (데이터 관리)

강민선:
  - ai/api/ (FastAPI 엔드포인트)
  - ai/models/ (모델 서빙)
  - ai/rag/ (RAG 엔진)
```

### 방법 2: 작업 전 소통

- Slack에 작업 시작 알림
- 같은 파일 동시 수정 피하기
- 작업 전 develop pull 필수

### 방법 3: 자주 동기화

```bash
# 작업 중에도 주기적으로 develop 변경사항 반영
git checkout develop
git pull origin develop
git checkout feature/your-branch
git merge develop
```

## 코드 리뷰 가이드

### 리뷰어 체크리스트

- 코드가 요구사항을 충족하는가?
- 코딩 컨벤션을 따르는가?
- 불필요한 코드가 없는가?
- 테스트가 필요한가?
- 보안 이슈는 없는가?
- 성능 문제는 없는가?

### 리뷰 예시

```
✅ 승인
- 로직이 명확하고 잘 구현되었습니다
- 컴포넌트 분리가 적절합니다

💬 제안
- useState 대신 useReducer 사용 고려해보세요
- 에러 핸들링 추가가 필요할 것 같습니다

🔧 수정 요청
- API 엔드포인트 URL 하드코딩 제거 필요
- 환경 변수로 관리해주세요
```

## 충돌 해결

### 충돌 발생 시

```bash
# develop 최신화
git checkout develop
git pull origin develop

# 작업 브랜치로 이동
git checkout feature/your-branch

# develop 병합
git merge develop

# 충돌 파일 확인
git status

# 충돌 해결 후
git add .
git commit -m "chore: resolve merge conflicts"
git push origin feature/your-branch
```

### 충돌 방지 팁

- 자주 develop pull
- 큰 기능은 작은 단위로 나눠서 작업
- 파일 수정 전 상대방과 소통

## 개발 환경 설정

### 프론트엔드 (이성민)

```bash
cd frontend
npm install
npm run dev
```

### 백엔드 (강민선)

```bash
# NestJS
cd backend
npm install
npm run start:dev

# FastAPI
cd ai
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### AI 개발 (공통)

**PDF 파이프라인:**
```bash
cd ai/preprocessing/pdf
conda activate docscanner-py3.10
python 3_embed.py
```

**법률 데이터 파이프라인:**
```bash
cd ai/preprocessing/legal
conda activate docscanner-py3.10
python 3_embed.py
```

**검색 테스트:**
```bash
cd ai/preprocessing/test
conda activate docscanner-py3.10
python search.py i
```

## 커뮤니케이션

### 일일 스탠드업

매일 작업 시작 전:
- 어제 한 일
- 오늘 할 일
- 막힌 부분

### 주간 회의

매주 정기 회의:
- 진행 상황 공유
- 다음 주 계획
- 이슈 논의

### Slack 채널

- general: 일반 소통
- dev: 개발 관련 논의
- github: PR, 이슈 알림

## 이슈 관리

### GitHub Issues 활용

```
제목: [Frontend] 계약서 업로드 오류
라벨: bug, frontend
담당자: @danlee-dev

내용:
- 현상: 파일 업로드 시 500 에러 발생
- 재현 방법: ...
- 예상 원인: ...
```

### 라벨 체계

- `frontend`: 프론트엔드 관련
- `backend`: 백엔드 관련
- `ai`: AI/ML 관련
- `bug`: 버그
- `feature`: 새 기능
- `docs`: 문서
- `urgent`: 긴급

## 배포 프로세스 (예정)

### 배포 흐름

```
develop → main (PR) → 배포
```

### 배포 전 체크리스트

- 모든 테스트 통과
- 코드 리뷰 완료
- 문서 업데이트
- 버전 태깅

## 참고 자료

- [Git 브랜치 전략](https://nvie.com/posts/a-successful-git-branching-model/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Code Review Best Practices](https://google.github.io/eng-practices/review/)
