# Branch 규칙
- main 브랜치에서는 직접 작업하지 않는다.
- 작업별로 브랜치를 생성한다.
- 브랜치명은 `작업명` 형식을 사용한다.
- 작업 완료 후 Pull Request를 통해 main에 병합한다.

# Commit 메시지 작성 규칙
- feat: 기능 추가
- fix: 오류 수정
- docs: 문서 수정
- refactor: 코드 정리
    ## 예시
    - git commit -m "feat: 고장 유형 분류 모델 추가"
    - git commit -m "fix: 결측치 처리 오류 수정"
    - git commit -m "docs: README 프로젝트 설명 추가"

# 작업 규칙
1. 작업 전 Issue 생성
2. Issue 번호를 확인하고 Branch 생성
3. 개인 Branch에서 작업
4. 작업 완료 후 Pull Request
5. 리뷰어 1명 이상 확인
6. 승인 후 main에 merge