# SAGE Experiment Report: Task A

## 1. 결과 테이블

| condition | intent | relation | emotion | attitude | avg |
|-----------|--------|----------|---------|----------|-----|
| Plain | 34.0% | 70.0% | 54.0% | 44.0% | 50.5% |
| SAGE_original | 44.0% | 64.0% | 50.0% | 42.0% | 50.0% |
| SAGE_enhanced | 46.0% | 70.0% | 56.0% | 38.0% | 52.5% |

## 2. 보강 내용 요약

### GRAPH_EXTRACTION_PROMPT 변경사항
- **역할 강화**: expert identity + 명시적 규칙 섹션 ([RULES], [OUTPUT FORMAT], [EXAMPLE])
- **appearance 필드 추가**: age_group (child|teen|adult|elder), gender (male|female|unknown)
- **symbol 시스템**: `<A>`, `<B>` 형식으로 캐릭터 참조 일관화 (SocialGPT의 `<P1>/<P2>` 방식 차용)
- **event.type 카테고리 제한**: argument|greeting|cooperation|confrontation|comfort|negotiation|other
- **few-shot example**: 1개 JSON 예시 포함으로 파싱 안정성 향상

### build_qa_prompt 변경사항
- **[EXPECTATION] 섹션**: 모델 역할과 태스크 목표 명시
- **[CONTEXT] 섹션**: 태스크별 정의 및 추론 기준 주입 (SocialGPT의 관계 정의 방식 차용)
- **[GUIDANCE] 섹션**: 태스크별 reasoning example 1개 in-context 추가
- **graph_to_text 재정렬**: task_type별 관련 정보 우선 배치
  - SSR_intent: event → role → state_change → interaction
  - SSR_relation: interaction → role → event → state_change
  - SSR_emotion: state_change → role → interaction → event
  - SSR_attitude: interaction → state_change → event → role

## 3. 오류 분석

- **SAGE_original**: 예측 실패(None) 17/200건
- **SAGE_enhanced**: 예측 실패(None) 17/200건

## 4. Next Step (Task B 준비)

- SocialGPT `social-story-main-PIPA/main.py`를 SIV-Bench에 맞게 포팅
  - BLIP-2 / SAM 없이 GPT-4o-mini가 직접 story 생성
  - 동일 200개 seed-42 subset에서 SAGE_enhanced vs SocialGPT_story 비교
- 전체 4,951개 SSR 샘플 풀 실험 검토