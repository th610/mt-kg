# SAGE Agent Task A: 프롬프트 보강 + 실험 실행

---

## 배경

이 태스크는 두 단계로 구성된다:

- **Task A (지금):** SocialGPT 프롬프트 설계 패턴을 분석해서 SAGE 프롬프트를 보강하고, 보강 전/후 실험을 비교한다
- **Task B (다음):** 보강된 SAGE vs SocialGPT baseline을 동일 데이터셋에서 직접 비교한다

---

## 보유 소스

1. `SocialGPT` 코드 (`social-story-main-PIPA/main.py` 포함)
2. `SAGE_implementation_spec.md`

---

## 데이터셋

**SIV-Bench** — HuggingFace에서 필요한 것만 가져오는 형태.

```python
from datasets import load_dataset
ds = load_dataset("your-org/SIV-Bench", split="test")  # 실제 repo명 확인 후 수정
```

**Target tasks (SSR 전체):**

| 태스크 | category 필드값 | 샘플 수 |
|--------|----------------|---------|
| Intent Inference | `"Intent Inference"` | 2,132 |
| Relation Inference | `"Relation Inference"` | 1,239 |
| Emotion Inference | `"Emotion Inference"` | 1,083 |
| Attitude Inference | `"Attitude Inference"` | 497 |

**빠른 테스트용 subset:** 각 태스크별 50개씩, random seed 42, 총 200개.

---

## Step 1: SocialGPT 프롬프트 분석

`social-story-main-PIPA/main.py`에서 다음을 추출해서 정리하라:

- System 프롬프트 전문
- Story generation 프롬프트 전문
- SocialPrompt (Expectation / Context / Guidance 섹션) 전문
- 어떤 visual information을 어떤 순서로 요청하는지

---

## Step 2: SAGE 프롬프트 보강

분석 결과를 바탕으로 `GRAPH_EXTRACTION_PROMPT`와 `build_qa_prompt` 두 개를 보강하라.

### 2-1. GRAPH_EXTRACTION_PROMPT 보강 포인트

- **역할 설정 강화:** SocialGPT의 System 섹션처럼 모델 역할, 규칙, 출력 형식을 명확히 선언
- **character 정보 확장:** SocialGPT가 age/gender/social event를 task-oriented caption으로 뽑는 것처럼, `appearance` 필드(나이대, 성별 추정) 추가
- **symbol 시스템 통합:** SocialGPT의 `<P1>`, `<P2>` 참조 방식을 SAGE의 A/B/C ID 시스템과 통합해 캐릭터 참조를 일관되게 유지
- **event.type 카테고리 힌트:** 현재 open-ended라 노이즈가 많으므로, 명시적 카테고리 후보 제시 (예: `argument / greeting / cooperation / confrontation / comfort / negotiation / ...`)
- **JSON 출력 안정성:** 짧은 few-shot example 1개 추가

**보강 후 JSON 스키마 (목표):**

```json
{
  "characters": [
    {
      "id": "A",
      "role": "<aggressor|target|mediator|observer|supporter|initiator|responder>",
      "emotion": "<emotion>",
      "appearance": {"age_group": "<child|teen|adult|elder>", "gender": "<male|female|unknown>"}
    }
  ],
  "interactions": [
    {
      "from": "<id>", "to": "<id>",
      "type": "<conflict|support|ignore|mediate|confront>",
      "intensity": "<0.0~1.0>"
    }
  ],
  "event": {
    "type": "<argument|greeting|cooperation|confrontation|comfort|negotiation|other>",
    "participants": ["<id>"],
    "stage": "<initiating|escalating|resolving|stable>"
  },
  "state_change": {
    "early": "<frames 1-8 묘사>",
    "late": "<frames 9-16 묘사>"
  }
}
```

---

### 2-2. build_qa_prompt 보강 포인트

- **Expectation 섹션 추가:** SocialGPT처럼 "당신이 해야 할 일"을 명확히 선언하는 섹션 추가
- **Context 섹션 추가:** 각 task_type별 정의 및 판단 기준 주입

  | task_type | Context 내용 |
  |-----------|-------------|
  | `SSR_intent` | intent란 무엇인지, role/event에서 motivation을 추론하는 방법 |
  | `SSR_relation` | relation category 종류(family/friend/colleague/romantic/stranger 등)와 판단 기준 |
  | `SSR_emotion` | emotion의 정의, visual/behavioral indicator 해석법 |
  | `SSR_attitude` | attitude의 정의, interaction type과 intensity에서 태도 추론하는 방법 |

- **Guidance 섹션 추가:** task별 reasoning example 1개씩 in-context로 추가
- **graph_to_text 순서 재정렬:** task_type에 따라 관련 정보를 앞으로 배치

  | task_type | 우선순서 |
  |-----------|---------|
  | `SSR_intent` | event → role → state_change |
  | `SSR_relation` | interactions → role → appearance |
  | `SSR_emotion` | state_change → character emotion → interactions |
  | `SSR_attitude` | interactions intensity → state_change → event stage |

---

## Step 3: 실험 실행

보강 전/후 포함 총 3개 조건으로 SIV-Bench subset 200개를 실행하라.

**실험 조건:**

| 조건 | 설명 |
|------|------|
| `Plain` | 프레임만 입력, graph 없이 바로 QA |
| `SAGE_original` | 보강 전 `SAGE_implementation_spec.md` 그대로 |
| `SAGE_enhanced` | Step 2에서 보강된 버전 |

**실험 조건 × 태스크 매트릭스:**

```
3 conditions × 4 tasks × 50 samples = 600 API calls (Step 1 포함 시 ~1,200)
```

**결과 출력 포맷:**

```
| condition       | intent | relation | emotion | attitude | avg  |
|----------------|--------|----------|---------|----------|------|
| Plain          |  0.xxx |  0.xxx   |  0.xxx  |  0.xxx   | 0.xxx|
| SAGE_original  |  0.xxx |  0.xxx   |  0.xxx  |  0.xxx   | 0.xxx|
| SAGE_enhanced  |  0.xxx |  0.xxx   |  0.xxx  |  0.xxx   | 0.xxx|
```

---

## Step 4: 출력물

다음 3개 파일을 저장하라:

### 4-1. `sage_enhanced.py`
보강된 프롬프트 포함 전체 파이프라인 코드. 아래 구조를 따를 것:

```
sage_enhanced.py
├── sample_frames()
├── GRAPH_EXTRACTION_PROMPT  (보강됨)
├── extract_social_graph()
├── graph_to_text(graph, task_type)  ← task_type 인자 추가
├── build_qa_prompt()  (보강됨)
├── answer_question()
├── run_sage_pipeline()
├── run_experiment(condition, data_dir, n_per_task=50)
└── main()  ← argparse로 --data_dir, --condition, --output_dir 받음
```

### 4-2. `results_comparison.json`
세 조건의 raw 결과. 아래 구조:

```json
{
  "Plain": {
    "SSR_intent": [
      {"sample_id": "...", "prediction": "A", "ground_truth": "B", "correct": false},
      ...
    ],
    "SSR_relation": [...],
    "SSR_emotion": [...],
    "SSR_attitude": [...]
  },
  "SAGE_original": { ... },
  "SAGE_enhanced": { ... }
}
```

### 4-3. `experiment_report.md`
다음 섹션을 포함한 실험 보고서:

1. **결과 테이블** (위 포맷)
2. **보강 내용 요약** — 어떤 변경이 어떤 태스크에 얼마나 영향을 줬는지
3. **오류 분석** — JSON parse 실패율, fallback 발생 케이스 수
4. **Next Step** — Task B (SocialGPT baseline 비교) 준비에 필요한 사항

---

## 실행 주의사항

- API key: 환경변수 `OPENAI_API_KEY`에서 읽어라
- Rate limit 대비 요청 사이 `time.sleep(0.5)` 추가
- JSON parse 실패 시 → `graph_text = "[Social Context: extraction failed]"` fallback, Plain처럼 동작
- Subset 샘플링: `random.seed(42)` 고정
- SIV-Bench HuggingFace 로드:
  ```python
  from datasets import load_dataset
  # SSR 태스크만 필터링해서 로드
  # category 필드: "Intent Inference" | "Relation Inference" | "Emotion Inference" | "Attitude Inference"
  # 실제 HuggingFace repo명과 필드명은 데이터셋 확인 후 맞춰라
  ```
- 비디오 파일은 HuggingFace에서 URL 또는 로컬 캐시로 접근 가능한 형태로 처리
- 모든 출력 파일은 `--output_dir` 경로에 저장

---

## 다음 태스크 예고 (Task B)

Task A 완료 후 Task B에서는:
- `sage_enhanced.py`의 pipeline을 기반으로
- SocialGPT의 `social-story-main-PIPA/main.py`를 SIV-Bench에 맞게 포팅
- SAGE_enhanced vs SocialGPT_story 두 조건을 동일 200개 subset에서 비교
- 이때 SocialGPT는 GPT-4o mini 기반으로 통일 (SAM/BLIP-2 없이, GPT-4o mini가 직접 story 생성)
