# 🎯 System Prompt 최적화 가이드

## 변경 사항 요약

기존 `prompts/system.txt`의 형식적이고 장황한 답변을 **자연스럽고 간결한 서비스형 챗봇 스타일**로 개선했습니다.

---

## 📊 Before vs After

### **Before (system.txt)** - 형식적, 장황함

**답변 예시:**
```
요약: Python에서 *args는 임의 개수의 위치 인자(순서대로 전달되는 값), **kwargs는 임의 개수의 키워드 인자(이름=값 형태)를 함수에 전달할 때 사용합니다 [DOC-3: Python Function Arguments, 요약].

세부 단계:
1) *args: 함수 정의에서 *args를 사용하면 여러 개의 위치 인자를 튜플로 받아 처리할 수 있습니다.
   예시:
   def foo(*args):
       for arg in args:
           print(arg)
2) **kwargs: 함수 정의에서 **kwargs를 사용하면 여러 개의 키워드 인자를 딕셔너리로 받아 처리할 수 있습니다.
   예시:
   def bar(**kwargs):
       for key, value in kwargs.items():
           print(key, value)
3) *args와 **kwargs를 함께 사용할 수 있으며, 이때는 반드시 순서대로(위치 인자, *args, 키워드 인자, **kwargs) 정의해야 합니다 [DOC-6: Python args and kwargs, 설명].
   예시:
   def baz(a, *args, **kwargs):
       pass

[증거 요약]
- [DOC-3] Python | Python Function Arguments | *args는 임의 개수의 위치 인자, **kwargs는 임의 개수의 키워드 인자 처리 [DOC-3]
- [DOC-6] ...
```

**문제점:**
- ❌ "요약:", "세부 단계:", "[증거 요약]" 등 과도한 구조화
- ❌ 모든 문장마다 [DOC-x: ...] 인용으로 가독성 저하
- ❌ 15줄 이상의 장황한 답변
- ❌ 형식적이고 부자연스러운 톤

---

### **After (system_v2.txt)** - 자연스럽고 간결함

**답변 예시:**
```
*args와 **kwargs는 함수에 가변 개수의 인자를 전달할 때 사용합니다.

• *args: 위치 인자를 튜플로 받음
• **kwargs: 키워드 인자(이름=값)를 딕셔너리로 받음

예시:
def greet(*args, **kwargs):
    print(args)    # ('Alice', 'Bob')
    print(kwargs)  # {'age': 25, 'city': 'Seoul'}

greet('Alice', 'Bob', age=25, city='Seoul')

함께 사용할 때는 순서가 중요해요: 일반 인자 → *args → 키워드 인자 → **kwargs

📚 참고: Python Function Arguments
```

**개선점:**
- ✅ 자연스러운 대화형 톤
- ✅ 간결한 답변 (8줄)
- ✅ 출처는 끝에 한 번만
- ✅ 핵심만 명확하게 전달

---

## 🔬 적용된 LLM 프롬프팅 이론

### 1. **Constitutional AI (Anthropic 2022)**
**목적:** 환각(hallucination) 방지

**적용:**
```
핵심 원칙:
1. 검색된 문서에만 기반하여 답변 (환각 방지)
2. 문서에 없는 정보는 절대 추측하지 마세요
3. 불확실하면 솔직하게 밝히기
```

**효과:**
- Faithfulness ↑ (문서 충실도)
- 근거 없는 답변 방지

---

### 2. **Few-shot Prompting (Brown et al., 2020)**
**목적:** 원하는 답변 스타일 학습

**적용:**
```
## 예시 (Few-shot Learning)

### 예시 1: 간단한 개념 설명
질문: "Python에서 *args와 **kwargs는 무엇인가요?"
답변: [자연스럽고 간결한 예시]

### 예시 2: 실용적 가이드
질문: "git rebase는 언제 쓰나요?"
답변: [상황별 설명 예시]

### 예시 3: 비교/차이 설명
질문: "얕은 복사와 깊은 복사의 차이는?"
답변: [비교 형식 예시]
```

**효과:**
- 답변 스타일 일관성 ↑
- 형식보다 예시가 더 강력함 (GPT는 패턴 학습)

---

### 3. **Context Grounding (Self-RAG, Asai et al., 2023)**
**목적:** 검색된 문서에만 의존

**적용:**
```
내부 작업 흐름:
Step 1: 문서 평가 (RELEVANT vs IRRELEVANT)
Step 2: 답변 구성 (관련 문서만 사용)
Step 3: 검증 (추측 제거)

⚠️ 위 흐름은 내부 작업이며, 사용자에게는 최종 답변만!
```

**효과:**
- Context Precision ↑
- 관련 없는 정보 필터링

---

### 4. **Chain-of-Thought (Wei et al., 2022)**
**목적:** 복잡한 질문 분해 (내부적으로)

**적용:**
```
복잡한 질문 → 내부적으로 2-4개 하위 문제로 분해
→ 각각 해결
→ 통합하여 간결한 최종 답변
```

**효과:**
- 복잡한 질문 답변 품질 ↑
- 하지만 사용자에게는 간결한 결과만

---

### 5. **Minimal Citation Strategy**
**목적:** 가독성 향상

**적용:**
```
기본: 답변 끝에 "📚 참고: [문서명]"
여러 문서: "📚 참고: [문서1], [문서2]"
중요 인용: 문장 중간 필요시만 "(git-rebase.md)"

⚠️ 모든 문장마다 인용하지 마세요!
```

**효과:**
- 가독성 대폭 향상
- 자연스러운 흐름 유지

---

## 📈 예상 성능 변화

| 측면 | Before | After | 개선 이유 |
|------|--------|-------|-----------|
| **답변 길이** | 15-20줄 | 5-8줄 | 불필요한 형식 제거 |
| **가독성** | 낮음 | 높음 | 인용 과다 제거 |
| **자연스러움** | 형식적 | 대화형 | 톤 최적화 |
| **Faithfulness** | 유지 | 유지/향상 | Constitutional AI |
| **사용자 만족도** | 낮음 | 높음 | 서비스형 스타일 |

---

## 🚀 테스트 방법

### **1. 재인덱싱 (이미 완료했다면 Skip)**
```bash
cd /workspace/rag_pipeline

# Chunking 파라미터 변경했으므로 재인덱싱 필요
python data_prep.py --config config/enhanced.yaml
python index_builder.py --config config/enhanced.yaml
```

### **2. 새 프롬프트로 테스트**

**간단한 개념 설명:**
```bash
python answerer_v2.py "Python에서 *args와 **kwargs는 무엇인가요?" --config config/enhanced.yaml
```

**비교/차이 질문:**
```bash
python answerer_v2.py "Python에서 얕은 복사와 깊은 복사의 차이는?" --config config/enhanced.yaml
```

**실용적 질문:**
```bash
python answerer_v2.py "git rebase는 언제 쓰나요?" --config config/enhanced.yaml
```

**복잡한 질문:**
```bash
python answerer_v2.py "Python 데코레이터는 어떻게 작동하고 언제 사용하나요?" --config config/enhanced.yaml
```

---

### **3. 기존 프롬프트와 비교 (선택)**

만약 기존 프롬프트와 직접 비교하고 싶다면:

```bash
# enhanced.yaml 임시 수정
# system_prompt_path: prompts/system.txt  (기존)

python answerer_v2.py "Python에서 *args와 **kwargs는 무엇인가요?" --config config/enhanced.yaml

# 다시 system_v2.txt로 변경
# system_prompt_path: prompts/system_v2.txt  (신규)

python answerer_v2.py "Python에서 *args와 **kwargs는 무엇인가요?" --config config/enhanced.yaml
```

---

## 🎯 체크리스트

### 답변 품질 검증
- [ ] **간결성**: 8줄 이내 (핵심만)
- [ ] **자연스러움**: 대화형 톤 (형식적 구조 없음)
- [ ] **정확성**: 문서 기반 (추측 없음)
- [ ] **가독성**: 인용 과다 없음 (끝에 한 번)
- [ ] **완전성**: 필요한 정보 모두 포함

### 환각 방지 검증
- [ ] 문서에 없는 정보 추가 안 함
- [ ] 불확실할 때 솔직하게 표현
- [ ] 출처 명확히 표시

### 사용자 경험
- [ ] 빠르게 읽을 수 있음
- [ ] 핵심이 명확함
- [ ] 추가 질문 여지 있음

---

## 📊 RAGAS 평가 (최종 검증)

새 프롬프트 적용 후 전체 RAGAS 평가 실행:

```bash
# Git/Python 질문만 평가
# ragas_questions.json에서 Git/Python 필터링하여 실행
```

**기대 목표:**

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| **Context Precision** | ~70% | **80%+** | Grounding + Filtering |
| **Context Recall** | ~70% | **85%+** | rerank_top_k=10 |
| **Faithfulness** | ~86% | **93%+** | Constitutional AI |
| **Answer Relevancy** | ~65% | **80%+** | Natural Style |
| **Answer Correctness** | ~60% | **75%+** | 종합 개선 |

---

## 💡 추가 최적화 팁

### 1. **온도(Temperature) 조정**
현재 `temperature: 0.2` (안정적)
- 더 창의적 답변 원하면: `0.3-0.4`
- 더 일관적 답변 원하면: `0.1` (현재도 충분)

### 2. **max_new_tokens 조정**
현재 `max_new_tokens: 300`
- 간결한 답변 강제: `200` (권장)
- 상세한 설명 필요시: `400`

### 3. **모델 변경**
현재 `gpt-4.1`
- 비용 절감: `gpt-4o-mini` (품질 약간 하락)
- 최고 품질: `gpt-4-turbo` 또는 `gpt-4o` (비용 증가)

---

## 🔧 Troubleshooting

### 문제 1: 여전히 답변이 길어요
**해결:**
```yaml
# config/enhanced.yaml 수정
llm:
  max_new_tokens: 200  # 300 → 200으로 감소
```

### 문제 2: 출처 표시가 없어요
**확인:**
- `system_v2.txt`에 "📚 참고:" 형식 명시되어 있음
- GPT가 문서를 찾았는지 확인 (Context Quality Filter 로그)

### 문제 3: 여전히 "요약:", "세부 단계:" 나와요
**해결:**
- `prompts/system_v2.txt` 파일 내용 확인
- "금지 사항" 섹션에 명시되어 있는지 확인
- Temperature를 0.1로 낮추기

### 문제 4: 환각이 발생해요
**해결:**
- Context Quality Filter 활성화 확인: `context_quality.enabled: true`
- System prompt에 "문서에만 기반" 원칙 명확히
- Temperature 낮추기: `0.1`

---

## 📚 참고 자료

### 논문
1. **Constitutional AI** (Anthropic 2022): https://arxiv.org/abs/2212.08073
2. **Self-RAG** (Asai et al., 2023): https://arxiv.org/abs/2310.11511
3. **Chain-of-Thought Prompting** (Wei et al., 2022): https://arxiv.org/abs/2201.11903
4. **Few-shot Learning** (Brown et al., 2020): https://arxiv.org/abs/2005.14165

### 블로그/가이드
- OpenAI Prompt Engineering Guide: https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic Prompt Library: https://docs.anthropic.com/claude/prompt-library
- LangChain RAG Best Practices: https://python.langchain.com/docs/use_cases/question_answering/

---

## ✅ 최종 권장사항

1. ✅ **system_v2.txt 사용** (enhanced.yaml에 이미 설정됨)
2. ✅ **재인덱싱 완료** (chunk_size: 900, overlap: 180)
3. ✅ **간단한 테스트** (3-5개 질문)
4. ✅ **RAGAS 평가** (최종 성능 측정)
5. ✅ **피드백 반영** (필요시 추가 튜닝)

---

**이제 Runpod에서 테스트해보세요!** 답변이 훨씬 자연스럽고 간결해졌을 거예요. 🚀
