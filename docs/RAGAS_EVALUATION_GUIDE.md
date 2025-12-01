# 🎯 RAGAS 평가 가이드

## 📊 평가 데이터셋 개요

### 총 80개 질문 (신뢰성 있는 평가)

**도메인별 분포:**
- Python: 40개 (50%)
- Git: 40개 (50%)

**난이도 분포:**
- Easy: 32개 (40%)
- Medium: 32개 (40%)
- Hard: 16개 (20%)

**질문 유형:**
1. **개념 설명** (Concept Explanation): "~는 무엇인가요?"
2. **사용법** (Usage/How-to): "~는 어떻게 사용하나요?"
3. **비교/차이** (Comparison): "A와 B의 차이는?"
4. **문제 해결** (Troubleshooting): "~문제를 해결하는 방법은?"
5. **Best Practice**: "~할 때 주의할 점은?"

---

## 🚀 평가 실행 방법

### 1. 사전 준비

**RAGAS 설치:**
```bash
pip install ragas datasets
```

**OpenAI API 키 설정:**
```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key"
```

**재인덱싱 확인 (중요!):**
```bash
# chunk_size=900, overlap=180으로 재인덱싱 완료 확인
ls artifacts/chroma_db/  # 인덱스 존재 확인
```

---

### 2. 평가 실행

**전체 평가 (Python 40 + Git 40 = 80개):**
```bash
cd /workspace/rag_pipeline  # 또는 작업 디렉토리

python run_ragas_evaluation.py
```

**예상 소요 시간:**
- Pipeline 실행: 80개 × 3초 = 약 4분
- RAGAS 평가: 80개 × 5초 = 약 7분
- **총 약 11-15분**

---

### 3. 출력 파일

평가 완료 후 생성되는 파일:

```
artifacts/ragas_evals/
├── ragas_eval_20251120_143022_detailed.json  (상세 결과)
└── ragas_eval_20251120_143022_report.txt     (텍스트 보고서)
```

---

## 📈 평가 지표 설명

### RAGAS Metrics

**1. Context Precision (컨텍스트 정밀도)**
- **의미**: 검색된 문서가 질문과 얼마나 관련이 있는가?
- **목표**: 80%+
- **개선 방법**: Hybrid Search, Context Quality Filter

**2. Context Recall (컨텍스트 재현율)**
- **의미**: Ground truth 답변에 필요한 정보를 모두 검색했는가?
- **목표**: 85%+
- **개선 방법**: rerank_top_k 증가, Hybrid Search

**3. Faithfulness (충실도)**
- **의미**: 답변이 검색된 문서에만 기반하는가? (환각 방지)
- **목표**: 93%+
- **개선 방법**: Constitutional AI, Context Quality Filter

**4. Answer Relevancy (답변 관련성)**
- **의미**: 답변이 질문과 얼마나 관련이 있는가?
- **목표**: 80%+
- **개선 방법**: System Prompt 최적화, 간결한 답변

**5. Answer Correctness (답변 정확도)**
- **의미**: 답변이 Ground truth와 얼마나 일치하는가?
- **목표**: 75%+
- **개선 방법**: 전체 파이프라인 최적화

---

## 📊 예상 결과 (Target)

```
RAGAS EVALUATION RESULTS
================================================================================

context_precision       :  80.00%+  ✓ EXCELLENT
context_recall          :  85.00%+  ✓ EXCELLENT
faithfulness            :  93.00%+  ✓ EXCELLENT
answer_relevancy        :  80.00%+  ✓ EXCELLENT
answer_correctness      :  75.00%+  ✓ GOOD

================================================================================
```

---

## 🔍 결과 분석 방법

### 1. 전체 지표 확인

```bash
# 보고서 확인
cat artifacts/ragas_evals/ragas_eval_*_report.txt
```

**체크 포인트:**
- [ ] 모든 지표가 목표치 이상인가?
- [ ] 특정 지표가 낮다면 어느 것인가?
- [ ] 도메인별(Python/Git) 차이가 있는가?
- [ ] 난이도별(Easy/Medium/Hard) 차이가 있는가?

---

### 2. 상세 결과 분석

```python
import json

# 상세 결과 로드
with open("artifacts/ragas_evals/ragas_eval_*_detailed.json") as f:
    data = json.load(f)

# 실패한 질문 확인
failed = [r for r in data["results"] if not r["success"]]
print(f"Failed: {len(failed)}")

# 낮은 점수 질문 확인 (RAGAS 평가 후)
# Context Precision이 낮은 질문 찾기
```

---

### 3. 도메인별 분석

**Python vs Git:**
```python
python_results = [r for r in results if r["domain"] == "python"]
git_results = [r for r in results if r["domain"] == "git"]

# 각각의 성공률, 평균 응답 시간 비교
```

---

### 4. 난이도별 분석

**Easy vs Medium vs Hard:**
```python
easy = [r for r in results if r["difficulty"] == "easy"]
medium = [r for r in results if r["difficulty"] == "medium"]
hard = [r for r in results if r["difficulty"] == "hard"]

# 각 난이도별 성공률 분석
# Hard 질문에서 낮은 점수가 나오는 것은 자연스러움
```

---

## 🛠️ 성능 개선 전략

### Faithfulness가 낮은 경우 (< 90%)

**원인:**
- LLM이 문서에 없는 정보 추가 (환각)
- Context Quality Filter 미작동

**해결:**
1. System Prompt 강화:
   ```
   "검색된 문서에만 기반하여 답변" 강조
   ```
2. Context Quality Filter 임계값 조정:
   ```yaml
   context_quality:
     threshold: 0.7  # 0.6 → 0.7로 상향
   ```
3. Temperature 낮추기:
   ```yaml
   llm:
     temperature: 0.1  # 0.2 → 0.1
   ```

---

### Context Recall이 낮은 경우 (< 80%)

**원인:**
- 필요한 문서를 검색하지 못함
- rerank_top_k가 너무 작음

**해결:**
1. rerank_top_k 증가:
   ```yaml
   retrieval:
     rerank_top_k: 15  # 10 → 15로 증가
   ```
2. hybrid_top_k 증가:
   ```yaml
   retrieval:
     hybrid_dense_top_k: 70  # 50 → 70으로 증가
   ```

---

### Answer Correctness가 낮은 경우 (< 70%)

**원인:**
- 검색은 잘 되지만 답변 생성 품질이 낮음
- System Prompt 문제

**해결:**
1. Ground truth와 답변 스타일 비교
2. Few-shot 예시 개선
3. max_new_tokens 조정:
   ```yaml
   llm:
     max_new_tokens: 400  # 300 → 400
   ```

---

## 📝 평가 질문 예시

### Python 질문 (샘플)

**Easy:**
```
Q: Python에서 리스트(list)와 튜플(tuple)의 차이는 무엇인가요?
Ground Truth: 리스트는 mutable하여 수정 가능하지만, 튜플은 immutable하여 수정 불가능합니다.
```

**Medium:**
```
Q: Python 데코레이터(decorator)는 무엇이고 어떻게 작동하나요?
Ground Truth: 함수나 클래스를 수정하지 않고 기능을 추가하는 고차 함수입니다. @decorator 문법으로 사용합니다.
```

**Hard:**
```
Q: Python의 메타클래스(metaclass)는 무엇이고 언제 사용하나요?
Ground Truth: 클래스의 클래스로, 클래스 생성 과정을 커스터마이즈합니다. ORM, API 프레임워크 등에서 동적 클래스 생성 시 사용합니다.
```

---

### Git 질문 (샘플)

**Easy:**
```
Q: Git에서 새로운 브랜치를 만들고 전환하는 명령어는?
Ground Truth: git checkout -b <branch-name> 또는 git switch -c <branch-name>을 사용합니다.
```

**Medium:**
```
Q: Git에서 merge와 rebase의 차이는?
Ground Truth: merge는 두 브랜치를 합치는 새 커밋을 생성하고, rebase는 한 브랜치의 커밋을 다른 브랜치 위로 재배치합니다.
```

**Hard:**
```
Q: Git의 reflog는 무엇이고 언제 사용하나요?
Ground Truth: reflog는 HEAD와 브랜치 참조의 변경 기록을 저장합니다. 잘못된 reset이나 rebase 후 커밋 복구 시 사용합니다.
```

---

## ✅ 평가 체크리스트

### 평가 전
- [ ] RAGAS 설치 완료 (`pip install ragas datasets`)
- [ ] OpenAI API 키 설정
- [ ] 재인덱싱 완료 (chunk_size=900, overlap=180)
- [ ] answerer_v2.py 정상 작동 확인
- [ ] config/enhanced.yaml 설정 확인

### 평가 실행
- [ ] `python run_ragas_evaluation.py` 실행
- [ ] 에러 없이 완료
- [ ] 80개 질문 모두 처리

### 평가 후
- [ ] 결과 파일 생성 확인 (detailed.json, report.txt)
- [ ] 모든 지표 목표치 이상인지 확인
- [ ] 특정 지표가 낮다면 원인 분석
- [ ] 샘플 답변 품질 확인
- [ ] 도메인별/난이도별 분석

---

## 🚨 Troubleshooting

### "ModuleNotFoundError: ragas"
```bash
pip install ragas datasets
```

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY="your-key"  # Linux/Mac
$env:OPENAI_API_KEY="your-key"    # Windows
```

### "ChromaDB collection not found"
```bash
# 재인덱싱 필요
python data_prep.py --config config/enhanced.yaml
python index_builder.py --config config/enhanced.yaml
```

### "Evaluation takes too long"
- 정상적으로 11-15분 소요
- 네트워크 상태 확인 (OpenAI API 호출)
- GPU 사용 가능 시 빠름 (CUDA)

### "Some questions failed"
- 로그에서 실패 원인 확인
- 특정 질문이 지속적으로 실패하면 질문 수정 고려

---

## 📊 결과 해석 가이드

### Excellent (80%+)
✅ 목표 달성! 프로덕션 배포 고려 가능

### Good (70-80%)
✓ 양호한 성능, 일부 개선 필요

### Acceptable (60-70%)
⚠️  추가 최적화 필요

### Needs Improvement (< 60%)
✗ 근본적인 문제 분석 및 재구성 필요

---

## 📚 참고 자료

**RAGAS 공식 문서:**
- https://docs.ragas.io/

**RAGAS GitHub:**
- https://github.com/explodinggradients/ragas

**논문:**
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)

---

## 🎯 최종 목표

| Metric | Baseline | Enhanced | Target | Stretch |
|--------|----------|----------|--------|---------|
| Faithfulness | 45% | ? | **93%+** | 95%+ |
| Answer Correctness | 44% | ? | **75%+** | 80%+ |
| Context Precision | 65% | ? | **80%+** | 85%+ |
| Context Recall | 68% | ? | **85%+** | 90%+ |
| Answer Relevancy | 70% | ? | **80%+** | 85%+ |

**이제 평가를 실행하고 목표를 달성했는지 확인해보세요!** 🚀
