# ROADMAP — What's Up Docs: Document Summarization with LLMs

Status indicators:

- `[x]` Completed
- `[-]` In Progress
- `[ ]` Not Started
- `[!]` Blocked / Needs Decision

---

## Phase 1 — Setup

> Goal: Establish a clean, reproducible development environment.

| #   | Task                                                              | Status | Notes                                          |
|-----|-------------------------------------------------------------------|--------|------------------------------------------------|
| 1.1 | Initialize git repository                                         | `[x]`  | Done — two initial commits on `main`           |
| 1.2 | Create directory skeleton (`data/`, `notebooks/`, `src/`, `outputs/`) | `[x]` | Done                                      |
| 1.3 | Move raw CSVs from project root → `data/raw/`                     | `[x]`  | `train.csv`, `test_features.csv`, `submission_format.csv` |
| 1.4 | Create `requirements.txt` with core dependencies                  | `[x]`  | See dependencies below                         |
| 1.5 | Create `.env.example` with all required API key names             | `[x]`  | Never commit actual keys                       |
| 1.6 | Create `.gitignore`                                               | `[x]`  | Covers `.venv`, `.env`, `outputs/`, `__pycache__` |
| 1.7 | Create virtual environment (`python -m venv .venv`)               | `[ ]`  | Python 3.10+ required                          |
| 1.8 | Install dependencies (`pip install -r requirements.txt`)          | `[ ]`  |                                                |
| 1.9 | Configure API keys (copy `.env.example` → `.env`, fill in values) | `[ ]` |                                               |
| 1.10 | Verify API keys work (smoke test with a minimal API call)        | `[ ]`  |                                                |

**Core dependencies (`requirements.txt`):**

```
openai>=1.30
anthropic>=0.28
pandas>=2.0
rouge-score>=0.1.2
tqdm>=4.66
python-dotenv>=1.0
tiktoken>=0.7
transformers>=4.40
jupyter>=1.0
```

---

## Phase 2 — Exploratory Data Analysis (EDA)

> Goal: Understand the data distribution and surface key challenges before building any model.

| #   | Task                                                                       | Status | Notes                                |
|-----|----------------------------------------------------------------------------|--------|--------------------------------------|
| 2.1 | Load `data/raw/train.csv` and inspect schema (nulls, duplicates, encoding) | `[ ]`  |                                      |
| 2.2 | Analyze text length distribution (chars, words, tokens via `tiktoken`)     | `[ ]`  | Use `cl100k_base` tokenizer          |
| 2.3 | Identify papers exceeding common context limits (4k, 8k, 16k, 128k tokens) | `[ ]` | Critical for strategy selection      |
| 2.4 | Analyze summary length distribution                                        | `[ ]`  | Target: ~184 words average           |
| 2.5 | Identify Markdown structure patterns (`#` headings, section counts)        | `[ ]`  | Informs section-aware splitting      |
| 2.6 | Sample and manually read 5–10 paper + summary pairs                        | `[ ]`  | Build intuition for quality          |
| 2.7 | Document findings in `notebooks/01_eda.ipynb`                              | `[ ]`  |                                      |

**Key questions to answer:**
- What fraction of papers exceed 128k tokens? (OpenAI limit) / 200k tokens? (Anthropic limit)
- What is the distribution of Markdown section counts across papers?
- How consistent is the summary style (length, format, first vs. third person)?
- Are there any papers with corrupted or truncated text?

---

## Phase 3 — Baseline

> Goal: Establish a reproducible lower-bound ROUGE-2 score as quickly as possible.

| #   | Task                                                               | Status | Notes                                        |
|-----|--------------------------------------------------------------------|--------|----------------------------------------------|
| 3.1 | Implement `src/summarizer.py` with a basic OpenAI API call         | `[ ]`  | Truncate to 3,000 tokens, zero-shot          |
| 3.2 | Implement `src/evaluation.py` with a ROUGE-2 scoring function      | `[ ]`  | Wraps `rouge_score` library                  |
| 3.3 | Run baseline on the full training set, record ROUGE-2 F1           | `[ ]`  | Use `gpt-4o-mini` for cost efficiency        |
| 3.4 | Run baseline on test set, generate first submission CSV            | `[ ]`  | Validate format against `submission_format.csv` |
| 3.5 | Document baseline prompt and score in `notebooks/02_baseline.ipynb` | `[ ]` |                                              |

**Starting prompt:**

```
You are an expert academic editor. Read the following social science paper
and write a single concise paragraph summarizing its research question,
methodology, key findings, and contributions. Write approximately 150–200 words.

PAPER:
{text}
```

**Cost estimate (baseline):**
- Model: `gpt-4o-mini` (~$0.15 / 1M input tokens)
- ~3,000 tokens input × 345 papers ≈ ~1M tokens → **~$0.15 for the full test set**

---

## Phase 4 — Iteration

> Goal: Systematically improve ROUGE-2 through better chunking, prompting, and model selection.

### 4A — Chunking Strategies for Long Documents

| #    | Task                                                                  | Status | Notes                             |
|------|-----------------------------------------------------------------------|--------|-----------------------------------|
| 4A.1 | Implement `truncate` in `src/preprocessing.py`                        | `[ ]`  | Already in baseline               |
| 4A.2 | Implement `section_split` (split on `#` Markdown headings)            | `[ ]`  | Preserve section labels           |
| 4A.3 | Implement `map_reduce` (summarize chunks independently, then combine) | `[ ]`  | Two LLM call stages               |
| 4A.4 | Implement `sliding_window` with configurable overlap                  | `[ ]`  | Overlap = 10% of chunk size       |
| 4A.5 | Implement `hierarchical` (chunks → section summaries → final para)    | `[ ]`  | Three-stage, highest quality      |
| 4A.6 | Benchmark all strategies on a 50-paper dev split from training set    | `[ ]`  | Use ROUGE-2 as selection criterion |

> Papers can reach ~197,000 chars (~50,000 tokens). Even 200k-context models may struggle.
> `map_reduce` and `hierarchical` are the most robust strategies for very long inputs.

### 4B — Prompt Engineering

| #    | Task                                                                         | Status | Notes                          |
|------|------------------------------------------------------------------------------|--------|--------------------------------|
| 4B.1 | Vary system prompt persona ("editor", "reviewer", "journalist")              | `[ ]`  |                                |
| 4B.2 | Test explicit length constraints ("exactly 180 words", "3–4 sentences")      | `[ ]`  |                                |
| 4B.3 | Add instruction to cover: RQ, methodology, findings, contributions           | `[ ]`  |                                |
| 4B.4 | Test 1-shot and 3-shot prompting using examples from training set            | `[ ]`  | Pick diverse examples          |
| 4B.5 | Test chain-of-thought: "First identify key claims, then write summary"       | `[ ]`  |                                |
| 4B.6 | Grid-search best prompt × chunking strategy combination                      | `[ ]`  | Use 50-paper dev split         |
| 4B.7 | Document all variants and scores in `notebooks/03_prompt_engineering.ipynb`  | `[ ]`  |                                |

### 4C — Model Selection

| #    | Task                                                              | Status | Notes                                       |
|------|-------------------------------------------------------------------|--------|---------------------------------------------|
| 4C.1 | Test `claude-haiku-4-5` (200k context, low cost)                  | `[ ]`  | May handle full papers without chunking     |
| 4C.2 | Test `claude-sonnet-4-6` (200k context, higher quality)           | `[ ]`  | Higher cost — use on best prompt only       |
| 4C.3 | Test `gpt-4o` (128k context)                                      | `[ ]`  | Spot-check on hard (long) papers only       |
| 4C.4 | Test `facebook/bart-large-cnn` (HuggingFace, free, 1k ctx)        | `[ ]`  | Lower quality but zero API cost             |
| 4C.5 | Select best model based on ROUGE-2 / cost trade-off               | `[ ]`  |                                             |

---

## Phase 5 — Evaluation

> Goal: Rigorous comparison of all approaches to select the final submission strategy.

| #   | Task                                                                      | Status | Notes                                     |
|-----|---------------------------------------------------------------------------|--------|-------------------------------------------|
| 5.1 | Define 80/20 train/dev split (fix random seed for reproducibility)        | `[ ]`  | 800 train / 200 dev                       |
| 5.2 | Run all surviving strategies on full dev set                              | `[ ]`  |                                           |
| 5.3 | Compute ROUGE-2 F1 for every approach (mean + std across 200 papers)      | `[ ]`  |                                           |
| 5.4 | Analyze failure cases by paper length quartile                            | `[ ]`  | Where does each strategy underperform?    |
| 5.5 | Select best single strategy for final submission                          | `[ ]`  |                                           |
| 5.6 | (Optional) Ensemble: re-rank or fuse multiple strategy outputs            | `[ ]`  | Re-rank by self-ROUGE against partial refs |
| 5.7 | Document all results in `notebooks/04_evaluation.ipynb`                   | `[ ]`  | Include comparison table + error analysis |

**Results tracking table:**

| Strategy     | Model         | Prompt | Dev ROUGE-2 F1 | Cost (test set) | Notes    |
|--------------|---------------|--------|----------------|-----------------|----------|
| truncate     | gpt-4o-mini   | v1     | —              | —               | baseline |
| map_reduce   | gpt-4o-mini   | v2     | —              | —               |          |
| hierarchical | claude-haiku-4-5 | v3  | —              | —               |          |

---

## Phase 6 — Submission

> Goal: Generate and validate the final submission file for the competition.

| #   | Task                                                                   | Status | Notes                                  |
|-----|------------------------------------------------------------------------|--------|----------------------------------------|
| 6.1 | Run winning strategy on full test set (`data/raw/test_features.csv`)   | `[ ]`  | 345 papers                             |
| 6.2 | Validate output CSV matches `data/raw/submission_format.csv` schema    | `[ ]`  | Same `paper_id` order, same columns    |
| 6.3 | Check for missing values, empty strings, or encoding issues            | `[ ]`  |                                        |
| 6.4 | Verify summary lengths are in a reasonable range (100–300 words)       | `[ ]`  |                                        |
| 6.5 | Save final file to `outputs/submissions/final_submission.csv`          | `[ ]`  |                                        |
| 6.6 | Submit to competition platform                                         | `[ ]`  |                                        |
| 6.7 | Record official ROUGE-2 score and update `README.md` results table     | `[ ]`  |                                        |

---

## Backlog / Future Ideas

Out of scope for the initial submission, but worth exploring if time permits:

- **Fine-tuning** — fine-tune a smaller open-source model (e.g., Mistral-7B) on the training set
- **Retrieval-augmented summarization** — index paper sections with embeddings, retrieve most relevant before summarizing
- **Summary fusion** — generate N candidate summaries per paper, pick the best by self-ROUGE
- **Output post-processing** — strip artifacts, enforce word count, normalize formatting
- **Caching layer** — cache LLM responses keyed by `(paper_id, model, prompt_hash)` to avoid re-running
- **Async batching** — use async API calls to process all 345 test papers in parallel

---

## Suggested Timeline

| Week   | Focus                                        |
|--------|----------------------------------------------|
| Week 1 | Phase 1 (Setup) + Phase 2 (EDA)              |
| Week 2 | Phase 3 (Baseline) + Phase 4A (Chunking)     |
| Week 3 | Phase 4B (Prompts) + Phase 4C (Models)       |
| Week 4 | Phase 5 (Evaluation) + Phase 6 (Submission)  |

---

*Last updated: 2026-03-19*
