# What's Up Docs — Document Summarization with LLMs

A data science competition project for **automatic summarization of social science academic papers**.
Given a full paper in Markdown format, the task is to generate a concise one-paragraph summary
and maximize the **ROUGE-2 F1** score against human-written reference summaries.

Papers are sourced from [SocArXiv](https://socopen.org/) and can be extremely long
(up to ~197,000 characters). A key engineering challenge is fitting these documents into
LLM context windows using chunking, truncation, or map-reduce strategies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Evaluation Metric](#evaluation-metric)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Approach](#approach)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

| Property           | Value                                      |
|--------------------|--------------------------------------------|
| Task               | Abstractive text summarization             |
| Domain             | Social science academic papers (SocArXiv)  |
| Input format       | Markdown full-text papers                  |
| Output format      | One-paragraph plain-text summary           |
| Train samples      | 1,000 papers                               |
| Test samples       | 345 papers                                 |
| Avg. input length  | ~42,000 characters (~6,300 words)          |
| Avg. summary length | ~1,275 characters (~184 words)            |
| Compression ratio  | ~97% reduction                             |
| Metric             | ROUGE-2 F1 (average across all documents)  |

---

## Dataset

Raw data lives under `data/raw/` and should never be modified:

| File                              | Rows | Columns                        | Description                           |
|-----------------------------------|------|--------------------------------|---------------------------------------|
| `data/raw/train.csv`              | 1000 | `paper_id`, `text`, `summary`  | Training set with reference summaries |
| `data/raw/test_features.csv`      |  345 | `paper_id`, `text`             | Test set — no summary provided        |
| `data/raw/submission_format.csv`  |  345 | `paper_id`, `summary`          | Template for the submission CSV       |

All papers are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and
sourced from [SocArXiv](https://socopen.org/).

### Key Challenge: Long Documents

Papers range from ~10,000 to ~197,000 characters. Most LLMs have context window limits
(e.g., 8k–128k tokens), so naive single-call prompting will fail for a significant portion
of the dataset. Strategies explored in this project:

| Strategy          | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `truncate`        | Keep only the first N tokens — fast but lossy                     |
| `section_split`   | Split on Markdown `#` headings, summarize each section            |
| `map_reduce`      | Summarize chunks independently, then combine into one paragraph   |
| `sliding_window`  | Overlapping token windows, merge with deduplication               |
| `hierarchical`    | Multi-stage: chunks → section summaries → final paragraph         |

---

## Evaluation Metric

The official metric is **ROUGE-2 F1**, averaged across all 345 test documents.

ROUGE-2 measures the overlap of **bigrams** (consecutive word pairs) between the generated
summary and the reference summary:

```
ROUGE-2 Precision = (matching bigrams) / (bigrams in generated summary)
ROUGE-2 Recall    = (matching bigrams) / (bigrams in reference summary)
ROUGE-2 F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

A score of **1.0** means perfect bigram overlap; **0.0** means no overlap.
Typical competitive scores for academic summarization range from 0.10 to 0.30.

Computing ROUGE-2 locally:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
result = scorer.score(reference_summary, generated_summary)
print(result["rouge2"].fmeasure)
```

---

## Project Structure

```
.
├── data/
│   ├── raw/                          # Original competition CSVs (do not modify)
│   │   ├── train.csv
│   │   ├── test_features.csv
│   │   └── submission_format.csv
│   └── processed/                    # Cleaned / chunked data for pipeline use
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_baseline.ipynb             # Baseline summarization experiments
│   ├── 03_prompt_engineering.ipynb   # Prompt iteration and comparison
│   └── 04_evaluation.ipynb           # ROUGE-2 scoring and model comparison
├── src/
│   ├── preprocessing.py              # Text cleaning, chunking, tokenization
│   ├── summarizer.py                 # LLM summarization logic
│   └── evaluation.py                 # ROUGE-2 scoring utilities
├── outputs/
│   ├── summaries/                    # Generated summaries per model/run
│   └── submissions/                  # Final submission CSV files
├── .env                              # API keys — NEVER commit this file
├── .env.example                      # Template with required variable names
├── .gitignore
├── requirements.txt
├── README.md
└── ROADMAP.md
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- An OpenAI and/or Anthropic API key (for proprietary LLM approaches)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/What-s-Up-Docs-Document-Summarization-with-LLMs.git
cd What-s-Up-Docs-Document-Summarization-with-LLMs
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACEHUB_API_TOKEN=hf_...
```

> `.env` is listed in `.gitignore` and will never be committed.

---

## Usage

### Run the summarization pipeline

```bash
python src/summarizer.py \
  --input data/raw/test_features.csv \
  --output outputs/submissions/submission_v1.csv \
  --model gpt-4o-mini \
  --strategy map_reduce \
  --chunk-size 4000
```

### Evaluate against the training set

```bash
python src/evaluation.py \
  --predictions outputs/summaries/train_predictions.csv \
  --references data/raw/train.csv \
  --output outputs/rouge2_scores.csv
```

### Run notebooks

```bash
jupyter lab
```

Open notebooks in order: `01_eda.ipynb` → `02_baseline.ipynb` → `03_prompt_engineering.ipynb` → `04_evaluation.ipynb`

---

## Approach

### Chunking Strategies

| Strategy         | Description                                              | Best for              |
|------------------|----------------------------------------------------------|-----------------------|
| `truncate`       | Keep first N tokens only                                 | Short papers, baseline |
| `section_split`  | Split on Markdown `#` headings, summarize each           | Well-structured papers |
| `map_reduce`     | Summarize chunks, then combine summaries into one        | Very long papers       |
| `sliding_window` | Overlapping token windows, merge with deduplication      | Dense, no clear sections |
| `hierarchical`   | Two-stage: chunk summaries → final paragraph             | Maximum quality        |

### Models Evaluated

| Model                     | Provider      | Context window | Cost tier |
|---------------------------|---------------|----------------|-----------|
| `gpt-4o-mini`             | OpenAI        | 128k tokens    | Low       |
| `gpt-4o`                  | OpenAI        | 128k tokens    | High      |
| `claude-haiku-4-5`        | Anthropic     | 200k tokens    | Low       |
| `claude-sonnet-4-6`       | Anthropic     | 200k tokens    | Medium    |
| `facebook/bart-large-cnn` | HuggingFace   | 1,024 tokens   | Free      |
| `google/pegasus-large`    | HuggingFace   | 1,024 tokens   | Free      |

---

## Results

| Model / Strategy               | ROUGE-2 F1 | Cost (test set) | Notes       |
|--------------------------------|------------|-----------------|-------------|
| Baseline (truncate, gpt-4o-mini) | —        | —               | To be filled |
| Map-reduce (gpt-4o-mini)       | —          | —               | To be filled |
| Hierarchical (claude-haiku-4-5) | —         | —               | To be filled |

> This table will be updated as experiments are completed.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full phased development plan.

---

## Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/my-experiment`
3. Commit with a descriptive message
4. Open a pull request against `main`

Conventions:
- One notebook per major experiment, named with a numeric prefix (e.g., `05_my_experiment.ipynb`)
- All reusable logic goes into `src/`, not inline in notebooks
- Never commit `.env`, large model weights, or generated CSVs larger than 10 MB

---

## License

This project is for competition and educational purposes.
Data is provided by the competition organizers under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Code in this repository is released under the [MIT License](LICENSE).
