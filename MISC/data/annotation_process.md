# Manual annotation protocol — `manual_annotation_200.csv`

COMP6713 · bravo11234 · Part B (team-collected dataset, 20 credits)

This document records **how** the 232-row manual annotation set was
constructed, and **how** inter-annotator agreement (Cohen's κ) was
measured. It is the primary evidence behind the 20-credit team-dataset
claim in `CONTRIBUTION.md` — the report §Annotation cites this file.

---

## 1. Goal

Build a small, **high-quality**, **out-of-distribution** evaluation
set. Not used for training any of the primary models (BERT-clean,
RoBERTa, DistilBERT, Naive Bayes) — only for evaluation and
qualitative error analysis. The BERT-aug ablation explicitly uses a
subset of this file for training; see top-level `README.md` §5.3 for
the methodological caveat there.

OOD is deliberate: the Jigsaw + Civil Comments training distribution is
news-article / Wikipedia discussion comments, whereas we want to stress
the models on short, informal, emoji-and-slang-laden social-media
content that the training distribution under-represents.

## 2. Data collection

232 comments, drawn from two sources:

1. **Stratified sample from the 100 k merged Jigsaw + Civil Comments
   pool** — 150 comments. Labels were discarded before annotation, and
   the sampled rows were **removed from the training pool** to avoid
   test-leakage.
2. **Fresh social-media collection** — 82 comments collected manually
   from Reddit, Instagram, YouTube, and X (Twitter) after the first
   round of annotation. Over-sampled along three dimensions we
   identified as hard in round 1: long comments with attitude shifts,
   current internet slang, and sarcasm / irony.

Final file at `MISC/data/manual_annotation_200.csv` has two columns:

| Column | Type | Meaning |
|---|---|---|
| `cleaned_text` | str | the comment after the same light cleaning applied to the transformer training data (URL / HTML / @-handle stripped, everything else preserved) |
| `toxic` | {0, 1} | final adjudicated binary label |

Class balance: **37.9 % toxic (88 / 232)**. This is deliberately much
higher than the training distribution's 9.2 %, so that recall on the
toxic class is stressed.

## 3. Annotation protocol

Binary labels under the following definition:

| Label | Definition |
|---|---|
| **Toxic (1)** | Insults, hate speech, harassment, threats, or language that attacks a person or a group |
| **Non-toxic (0)** | Neutral, factual, opinionated but non-offensive, or strong opinion without a personal attack |

Each comment was **independently** labelled by two annotators, then
compared. Disagreements were resolved through discussion and — where
needed — refinement of the labelling guideline.

## 4. Inter-annotator agreement

Cohen's κ was computed on the independent annotations using scikit-learn's
`sklearn.metrics.cohen_kappa_score` (equivalent closed-form formula
for a 2 × 2 agreement matrix).

### Round 1 (initial 200 comments)

- Two independent annotators labelled all ~200 comments.
- **κ = 0.40** → moderate agreement (Landis & Koch, 1977).
- **44 disagreement cases** identified. These were reviewed and
  re-labelled through discussion, and retained in the final set as
  "high-value hard examples".

### Round 2 (refinement: 150 new comments)

Between rounds, the annotators discussed and refined the labelling
guideline, focusing on four categories that had driven most of the
round-1 disagreements:

- Long comments with attitude shifts
- Emerging internet slang
- Sarcastic / ironic expressions
- Borderline neutral-vs-toxic cases

150 freshly collected social-media comments were then independently
annotated under the refined guideline.

- **κ = 0.73** → substantial agreement.
- **~19 disagreement cases** remained; all were resolved by
  discussion to produce the final label.

### Interpretation

The κ improvement from 0.40 to 0.73 across the two rounds reflects
the genuine subjectivity of the "toxic vs strong opinion" boundary,
particularly on political content, sarcasm, and implicit toxicity
without explicit slurs. This is consistent with what the models
themselves struggle with on the same set (recall-tox ~0.36–0.42 for
every in-domain transformer — see top-level `README.md` §5.4).

## 5. Final set composition

| Subset | N | Notes |
|---|---:|---|
| Round-2 newly annotated comments (high consistency) | ~170 | post-refinement round, κ ≈ 0.73 |
| Round-1 retained difficult examples | 44 | previously disagreed, adjudicated |
| Round-1 originally agreed comments | ~60 | both annotators matched on first pass |
| **After de-duplication** | **232** | final file size |

(The initially quoted "200" in file / variable names is a legacy
rounding from the design phase — the final adjudicated set is 232
rows, ~234 including a few duplicates that were later merged.)
