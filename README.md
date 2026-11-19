# ReadmitAI

**A multimodal clinical AI system for predicting 30-day ICU readmission.**

ReadmitAI is built on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), a de-identified database of 40,000+ real ICU patients from Beth Israel Deaconess Medical Center. It combines structured clinical features (labs, diagnoses, medications) with unstructured discharge notes — running a scispaCy NER pipeline for medical entity and SDoH extraction, an XGBoost tabular model with SHAP interpretability, and a fine-tuned ClinicalBERT text model — fused via a late-fusion MLP for final readmission risk prediction.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Model](https://img.shields.io/badge/model-ClinicalBERT-purple)
![Tabular](https://img.shields.io/badge/tabular-XGBoost-orange)
![NER](https://img.shields.io/badge/NER-scispaCy-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Architecture

```
MIMIC-III tables
    |
    v
Data Pipeline (ingest/)
    |-- ADMISSIONS + PATIENTS     -> demographics, label engineering (30-day readmission)
    |-- DIAGNOSES_ICD             -> ICD-9 code features
    |-- LABEVENTS                 -> lab time-series aggregates (mean/min/max/last)
    |-- PRESCRIPTIONS             -> medication count features
    |-- NOTEEVENTS                -> discharge note text
    |
    v
Feature Engineering (features/)
    |-- Tabular feature matrix    -> XGBoost classifier + SHAP
    |-- NER pipeline (ner/)       -> scispaCy entity extraction + SDoH flags
    |
    v
Text Pipeline (models/)
    |-- ClinicalBERT fine-tuned on discharge notes -> readmission probability + [CLS] embedding
    |
    v
Late Fusion MLP
    |-- XGBoost probability score + ClinicalBERT [CLS] embedding -> final prediction
    |
    v
FastAPI  (POST /predict)
    |-- readmission_probability, top SHAP features, extracted NER entities
```

**Tabular path:** six MIMIC-III tables joined and aggregated into a structured feature matrix -> XGBoost trained with `scale_pos_weight` for class imbalance -> SHAP global/local explanations.

**Text path:** discharge summaries tokenized and truncated to 512 tokens (first-512 strategy) -> `emilyalsentzer/Bio_ClinicalBERT` fine-tuned for binary readmission classification.

**NER pipeline:** scispaCy `en_ner_bc5cdr_md` extracts diseases and drugs from discharge notes -> custom rule-based patterns extract Social Determinants of Health (smoking, alcohol, housing instability) -> entity counts and SDoH flags added to tabular feature matrix.

---

## Evaluation Results

| Model | AUROC | AUPRC |
|---|---|---|
| Tabular (XGBoost) | TBD | TBD |
| Text (ClinicalBERT) | TBD | TBD |
| Fusion (Multimodal) | TBD | TBD |

Evaluated on a held-out test split stratified by readmission label. AUPRC reported alongside AUROC due to class imbalance (~12% positive rate in MIMIC-III).

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Dataset | MIMIC-III (PhysioNet) | Real credentialed ICU data, gold standard for clinical ML |
| Label | 30-day readmission (engineered) | No readmission column — derived from admission timestamps |
| NER | scispaCy `en_ner_bc5cdr_md` | Pre-trained on biomedical text, plug-and-play entity extraction |
| SDoH | Custom rule-based patterns | Keyword + negation detection for social risk signals in notes |
| Tabular model | XGBoost + SHAP | Trees outperform NNs on tabular data; SHAP has native support |
| Text model | Bio_ClinicalBERT | Pre-trained on MIMIC-III notes — same domain as inference data |
| Truncation | First 512 tokens | Admission diagnosis and chief complaint appear early in notes |
| Fusion | Late fusion MLP | Each modality optimally processed independently before combining |
| Interpretability | SHAP (tabular) + LIME (text) | Required for clinical auditability |
| API | FastAPI + uvicorn | Auto-docs, Pydantic validation, async-ready |
| Deployment | Docker + HuggingFace Spaces | Reproducible environment, public live demo |

---

## Repository Structure

```
readmitai/
  data/           raw/              MIMIC-III tables (gitignored — credentialed data)
                  processed/        Engineered feature matrices (gitignored)
  ingest/                           MIMIC table loaders, label engineering
  features/                         Tabular feature engineering, NER integration
  ner/                              NER pipeline (scispaCy + SDoH rule patterns)
  models/                           XGBoost, ClinicalBERT, late-fusion MLP training
  eval/                             AUROC/AUPRC evaluation, SHAP plots, subgroup analysis
  api/            main.py           FastAPI app, /predict and /health endpoints
                  models.py         Pydantic request/response models
  ui/             app.py            Gradio demo (HuggingFace Spaces)
  notebooks/      01_eda.ipynb      MIMIC-III schema exploration and label distribution
  Dockerfile
  requirements.txt
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Environment, MIMIC-III demo data, EDA notebook | In progress |
| 1 | Data pipeline: table joins, label engineering, tabular feature matrix | Planned |
| 2 | NER pipeline: scispaCy entity extraction + SDoH rule patterns | Planned |
| 3 | Tabular model: XGBoost + SHAP interpretability | Planned |
| 4 | Text model: ClinicalBERT fine-tuning on discharge summaries | Planned |
| 5 | Multimodal fusion: late-fusion MLP combining both modalities | Planned |
| 6 | Interpretability: subgroup analysis, calibration, LIME text explanations | Planned |
| 7 | FastAPI REST endpoint + Docker | Planned |
| 8 | Gradio demo + HuggingFace Spaces deployment | Planned |

---

## Quick Start

```bash
git clone https://github.com/Victorine07/ReadmitAI.git
cd ReadmitAI
conda create -n readmitai python=3.11 && conda activate readmitai
pip install -r requirements.txt
# Download MIMIC-III data from PhysioNet (credentialed access required)
# Place CSV files in data/raw/mimic-iii-clinical-database-1.4/
jupyter lab notebooks/01_eda.ipynb
```

> MIMIC-III requires completing CITI human subjects research training and signing a data use agreement at [physionet.org](https://physionet.org/content/mimiciii/1.4/). The demo subset (100 patients) is available without credentialing.

---

## Author

**Victorine Younang, Ph.D.** — Machine Learning / Clinical AI
[LinkedIn](https://www.linkedin.com/in/victorine-wakam-younang-4947b712b) · [GitHub](https://github.com/Victorine07) · [Google Scholar](https://scholar.google.com/citations?user=8LaXG4AAAAAJ)
