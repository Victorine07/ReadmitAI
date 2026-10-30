# ReadmitAI — Clinical 30-Day Readmission Prediction

Multimodal deep learning system for predicting 30-day hospital readmission from MIMIC-III ICU data. Combines structured clinical features (vitals, labs, diagnoses, medications) with unstructured discharge notes via ClinicalBERT, with a NER pipeline for medical entity and SDoH extraction, SHAP/LIME interpretability, and a live Gradio demo.

**Dataset:** MIMIC-III Clinical Database (PhysioNet, credentialed access)
**Evaluation target:** AUROC ≥ 0.80, AUPRC reported
**Demo:** *(link added on deployment)*

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data | MIMIC-III (40K+ ICU patients) | Gold-standard real clinical data |
| NER | scispaCy + BioBERT | Pre-trained on biomedical text |
| Tabular model | XGBoost + SHAP | Interpretable, strong clinical baseline |
| Text model | ClinicalBERT (fine-tuned) | Pre-trained on MIMIC discharge notes |
| Fusion | Late fusion (concat + MLP) | Simple, interpretable, auditable |
| Interpretability | SHAP (tabular) + LIME (text) | Required for clinical AI credibility |
| API | FastAPI + Docker | Production deployment |
| Demo | Gradio on HuggingFace Spaces | Public live demo |

---

## Roadmap

### Phase 0 — Environment + Data + EDA
**Goal:** working environment, MIMIC-III demo data loaded, understand the schema.

- Apply for MIMIC-III credentialed access on PhysioNet
- Download MIMIC-III Demo (100 patients, open access) to start immediately
- Create conda environment, install Phase 0 deps
- Load core MIMIC tables: `ADMISSIONS`, `PATIENTS`, `DIAGNOSES_ICD`, `NOTEEVENTS`
- EDA notebook: admission counts, readmission rates, note lengths, missing data, class imbalance
- **Output:** EDA notebook, understanding of raw schema

---

### Phase 1 — Data Pipeline & Label Engineering
**Goal:** clean tabular feature matrix + 30-day readmission binary label, ready for modeling.

**Key tables to join:**
- `ADMISSIONS` — admission/discharge timestamps, diagnosis, insurance, marital status
- `PATIENTS` — DOB, gender, DOD
- `DIAGNOSES_ICD` — ICD-9 diagnosis codes per admission
- `PROCEDURES_ICD` — procedure codes
- `LABEVENTS` — lab results (glucose, creatinine, BUN, etc.)
- `PRESCRIPTIONS` — medications
- `NOTEEVENTS` — discharge summaries (text)

**Label engineering:** for each admission, look up the patient's next admission. If it occurs within 30 days and is not a scheduled transfer, label = 1 (readmitted). Exclude deaths during admission.

**Tabular features to engineer:**
- Demographics: age at admission, gender
- Administrative: insurance type, admission type (emergency/elective/urgent), marital status
- Clinical: top-N ICD-9 diagnosis codes (one-hot or embedding), procedure count, number of diagnoses
- Labs: mean/min/max/last value for key labs (glucose, creatinine, BUN, WBC, hemoglobin)
- Medications: medication count, high-risk drug flags
- Temporal: length of stay, hour of admission, prior admissions count

**Output:** `data/processed/tabular_features.parquet`, `data/processed/labels.parquet`

---

### Phase 2 — NER Pipeline
**Goal:** extract structured medical entities and SDoH signals from discharge notes.

**Tool:** scispaCy (`en_ner_bc5cdr_md` model) for biomedical NER + custom rules for SDoH.

**Entity types to extract:**
- Diseases and conditions (DISEASE)
- Chemicals and drugs (CHEMICAL)
- SDoH signals: smoking status, alcohol use, housing instability, employment — keyword + pattern rules

**Inputs:** `NOTEEVENTS.TEXT` (discharge summary for each admission)
**Outputs:** per-admission entity counts, SDoH flags → added as features to tabular matrix

**Output:** `ner/` module with `extract_entities(text: str) -> dict`, updated feature matrix

---

### Phase 3 — Tabular Baseline (XGBoost + SHAP)
**Goal:** strong baseline model on tabular features only; understand feature importance.

- Train/val/test split — stratified by label (handle class imbalance with `scale_pos_weight`)
- XGBoost classifier, hyperparameter tuning with cross-validation
- Evaluation: AUROC, AUPRC, calibration curve, confusion matrix at 0.5 threshold
- SHAP: global feature importance bar plot + beeswarm, local explanation for single patient
- **Target:** AUROC ≥ 0.75 on tabular alone

**Output:** `models/tabular_model.pkl`, `eval/tabular_results.json`, SHAP plots

---

### Phase 4 — Text Model (ClinicalBERT)
**Goal:** fine-tune ClinicalBERT on discharge summaries for readmission prediction.

**Model:** `emilyalsentzer/Bio_ClinicalBERT` from HuggingFace
**Input:** discharge summary text (truncated to 512 tokens)
**Output:** binary readmission probability

- Tokenize discharge notes, handle long documents (first 512 tokens = most predictive)
- Fine-tune with AdamW, learning rate warmup, early stopping
- Evaluation: same metrics as Phase 3 for fair comparison

**Output:** `models/text_model/` (fine-tuned weights), `eval/text_results.json`

---

### Phase 5 — Multimodal Fusion
**Goal:** combine tabular + text representations for best overall prediction.

**Strategy: late fusion**
- Run tabular model → get probability score
- Run ClinicalBERT → get probability score + [CLS] embedding
- Concatenate: [tabular features + tabular prob + CLS embedding] → small MLP head
- Train only the MLP head (freeze both base models initially)

**Evaluation:** compare Tabular-only vs Text-only vs Fusion on AUROC and AUPRC
**Target:** Fusion AUROC ≥ 0.80

**Output:** `models/fusion_model.pkl`, `eval/fusion_results.json`, comparison table

---

### Phase 6 — Interpretability
**Goal:** make the model auditable — required for any clinical AI claim.

- SHAP waterfall plots for individual patient predictions (tabular path)
- LIME text explanations: which words/phrases drove the readmission score
- Subgroup analysis: does performance hold across age groups, gender, insurance type?
- Calibration: are predicted probabilities meaningful (Brier score, reliability diagram)?

**Output:** `eval/interpretability/` with saved plots, `eval/subgroup_results.json`

---

### Phase 7 — FastAPI + Docker
**Goal:** production REST endpoint, same pattern as RepoSage.

**Endpoint:** `POST /predict`
- Input: structured patient features + discharge note text
- Output: readmission probability, top SHAP features, NER entities extracted

`GET /health` — liveness check

Dockerized with `python:3.11-slim`, environment variable injection at runtime.

**Output:** `api/main.py`, `Dockerfile`, `docker-compose.yml`

---

### Phase 8 — Gradio Demo + HuggingFace Spaces
**Goal:** public live demo with a realistic patient scenario.

**Demo design:**
- Input: pre-filled example patient (demographics + discharge note text)
- Editable fields so a visitor can change values and re-run prediction
- Output: readmission probability + confidence, top 5 SHAP features, NER entities found
- Note displayed: "This is a research demo trained on de-identified ICU data. Not for clinical use."

Deployed to HuggingFace Spaces (Gradio SDK, ZeroGPU).

**Output:** `ui/app.py`, deployed Space URL

---

## Evaluation Summary (target)

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Tabular (XGBoost) | ≥ 0.75 | reported |
| Text (ClinicalBERT) | ≥ 0.77 | reported |
| Fusion (Multimodal) | ≥ 0.80 | reported |

---

## Project Structure

```
readmitai/
├── data/
│   ├── raw/          # MIMIC-III tables — gitignored, never committed
│   └── processed/    # engineered features — gitignored
├── ingest/           # MIMIC table loaders, label engineering
├── features/         # tabular feature engineering, NER feature integration
├── ner/              # NER pipeline (scispaCy + SDoH rules)
├── models/           # training scripts for tabular, text, fusion models
├── eval/             # evaluation metrics, SHAP plots, results JSON
├── api/              # FastAPI endpoint
├── ui/               # Gradio demo
├── notebooks/        # EDA and exploration
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Resume Entry (draft)

**ReadmitAI — Multimodal Clinical Readmission Prediction**
*Python · ClinicalBERT · XGBoost · scispaCy · SHAP · FastAPI · Docker · HuggingFace Spaces*

- Built end-to-end multimodal readmission prediction system on MIMIC-III ICU data (40K+ patients): tabular feature engineering across MIMIC-III tables (diagnoses, labs, medications, demographics) + ClinicalBERT fine-tuned on discharge summaries; late-fusion MLP combining both modalities — **AUROC X.XX** on held-out test set.
- NER pipeline (scispaCy `en_ner_bc5cdr_md`) extracting medical entities (diseases, drugs) and SDoH signals (smoking, alcohol, housing) from unstructured discharge notes; entity counts integrated as tabular features.
- SHAP global/local interpretability for tabular path; LIME text explanations for discharge note path; subgroup analysis across age, gender, and insurance type.
- FastAPI REST endpoint (`POST /predict`) returning readmission probability + top SHAP features + extracted entities; Dockerized deployment; public Gradio demo on HuggingFace Spaces.
