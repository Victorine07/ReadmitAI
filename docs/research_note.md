# When Does Graph Structure Help? Relational Deep Learning for 30-Day Readmission on Real EHR Data

*ReadmitAI research note · Diabetes 130-US Hospitals (1999–2008)*

## Summary

Relational deep learning treats an electronic health record (EHR) as a graph of linked tables rather than one flat
feature matrix. This note asks whether that structure improves 30-day readmission prediction on real hospital data.
On 99,340 inpatient encounters, a heterogeneous graph neural network (GNN) over visits, ICD-9 codes, the ICD-9
hierarchy and medications was compared with logistic regression, XGBoost and an MLP control on identical features.
Splits are by patient, graph edges follow a strict past-only rule, and every comparison uses a paired patient-level
bootstrap.

**The graph did not help.** The GNN matched the MLP on AUROC (Δ −0.001, 95% CI −0.007 to +0.005) and was worse on
AUPRC (Δ −0.011, −0.020 to −0.003). XGBoost remained the best model (AUROC 0.676, AUPRC 0.226). Among the graph's
parts, only past-visit edges gave a measurable gain (+0.004 AUROC), the same size as a single engineered history
column in the flat models. The ICD-9 hierarchy did not improve ranking, even for encounters with rare diagnoses.
The signal in this dataset sits in per-encounter utilisation features, which leaves the relational structure little
to add.

## 1. Motivation

Relational and graph foundation models (relational deep learning, RelBench, graph transformers over databases)
promise to learn directly from linked tables. Hospital records are a natural case: visits link to diagnoses,
diagnoses sit in a coding hierarchy, and patients return. Two concrete hypotheses follow:

1. **Hierarchy helps rare codes.** A flat model sees a rare ICD-9 code as an almost-empty column. A graph can share
   strength through the code's category and chapter.
2. **History helps.** Edges from a patient's earlier visits carry information a single row does not.

A useful study has to be able to show these effects, or show that they are absent. This one was designed with
controls that make either outcome interpretable.

## 2. Data

[Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(Strack et al., 2014): 101,766 de-identified inpatient encounters of diabetic patients from 130 US hospitals, with
demographics, utilisation counts, up to three ICD-9 diagnoses, 23 medication indicators, lab flags and a readmission
label.

- **Label:** readmitted within 30 days (`readmitted == "<30"`).
- **Exclusions:** encounters ending in death or hospice (they cannot be readmitted, so they would be trivially easy
  negatives), and 3 rows with unknown gender. 99,340 encounters remain, from 69,987 patients; 11.39% are positive.
- **Time:** the dataset has no dates, so `encounter_id` is used as a visit-order proxy. It was validated during EDA
  with two checks. First, a patient's last encounter is labelled "not readmitted" far more often than earlier ones,
  as expected if it really is the last visit. Second, prior-year inpatient counts mostly stay the same or increase
  along a patient's encounters.

## 3. Leakage controls

| Risk | Control |
|---|---|
| The same patient in train and test lets a model recognise patients | 70/15/15 split **by patient** with `StratifiedGroupKFold`; asserted that no patient spans two splits and that positive rates match within 0.5 points |
| Preprocessing fit on all data | Every transformer (scaling, one-hot, code vocabulary) is fit on train rows only |
| Future visits informing past ones | History features count only *earlier* encounters; visit edges point only from earlier to later encounters of the same patient |
| Shared nodes mixing patient data | Code, category, chapter and medication nodes never receive messages from encounters, so they act as a shared vocabulary and cannot carry one patient's future back to their past. There is no patient node, and the graph is never made undirected |
| Over-reading the test set | Hyperparameters chosen on validation only; the test set is evaluated once per model |

The graph build asserts these properties: no edge type from encounters into vocabulary nodes; every visit edge
same-patient, forward in time and within one split.

## 4. Methods

**Flat features (630).** Demographics, admission and discharge categories, utilisation counts, medications and lab
flags, multi-hot ICD-9 codes (codes seen ≥ 20 times in train) and a past-only count of earlier encounters.

**Flat models.** Logistic regression (L2, C tuned) and XGBoost (depth and minimum child weight tuned, early stopping
on validation). An XGBoost model without the history feature serves as an ablation. No class weighting was used, to
keep calibrated probabilities.

**Graph.** A PyTorch Geometric `HeteroData` object with 99,340 encounter nodes (240 features: the flat features minus
the code columns), 912 ICD-9 codes, 878 categories, 18 chapters and 21 medications. Edges:

- primary and secondary diagnosis → encounter;
- medication → encounter;
- code ↔ category ↔ chapter;
- earlier → later encounter of the same patient (29,353 edges).

**GNN.** Three `HeteroConv` layers with one GraphSAGE convolution per edge type, so that chapter information can reach
an encounter (chapter → category → code → encounter). Vocabulary nodes have learnable embeddings; encounter
features are clipped to ±5 SD. Full-batch training with binary cross-entropy on training encounters only, Adam, and
early stopping on validation AUROC.

**MLP control.** The same 240 encounter features and output head, with no graph. GNN vs MLP isolates the effect of
the graph; GNN vs XGBoost mixes model family, feature representation and structure.

**Tuning.** GNN and MLP share the same 8-configuration grid (hidden size {64, 128} × dropout {0.2, 0.4} × learning rate
{1e-3, 3e-3}, weight decay 1e-3). Both selected hidden 64, dropout 0.2, learning rate 3e-3. The ablations reuse the
tuned GNN configuration. Each neural model is trained with 5 seeds, and the seeds' test predictions are averaged.

**Evaluation.**
- AUROC, AUPRC (random baseline 0.114) and Brier score (prevalence-only baseline 0.1008).
- 95% CIs from a patient-level cluster bootstrap (1,000 resamples).
- Paired bootstrap CIs for differences between models, using the same resamples for both models.
- Calibration curves, subgroup AUROC by race, sex and age, and rare-code buckets: encounters grouped by how often
  their rarest diagnosis category appears in train.

## 5. Results

Test set: 14,898 encounters from 10,522 patients not seen in training.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | AUROC, rare codes¹ | AUROC, common codes² |
|---|---|---|---|---|---|
| Logistic regression | 0.661 (0.644–0.677) | 0.209 (0.187–0.234) | 0.0971 | 0.686 | 0.658 |
| XGBoost, no history feature | 0.671 (0.655–0.687) | 0.223 (0.200–0.249) | 0.0962 | 0.703 | 0.670 |
| **XGBoost** | **0.676 (0.660–0.692)** | **0.226 (0.205–0.251)** | **0.0960** | 0.701 | 0.669 |
| MLP (no graph) | 0.672 (0.657–0.688) | 0.219 (0.198–0.244) | 0.0963 | 0.710 | 0.665 |
| Hetero-GNN | 0.672 (0.656–0.687) | 0.208 (0.186–0.236) | 0.0969 | 0.681 | 0.665 |
| Hetero-GNN, no hierarchy | 0.670 (0.655–0.686) | 0.204 (0.184–0.230) | 0.0969 | 0.688 | 0.663 |
| Hetero-GNN, no past visits | 0.668 (0.652–0.683) | 0.205 (0.184–0.230) | 0.0970 | 0.680 | 0.659 |

¹ Rarest diagnosis category seen < 20 times in train: 591 encounters, 63 positives, so the uncertainty is wide.
² Rarest category seen > 1,000 times in train: 5,539 encounters.

**Paired differences (95% CI)**

| Comparison | ΔAUROC | ΔAUPRC |
|---|---|---|
| XGBoost − logistic regression | **+0.015 (+0.009, +0.022)** | **+0.017 (+0.009, +0.028)** |
| XGBoost − XGBoost without history | **+0.005 (+0.001, +0.009)** | +0.003 (−0.002, +0.008) |
| GNN − MLP | −0.001 (−0.007, +0.005) | **−0.011 (−0.020, −0.003)** |
| GNN − XGBoost | −0.004 (−0.011, +0.002) | **−0.018 (−0.028, −0.009)** |
| GNN − GNN without hierarchy | +0.001 (−0.001, +0.003) | **+0.004 (+0.001, +0.007)** |
| GNN − GNN without past visits | **+0.004 (+0.002, +0.006)** | +0.003 (−0.001, +0.007) |

**Robustness.**
- Results barely move across tuning configurations. All 8 differ by at most 0.003 validation AUROC for the GNN and
  0.002 for the MLP, about the same as the variation between seeds (std ≈ 0.001).
- No run came near its epoch limit (best epochs: GNN 61–77 of 400; MLP 163–191 of 1,000).

**Calibration and interpretation.**
- All models are well calibrated without class weighting.
- The top SHAP features for XGBoost are prior-year inpatient visits, discharge to home, and the number of earlier
  encounters in the data.

**Subgroups.** AUROC is similar across the subgroups large enough to measure: Caucasian 0.676 vs African American
0.664, women 0.682 vs men 0.667 (XGBoost). Extreme values occur only in groups with fewer than 30 positives.

## 6. Discussion

**Hypothesis 1 (hierarchy helps rare codes) is not supported.** It was already in doubt before the GNN was built:
XGBoost does not perform worse on rare-code encounters (AUROC 0.701 vs 0.669 on common codes). Removing the
hierarchy from the GNN does not change ranking. For codes to help, they have to carry signal, and here they carry
little: in XGBoost's top 20 SHAP features, diagnoses appear only as the primary diagnosis chapter (5th) and one
code (403, hypertensive kidney disease, 14th), well behind utilisation and discharge features.

**Hypothesis 2 (history helps) is supported, but weakly, and a graph is not needed for it.** Past-visit edges add
+0.004 AUROC to the GNN. A single past-only count column adds +0.005 to XGBoost. The graph recovers the history
signal, but it is shallow: only 23.5% of patients have more than one encounter, and prior-year utilisation counts
already summarise most of it.

**Why the GNN is worse on AUPRC than the MLP.** The two models rank encounters equally well overall, but the GNN is
less precise among the highest-risk predictions. A plausible cause is that summing messages from many weak code
neighbours adds noise to the representations of the encounters with the most diagnoses. This was not tested
directly.

**What this says about relational models.** Relational structure pays off when the useful signal is spread across
linked records: dense patient histories, many interacting entity types, and information that cannot be
pre-aggregated into one row. Diabetes 130 is close to the opposite: a single encounter table with weak codes and
sparse histories. This null result therefore narrows down *when* to reach for a graph model; it is not a general
verdict against them.

## 7. Limitations

- **No timestamps.** Visit order relies on the `encounter_id` proxy, and a true temporal split (train on earlier
  years, test on later ones) is not possible.
- **One population.** Diabetic inpatients from one US hospital network, 1999–2008. Coding practice and care have
  changed since.
- **Limited relational content.** At most three diagnoses per visit, no procedures, lab values, notes or transfers.
- **One graph architecture.** GraphSAGE only. A GAT or a relational graph transformer might use the structure better,
  although the flat tuning results and ablations suggest there is little signal left to find.
- **Small rare-code bucket.** 63 positives, so rare-code AUROCs have wide uncertainty.
- Research code only, **not for clinical use**.

## 8. Next steps

- Port the pipeline to **MIMIC-IV**, which has real timestamps, labs, procedures, transfers and discharge notes.
  That allows a temporal split and a much denser relational structure, where the graph has a real chance to help.
- Try attention-based message passing (GAT, relational transformers) and report which relations the model attends to.
- Add bootstrap CIs to subgroup metrics.

## Reproducibility

All code, seeds (42 for splits and flat models; 42–46 for neural models) and commands are in the repository README.
Each script prints its integrity checks and writes metrics to `results/`. Full pipeline runtime on a laptop CPU:
under an hour, most of it GNN training.

## Reference

Strack, B., DeShazo, J. P., Gennings, C., et al. (2014). Impact of HbA1c measurement on hospital readmission rates:
analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, 781670.
