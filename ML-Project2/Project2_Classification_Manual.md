# Project 2 — Classification (SA‑Heart): Statistically Focused Manual

This manual is written for a professional programmer. It tells you **what to compute, what to plot, and why**—with emphasis on the **statistical approach**. Programming details are intentionally light so you can implement in your preferred stack.

---

## 1) Scope & default modeling choices

**Goal.** Predict **CHD** (binary) from the remaining attributes.

**Primary metric.** **Error rate** (misclassification rate):
\[
\text{Error rate}=\frac{\mathrm{FP}+\mathrm{FN}}{N}=1-\text{Accuracy}.
\]
Use a **0.5** decision threshold on predicted probabilities unless you explicitly re‑tune it later.

**Models to compare (exactly three):**
- **Baseline** — predict the **majority class of the *training split***.
- **Logistic regression (L2)** — choose regularization strength \(\lambda\).
- **Method 2: ANN (1 hidden layer)** — choose the number of hidden units \(H\) (must include \(H=1\)).

**Why these.** Baseline is a sanity floor; Logistic is interpretable and well‑calibrated for linear log‑odds; a small ANN adds controlled nonlinearity suitable for tabular data of this size.

---

## 2) Data audit & plots before modeling

Focus on **insightful plots** that inform later modeling and evaluation decisions.

### 2.1 Class balance & label sanity
- **Compute/Report:** counts per class; majority‑class rate.
- **Plot:** bar chart of class counts.
- **Why:** confirms choice of accuracy/error rate and flags potential imbalance (in which case you’ll add ROC–AUC in an appendix).

### 2.2 Marginals & correlations
- **Numeric features:** histograms (or KDEs) **stratified by class**.
- **Categoricals (e.g., `famhist`):** stacked bars by class.
- **Correlation heatmap** of numeric features.
- **Why:** reveals separability, skew, and redundancy (collinearity) that affect capacity/regularization choices.

### 2.3 Missingness & outliers
- **Compute/Report:** missing counts per feature; basic outlier check (boxplots).
- **Why:** any imputation/transform **must be fitted inside CV** to avoid leakage (see §4).

---

## 3) Primary metric, baseline, and sanity plots

### 3.1 Define the metric clearly
- Use **error rate** on each **outer test fold** (see §4). Also show a **confusion matrix** for at least one fold to interpret FP/FN types.

### 3.2 Baseline model
- **Train** baseline on the *training* portion only (use that split’s majority class).
- **Evaluate** on the matching test portion.
- **Plot:** one **confusion matrix** labeled “Baseline (train‑majority)”.

---

## 4) Two‑level (nested) cross‑validation: selection **and** estimation

You must separate **model selection** (inner loop) from **performance estimation** (outer loop) to avoid optimistic bias that occurs when selecting and assessing on the same resamples.

### 4.1 Design
- **Outer CV:** \(K_1=10\) **stratified** folds (recommended). **Reuse the identical outer splits** for all three models.
- **Inner CV:** \(K_2=10\) **stratified** folds for hyperparameter tuning:
  - Logistic: pick \(\lambda^*\) from a **log‑spaced** grid (e.g., \(10^{-4}\) to \(10^{2}\)).
  - ANN: pick \(H^*\) from a **small** grid (e.g., \(H\in\{1,2,4,8,16,32\}\)).
- **Preprocessing inside CV only:** standardize numerics; one‑hot encode categoricals; any imputation/transform is **fitted on the training part of each fold** and applied to its validation/test part.

### 4.2 Per outer fold \(i\) record
- \(H_i^*\) and **outer test error** \(E^{\text{test}}_{\text{ANN},i}\).
- \(\lambda_i^*\) and **outer test error** \(E^{\text{test}}_{\text{LOG},i}\).
- **Outer test error** \(E^{\text{test}}_{\text{BASE},i}\).
- (Optional) \(|D^{\text{test}}_i|\) if you later compute weighted averages.

### 4.3 Plots to produce here
1) **Inner‑CV profiles**
   - Logistic: mean validation error \( \pm \) one SE **vs** \(\log_{10}\lambda\).
   - ANN: mean validation error \( \pm \) one SE **vs** \(H\).
   - **Why:** shows stability of the minimum and supports the chosen capacity.
2) **Outer per‑fold errors**
   - **Lines/dots** across outer fold index for each model’s **test error**.
   - **Optional boxplot** comparing per‑fold error distributions.
   - **Why:** visualizes variability that the formal tests (next section) summarize.

> Interpretation note: Two‑level CV evaluates the **select‑then‑refit procedure**; the outer‑fold mean error is an estimate of that procedure’s generalization error.

---

## 5) Statistical comparison — **Use Setup II** (training set is random)

Because outer folds change the training data, comparisons across models are **Setup II** problems. Use the **correlated t‑test** with the **correlation heuristic** \( \rho \approx 1/K \) (where \(K\) is the number of folds used to generate the paired results—here, the outer CV folds).

### 5.1 What to compute for each pair of models \(A,B\)
1) For each outer split \(j=1,\ldots,J\) (normally \(J=K_1\); you can repeat CV to increase \(J\)), compute the **paired difference in error**:
\[
r_j \;=\; \frac{1}{n_j}\sum_{i=1}^{n_j}\big[ \mathbb{1}\{\hat y^{A}_{j,i}\neq y_{j,i}\} - \mathbb{1}\{\hat y^{B}_{j,i}\neq y_{j,i}\} \big].
\]
2) Model \( r_j = \bar z + v_j \) with correlated noise \(v\) (equal variance, constant correlation). Use the **heuristic** \( \rho = 1/K \) to inflate the SE.
3) Report for each pair:
   - **Mean difference** \( \hat r = \frac{1}{J}\sum_j r_j \) (positive means \(A\) has **higher** error than \(B\)).
   - **95% CI** for \( \bar z \) using the Student‑t with correlation‑inflated SE.
   - **p‑value** from the same t‑statistic.

> Run **three pairwise contrasts** on the **same** set of outer folds: ANN vs Logistic, ANN vs Baseline, Logistic vs Baseline.

### 5.2 Plots for inference
- **Distribution of \(r_j\)** (histogram or dot plot) with the **95% CI** whisker drawn for \(\bar z\).
- **Paired fold scatter:** \(E^{\text{test}}_A\) vs \(E^{\text{test}}_B\) with the 45° line; annotate mean difference.
- **(Optional) Q–Q plot** of \(\{r_j\}\) to assess approximate normality.

### 5.3 When to use Setup I / McNemar instead
Only when you keep a **single fixed training set** (e.g., a single hold‑out) and obtain **paired predictions on the same test items**. Then use **McNemar** on discordant counts \(n_{12}\) (A correct, B wrong) and \(n_{21}\) (A wrong, B correct):
- **Statistic:** exact two‑sided binomial test under \(n_{12}\sim\text{Binom}(n_{12}+n_{21}, 0.5)\).
- **Effect & CI:** \(\Delta_{\text{acc}}=\frac{n_{12}-n_{21}}{n_{12}+n_{21}}\) with a Beta‑based **95% CI**.
- **Plot:** 2×2 table with counts; bar of \(n_{12}\) vs \(n_{21}\).
(*Do not* use McNemar to compare across folds with changing training sets—use Setup II.)

---

## 6) Results table (what the marker expects)

A single table with \(K_1\) rows (outer folds), containing for each fold \(i\):
- \(H_i^*\), \(E^{\text{test}}_{\text{ANN},i}\);
- \(\lambda_i^*\), \(E^{\text{test}}_{\text{LOG},i}\);
- \(E^{\text{test}}_{\text{BASE},i}\).

Add a final row with each model’s **mean outer‑test error** (unweighted or weighted by test size).

**Companion plots:**
- **Bar + error bars** (mean ± SE) of outer‑test error by model.
- **Per‑fold lines** (three series) across fold index to visualize covariance between models across folds.

---

## 7) Learning‑curve diagnostics (optional but persuasive)

Produce **learning curves** (training error and **outer** test error vs training size) for Logistic and ANN:
- **Why:** diagnoses under/overfit and method adequacy; if curves converge high, the bias is structural (need features/capacity). If the gap is large, you may be variance‑limited (regularization/data needed).
- **Plot:** For each model, two lines (train vs outer‑test error) as the effective training size increases. Overlay the **baseline** as a reference and, if desired, a “target band”.

---

## 8) Class‑imbalance appendix (only if skew is notable)

Keep **error rate** as the primary metric, but add **ROC–AUC** and ROC curves to the appendix:
- **What:** Compute ROC curves and **AUC** on **pooled outer test predictions** (concatenate across folds).
- **Plot:** ROC curves for Logistic and ANN; annotate AUC. Briefly note threshold dependence vs AUC.

---

## 9) Final interpretability (optional, clearly labeled)

Nested CV evaluates the **procedure**, not a single frozen model. For interpretability only:
- **Refit Logistic on all data** with a “reasonable” \(\lambda\) (e.g., the **median** of \(\{\lambda_i^*\}\)).
- **Plot:** horizontal bar chart of **|standardized coefficients|** (top \(k\)).
- **Caution:** performance claims **must** come from the nested‑CV outer‑fold estimates; this refit is for explanation only.

---

## Deliverables checklist (plots & stats)

**Data & sanity**
- Class balance bar chart; missingness table; marginals by class; correlation heatmap.

**Model selection (inner CV)**
- Logistic: validation error vs \(\log_{10}\lambda\) (mean ± SE).
- ANN: validation error vs \(H\) (mean ± SE).

**Generalization estimation (outer CV)**
- Per‑fold outer test errors (lines/dots) for all models.
- Mean ± SE bar chart of outer‑test errors.

**Statistical comparison**
- **Setup II** correlated t‑test: \(r_j\) distribution + 95% CI + p‑value for the three pairs; paired fold scatter vs 45° line.
- **(Alternative)** Setup I/McNemar (only if fixed train set): 2×2 table, discordant counts bar, \(\Delta_{\text{acc}}\) with CI, p‑value.

**Diagnostics**
- Learning curves (train vs outer‑test error vs training size) for Logistic and ANN; include baseline reference.
- (If skew) ROC curves + AUC (appendix).

**Interpretability (optional)**
- Final Logistic **|coef|** bar plot, clearly labeled as explanatory; all performance claims cite nested‑CV means/CIs.

---

## One‑paragraph Methods text (ready to paste)

We compared a baseline (train‑majority), L2‑regularized logistic regression, and a one‑hidden‑layer ANN (capacity \(H\)) on the SA‑Heart dataset. Preprocessing (standardization/one‑hot) was fitted **inside** resampling only. Model selection used **two‑level cross‑validation**: \(K_2=10\) inner folds tuned \(\lambda\) and \(H\); \(K_1=10\) outer folds estimated generalization for the select‑then‑refit procedures. We reported **outer‑test error rates** per fold and means. Statistical comparisons used **Setup II** (training set random): for each outer split we formed paired error differences \(r_j\) and applied a **correlated t‑test** with the **\(\rho=1/K\)** heuristic to report 95% CIs and p‑values (three pairwise contrasts). For completeness we show inner‑CV selection profiles, per‑fold error plots, and (if class skew) ROC–AUC in the appendix; confusion matrices accompany main results. Performance claims come from nested‑CV; a final logistic refit (with a central \(\lambda\)) is shown only for interpretability.
