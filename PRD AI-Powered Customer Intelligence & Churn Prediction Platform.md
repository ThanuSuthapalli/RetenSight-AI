# PRD: AI-Powered Customer Intelligence & Churn Prediction Platform

Product Requirements Document · Version 1.0 · Prepared for Thanu

## 1. What Is This Project? (Plain-Language Explanation)

**Customer churn** means a customer stops using a service: a telecom user cancels their SIM plan, a subscriber ends a streaming subscription, a bank customer closes an account. Acquiring a new customer typically costs far more than keeping an existing one, so companies badly want to know **who is about to leave, and why**, early enough to do something about it.

This project is a web platform that answers exactly that, using machine learning:

1. A business user uploads a CSV of customer data (age, plan type, monthly charges, contract length, support calls, and so on) that includes a column showing whether each past customer left.
2. The platform cleans the data automatically (missing values, text categories, scaling).
3. It trains several ML models (Logistic Regression, Random Forest, XGBoost), tunes them, and compares them fairly using cross-validation.
4. It picks the best model and lets the user enter or upload a customer to get a **churn probability** (e.g. "82% likely to leave").
5. Using **SHAP**, it explains *why*: "Month-to-month contract and high support-call count are pushing this customer toward churn."
6. It turns the findings into **business insights and charts** so a non-technical manager can decide what to do (offer a discount, push annual contracts, fix a support problem).

**Think of it as:** a mini "Data Scientist in a box" with a manager-friendly dashboard on top.

### Why it is a strong resume project

It covers the **complete ML lifecycle** in one product: data ingestion → cleaning → feature engineering → model training → tuning → evaluation → explainability → deployment as an app → business communication. Most student projects stop at a notebook; this ships a usable tool.

### The ML lifecycle this project maps to

| Stage | What happens here |
| --- | --- |
| Data collection | User uploads CSV |
| Data cleaning | Handle missing values, duplicates, wrong types |
| Feature engineering | Create tenure groups, charge ratios, interaction features |
| Encoding & scaling | One-hot / ordinal encoding, standard scaling |
| Modeling | Logistic Regression, Random Forest, XGBoost |
| Tuning & validation | GridSearch/RandomizedSearch + stratified k-fold CV |
| Evaluation | Precision, Recall, F1, ROC-AUC, confusion matrix |
| Explainability | SHAP global and per-customer explanations |
| Delivery | Streamlit dashboard + business insights |

## 2. Product Overview

### 2.1 Problem Statement

Businesses lose revenue when customers leave unexpectedly. Retention teams usually lack data-science skills to build, compare, and interpret churn models, and black-box predictions are not trusted without explanations. There is a need for a self-serve tool that takes raw customer data and produces accurate, explainable churn predictions and actionable recommendations.

### 2.2 Vision

A no-code, explainable churn intelligence platform where any analyst can go from raw CSV to ranked at-risk customers and clear retention actions in under 10 minutes.

### 2.3 Goals

- G1: Automate end-to-end preprocessing for typical tabular customer datasets.
- G2: Train and fairly compare at least three classification models.
- G3: Deliver per-customer churn probability with human-readable reasons.
- G4: Surface global churn drivers and business insights through visualizations.
- G5: Demonstrate best-practice ML engineering (no data leakage, reproducibility, proper validation).

### 2.4 Non-Goals (Version 1)

- Real-time streaming predictions or live CRM integration.
- Deep learning models or time-series survival analysis.
- Multi-user authentication, billing, or multi-tenant architecture.
- Automated sending of retention emails or offers.

### 2.5 Success Metrics

| Metric | Target |
| --- | --- |
| Best-model ROC-AUC on reference dataset (Telco Churn) | ≥ 0.84 |
| Best-model Recall on churn class (at tuned threshold) | ≥ 0.75 |
| End-to-end run (upload → trained models) on 7K rows | < 2 minutes |
| Prediction + SHAP explanation latency (single customer) | < 3 seconds |
| Works on a new, unseen CSV without code changes | Yes, with column mapping |
| Unit test coverage on preprocessing and modeling modules | ≥ 70% |

## 3. Users and Personas

**Persona 1: Retention Manager (primary).** Non-technical. Wants a list of at-risk customers and clear reasons so the team can act. Cares about dashboards, simple language, and exportable lists.

**Persona 2: Data Analyst (secondary).** Wants to compare models, inspect metrics, tweak the decision threshold, and download results.

**Persona 3: Recruiter / Interviewer (portfolio audience).** Wants to quickly see ML depth: validation approach, metrics, explainability, clean architecture.

## 4. User Stories

| ID | As a... | I want to... | So that... |
| --- | --- | --- | --- |
| US-1 | Analyst | upload a CSV and see a data summary | I know the data is read correctly |
| US-2 | Analyst | choose the target (churn) column | the tool works with any dataset |
| US-3 | Analyst | have missing values and categories handled automatically | I skip manual cleaning |
| US-4 | Analyst | train multiple models with one click | I can compare them without coding |
| US-5 | Analyst | see a metrics comparison table and ROC curves | I can pick the best model |
| US-6 | Manager | enter one customer's details and get churn probability | I can assess a specific account |
| US-7 | Manager | see why a customer is predicted to churn | I trust the prediction and know what to fix |
| US-8 | Manager | see the top churn drivers across all customers | I can plan company-wide retention |
| US-9 | Manager | download a ranked list of high-risk customers | my team can contact them |
| US-10 | Manager | read auto-generated business insights | I get recommendations in plain English |

## 5. Functional Requirements

### FR-1: Dataset Upload

- Accept CSV (required) and XLSX (optional) up to 50 MB.
- Preview first 100 rows, row/column counts, data types, and memory size.
- Validate: file is readable, has at least 500 rows, at least 5 columns, and a binary target column exists.
- User selects the target column and the positive class label (e.g. "Yes" = churned). Optionally selects an ID column to exclude from training.
- Provide a bundled sample dataset (Telco Customer Churn) so the app can be demoed with one click.

### FR-2: Automated Data Preprocessing

- Drop exact duplicate rows and the ID column from features.
- Fix data types (e.g. numeric columns stored as text with blank strings, as in Telco's TotalCharges).
- Detect column types automatically: numeric, categorical, binary, high-cardinality text, datetime.
- Remove or flag constant columns and columns with more than 60% missing values (configurable).
- Show a preprocessing report: what was changed, how many values imputed, which columns dropped.

### FR-3: Missing Values and Categorical Variables

| Case | Default strategy | Options |
| --- | --- | --- |
| Numeric missing | Median imputation | Mean, constant, KNN |
| Categorical missing | Most frequent / "Unknown" | Constant |
| Low-cardinality categorical | One-hot encoding | Ordinal |
| High-cardinality categorical (> 20 levels) | Frequency or target-safe encoding | Drop |
| Binary Yes/No | Map to 0/1 | n/a |
| Numeric scaling | StandardScaler (for Logistic Regression) | MinMax, none for trees |
| Outliers | Report via IQR; optional capping | Off |

All steps must live inside a scikit-learn **Pipeline / ColumnTransformer** so that imputers, encoders, and scalers are fit only on training folds. This prevents data leakage.

### FR-4: Feature Engineering

Generated automatically when relevant columns exist, and togglable by the user:

- Tenure buckets (0-12, 13-24, 25-48, 49+ months).
- Average monthly spend = total charges / tenure.
- Count of subscribed add-on services.
- Charge-to-tenure ratio and monthly-vs-average charge delta.
- Contract-type × payment-method interaction flags.
- Support-interaction rate (calls / tenure) where available.

The UI shows a before/after feature count and lets the user compare model performance with and without engineered features.

### FR-5: Train Multiple Models

- Models: **Logistic Regression**, **Random Forest**, **XGBoost** (v1 required); optional Gradient Boosting or LightGBM.
- Train/test split: 80/20 stratified, with a fixed random seed.
- **Class imbalance handling** (churn is usually 15-30% of rows): class weights (default), optional SMOTE applied inside the training fold only. The user can choose.
- Hyperparameter tuning: RandomizedSearchCV or GridSearchCV with stratified 5-fold cross-validation. Scoring metric selectable (ROC-AUC default, F1, recall).
- Training runs with a progress indicator and a configurable time budget ("Quick" vs "Thorough" mode).
- Trained models and the preprocessing pipeline are saved with joblib in a versioned folder.

Suggested tuning search spaces:

| Model | Parameters |
| --- | --- |
| Logistic Regression | C, penalty (l1/l2), solver, class\_weight |
| Random Forest | n\_estimators, max\_depth, min\_samples\_split, min\_samples\_leaf, max\_features |
| XGBoost | n\_estimators, learning\_rate, max\_depth, subsample, colsample\_bytree, scale\_pos\_weight, reg\_lambda |

### FR-6: Compare Model Performance

- Comparison table: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, training time, CV mean ± std.
- Charts: overlaid ROC curves, precision-recall curves, confusion matrix per model, bar chart of metrics.
- Highlight the "best model" with a stated reason (e.g. highest ROC-AUC with recall above the floor).
- Adjustable decision threshold slider (default 0.5) that live-updates precision, recall, and confusion matrix, so users can trade off false alarms against missed churners.
- Plain-language metric tooltips ("Recall = of all customers who actually left, how many we caught").

### FR-7: Predict Individual Customer Churn

- Form auto-generated from the training schema with correct input types (dropdowns for categories, number inputs with ranges).
- Output: churn probability as a percentage, risk band (Low < 30%, Medium 30-60%, High > 60%, configurable), and predicted label at the chosen threshold.
- Batch prediction: upload a new CSV and download scored results sorted by risk.
- Choose which trained model to use (default: best).

### FR-8: Display Important Churn Factors (Explainability)

- **Global:** SHAP summary (beeswarm) plot and mean-|SHAP| bar plot for the best model. Fallback to coefficients (Logistic Regression) and feature importances (tree models) for comparison.
- **Local:** SHAP waterfall or force plot for a single customer, followed by a plain-English sentence such as "Month-to-month contract (+0.31) and no tech support (+0.18) increase churn risk; 3-year tenure (-0.22) reduces it."
- **Dependence plots** for the top 3-5 features.
- Cache SHAP values to keep the UI fast; use TreeExplainer for tree models and LinearExplainer for Logistic Regression; sample at most 2,000 rows for global plots.

### FR-9: Business Insights and Visualizations

- Churn rate overall and by segment (contract, tenure bucket, payment method, region, etc.) using Seaborn/Matplotlib charts.
- Correlation heatmap and distribution plots of numeric features by churn status.
- **Auto-generated insight cards**, e.g. "Customers on month-to-month contracts churn at 42%, vs 3% on two-year contracts."
- **Revenue at risk** estimate: sum of monthly revenue for customers predicted high risk (when a revenue column exists).
- **Retention recommendations** mapped from top SHAP drivers to suggested actions (rule-based templates in v1).
- Export: CSV of at-risk customers, PNG of charts, and a downloadable PDF or HTML summary report (stretch).

## 6. Machine Learning Requirements

### 6.1 Evaluation Protocol

1. Stratified train/test split; test set is never touched during tuning.
2. Stratified 5-fold CV on the training set for hyperparameter search and for reporting CV mean ± std.
3. Final metrics reported on the held-out test set.
4. Primary metric: ROC-AUC. Business metric: recall at an acceptable precision, because missing a churner is usually costlier than a false alarm.
5. Calibration check (reliability curve) so that "80%" really means about 80%; optional CalibratedClassifierCV.

### 6.2 Anti-Leakage Rules

- Fit all transformers inside the pipeline on training folds only.
- Reject or warn on columns that obviously leak the target (e.g. a "cancellation date" column).
- Resampling (SMOTE) only inside CV folds.

### 6.3 Reproducibility

- Global random seed, pinned dependency versions, and saved run metadata (dataset hash, parameters, metrics, timestamp) in a JSON file per experiment.

## 7. Dashboard Design (Streamlit)

Multi-page app with a sidebar for navigation and a persistent "current dataset / model" status chip.

| Page | Contents |
| --- | --- |
| 1. Home / Upload | Project intro, sample data button, upload widget, target selection, data preview |
| 2. Data Explorer (EDA) | Summary stats, missing value chart, distributions, correlation heatmap, churn by segment |
| 3. Preprocessing | Auto-detected column types, chosen strategies (editable), feature engineering toggles, preprocessing report |
| 4. Model Training | Model checkboxes, tuning mode, imbalance option, Train button, progress, saved-model list |
| 5. Model Comparison | Metrics table, ROC/PR curves, confusion matrices, threshold slider, best-model badge |
| 6. Explainability | Global SHAP plots, feature importance comparison, dependence plots |
| 7. Predict | Single-customer form, probability gauge, local SHAP explanation, batch scoring upload |
| 8. Business Insights | Insight cards, segment charts, revenue at risk, recommendations, exports |

**UX principles:** every chart gets a one-line takeaway; technical terms have tooltips; loading states for long operations (`st.spinner`, `st.progress`); state kept in `st.session_state`; heavy work cached with `st.cache_data` and `st.cache_resource`; consistent color coding (red = high risk, amber = medium, green = low).

Since design is a strength of yours, treat the dashboard polish (custom theme in `.streamlit/config.toml`, clean card layouts, consistent palette) as a differentiator versus typical notebook-style ML projects.

## 8. System Architecture

```
[Streamlit UI]
    |
    v
[App layer: pages, session state, caching]
    |
    +--> [data/ loader, validator, profiler]
    +--> [preprocessing/ pipeline builder, feature engineering]
    +--> [models/ trainer, tuner, evaluator, registry (joblib)]
    +--> [explain/ SHAP wrappers, insight generator]
    +--> [viz/ Matplotlib and Seaborn chart helpers]
    |
    v
[Local storage: uploads/, artifacts/models/, artifacts/reports/]
```

### 8.1 Suggested Project Structure

```
churn-platform/
├── app/
│   ├── Home.py
│   └── pages/ (1_Explorer.py ... 8_Insights.py)
├── src/
│   ├── data/ (loader.py, validator.py)
│   ├── preprocessing/ (pipeline.py, features.py)
│   ├── models/ (train.py, tune.py, evaluate.py, registry.py)
│   ├── explain/ (shap_utils.py, insights.py)
│   └── viz/ (plots.py)
├── notebooks/ (EDA and experiments)
├── tests/
├── data/sample/ (telco_churn.csv)
├── artifacts/
├── .streamlit/config.toml
├── requirements.txt
├── Dockerfile
└── README.md
```

### 8.2 Technology Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, imbalanced-learn (optional) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn (Plotly optional for interactivity) |
| UI | Streamlit |
| Persistence | joblib, JSON, optional SQLite for experiment log |
| Quality | pytest, Ruff/Black, pre-commit |
| Deployment | Streamlit Community Cloud, Hugging Face Spaces, or Docker on Render |

## 9. Data Requirements

### 9.1 Reference Datasets

- **Telco Customer Churn (IBM / Kaggle):** \~7,043 rows, 21 columns, \~26% churn. Ideal primary dataset. It has real-world quirks (blank TotalCharges, many categorical columns).
- **Bank Customer Churn (Kaggle):** \~10,000 rows, used to prove the platform generalizes to a different domain.
- Optional: a synthetic SaaS-subscription dataset you generate yourself to show data-generation skills.

### 9.2 Expected Input Schema (flexible)

| Column type | Example | Required |
| --- | --- | --- |
| Customer ID | customerID | Optional |
| Target (binary) | Churn = Yes/No | Required |
| Numeric | tenure, MonthlyCharges | At least 1 |
| Categorical | Contract, PaymentMethod | Recommended |
| Revenue column | MonthlyCharges | Optional (for revenue-at-risk) |

### 9.3 Data Privacy

Uploaded files are processed locally or in-session only, not shared, and are deleted when the session ends. The README must advise using anonymized data and avoiding personal identifiers.

## 10. Non-Functional Requirements

| Area | Requirement |
| --- | --- |
| Performance | Quick-mode training < 2 min on 10K rows; page loads < 3 s with caching |
| Reliability | Graceful errors for malformed files, wrong target column, single-class targets |
| Usability | A non-technical user completes the flow without documentation |
| Maintainability | Modular code, type hints, docstrings, separation of UI and ML logic |
| Testability | Unit tests for preprocessing, pipeline leakage, metric computation, and prediction schema |
| Portability | One-command setup via requirements.txt and Docker |
| Security | File type and size validation; no execution of uploaded content; no pickle loading from untrusted uploads |
| Accessibility | Colorblind-safe palettes; text labels alongside colors |

## 11. Development Plan (4-6 Weeks)

| Phase | Duration | Deliverables |
| --- | --- | --- |
| 1. Foundations & EDA | Week 1 | Repo setup, sample data, notebook EDA, churn insights, baseline Logistic Regression |
| 2. Preprocessing & Features | Week 2 | ColumnTransformer pipeline, auto type detection, feature engineering, unit tests |
| 3. Modeling & Tuning | Week 3 | Three models, CV tuning, imbalance handling, evaluation module, model registry |
| 4. Explainability | Week 4 | SHAP global/local, plain-English explanations, insight generator |
| 5. Streamlit App | Week 5 | All 8 pages wired up, session state, caching, theming |
| 6. Polish & Deploy | Week 6 | Tests, README with screenshots/GIF, Docker, deployment, demo video |

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Data leakage inflates metrics | Misleading results, interview red flag | Pipeline-only preprocessing, CV inside training set, leakage warnings |
| Class imbalance hurts recall | Misses churners | Class weights, threshold tuning, PR-AUC reporting |
| SHAP is slow on large data | Poor UX | Sampling, caching, TreeExplainer |
| Arbitrary user CSVs break the pipeline | Crashes | Schema validation, type detection, clear error messages |
| Tuning takes too long in-browser | Frustration | Quick/Thorough modes, RandomizedSearch, n\_jobs=-1 |
| Overfitting | Poor real-world accuracy | CV, regularization, held-out test set |
| Scope creep | Delays | Strict v1 non-goals; stretch features listed separately |

## 13. Acceptance Criteria (Definition of Done)

- User can upload the Telco CSV and reach a trained model without writing code.
- Preprocessing report lists every transformation applied.
- Three models are trained with CV-tuned hyperparameters and compared on a test set.
- Best model reaches ROC-AUC ≥ 0.84 on Telco (or the README explains the achieved value honestly).
- Single-customer prediction shows probability, risk band, and a SHAP-based explanation.
- Global churn factors and at least 5 auto-generated business insights are displayed.
- At-risk customer list can be downloaded as CSV.
- App is deployed publicly, and README includes architecture diagram, screenshots, setup steps, and metric tables.
- Tests pass and code has no data leakage between train and test.

## 14. Stretch Goals (Version 2)

- Customer Lifetime Value and survival analysis (Kaplan-Meier, Cox) for time-to-churn.
- Customer segmentation with K-Means to build retention personas.
- What-if simulator: change contract type or price and see churn probability move.
- MLflow experiment tracking and model versioning.
- FastAPI prediction endpoint with a separate frontend.
- Drift monitoring and scheduled retraining.
- LLM-generated retention email suggestions based on SHAP drivers.
- Fairness checks across demographic groups.

## 15. Resume and Interview Value

### Sample resume bullets

- Built an end-to-end churn prediction platform in Python (Pandas, Scikit-learn, XGBoost) with automated preprocessing, feature engineering, and imbalance handling, reaching \~0.85 ROC-AUC on 7K-customer telecom data.
- Tuned and compared Logistic Regression, Random Forest, and XGBoost using stratified 5-fold CV and RandomizedSearch, evaluating with ROC-AUC, PR-AUC, recall, and F1.
- Integrated SHAP to deliver global and per-customer explanations, converting model outputs into plain-language retention recommendations for business users.
- Delivered a multi-page Streamlit dashboard supporting dataset upload, model comparison, threshold tuning, batch scoring, and exportable at-risk customer lists.

(Replace metrics with your actual results.)

### Questions you should be ready to answer

- Why recall and PR-AUC instead of accuracy for churn?
- How did you prevent data leakage?
- Why did XGBoost (or Logistic Regression) win, and what is the trade-off between accuracy and interpretability?
- What does a SHAP value mean, and how does it differ from feature importance?
- How did you choose the decision threshold, and what would it cost the business?
- How would you deploy and monitor this in production?

## 16. Glossary

- **Churn:** a customer ceasing to use the service.
- **Precision:** of customers predicted to churn, the share who truly churn.
- **Recall:** of customers who truly churn, the share the model catches.
- **ROC-AUC:** how well the model ranks churners above non-churners (1.0 perfect, 0.5 random).
- **Cross-validation:** repeated train/validate splits to estimate stable performance.
- **SHAP:** a method that attributes each prediction to the contribution of each feature.
- **Data leakage:** information from outside the training data (or from the test set) leaking into training, producing falsely high scores.
