# Sri Lanka Paddy Yield Prediction & Analytics Dashboard 🌾

An end-to-end machine learning and data analytics project for exploring historical paddy yields across **25 districts in Sri Lanka** and estimating district-level yield using a trained **Linear Regression** model. The project includes a public, interactive **Streamlit** dashboard for islandwide summaries, district rankings, comparisons, retrospective model performance, and experimental yield predictions.

**Live dashboard:** [Sri Lanka Paddy Yield Prediction](https://sri-lanka-paddy-yield-prediction.streamlit.app/)

> **Research and educational prototype:** Predictions are estimates, not agricultural advice or official forecasts. Forecast-date availability of historical yield and cultivated-area inputs has not been independently verified.

## Features

- **Islandwide overview:** Explore long-term trends, Maha vs Yala seasons, sown extent, and coverage across districts.
- **District rankings:** Compare recorded district yields for a selected year and cultivation season.
- **District comparison:** View historical yield trends for two districts side by side.
- **Model performance:** Inspect actual vs predicted yields, MAE/RMSE/R², and yearly prediction errors on the 2020–2026 evaluation period.
- **Yield prediction:** Select a district, season, year, and sown extent. The application automatically derives available lag features from historical district records and passes them through the saved preprocessing and model pipeline.

**Interpretation note:** The dashboard's “islandwide average” is the **unweighted mean of observed district-level yields**, *not* the official production-weighted national average yield. Sown-extent summaries likewise reflect the observations included in the dataset.

## Data

Historical paddy statistics were obtained from workbooks published by the **Department of Census and Statistics (DCS), Sri Lanka**. The preprocessing workflow extracted and standardized district, season, year, average yield, and cultivation-area information. The combined analytical dataset contains **2,274 district-season observations** across 25 districts.

Main fields:

| Field | Description |
| --- | --- |
| `district` | Sri Lankan administrative district |
| `season` | Maha or Yala cultivation season |
| `year` | Original season label, e.g. `2021/2022` or `2022` |
| `season_year` | Ending year for Maha; calendar year for Yala |
| `average_yield_kg_ha` | Observed paddy yield (model target), in kg/ha |
| `sown_extent_ha` | Recorded sown area in hectares |
| `season_gap` | Number of seasons since the last *observed* district yield |
| `previous_yield` | Immediately preceding season's yield if its record exists |
| `yield_3season_avg` | Average yield over three **consecutive** preceding seasons, if available |

To ensure temporal ordering, the project represents each cultivation season with a sequential season index and checks for missing intervening seasons. In the V2 features:

- 2,178 observations have a recorded immediately preceding season.
- 71 historical transitions have gaps greater than one season.
- `previous_yield` has 96 missing values, and `yield_3season_avg` has 188 missing values; the fitted preprocessing pipeline imputes missing numerical features.

**Data-quality considerations:** The source workbooks required header normalization, year-label cleanup, and district-name standardization. Some production and area measurements were inconsistent across source files, so the initial prediction model does **not** use production or harvested extent as predictors. Source publication dates and redistribution terms should be checked before reusing or republishing raw files.

## Methodology

1. **Collect and clean:** Extract official Excel workbooks, standardize district names and seasons, and create district-level analytical records.
2. **Explore:** Analyze yield distributions, district differences, Maha/Yala trends, correlations, and historical changes using Pandas, Matplotlib, and Seaborn.
3. **Engineer historical features:** Calculate lagged yields, three-season averages, and missing-season gaps without using target-season yields as inputs.
4. **Preprocess:** One-hot encode `district`/`season`; impute missing numerical features; scale numerical inputs for Linear Regression.
5. **Train and compare:** Evaluate a previous-yield baseline, Linear Regression, and Random Forest; tune Random Forest against a pre-2020 validation period.
6. **Analyze failures:** Compare errors by district, season, and year, especially the unusually high errors in 2022.
7. **Serve predictions:** Save the fitted Linear Regression V2 pipeline with `joblib` and load it in Streamlit.

### Chronological evaluation

| Partition | Seasons ending in | V2 observations |
| --- | --- | ---: |
| Training | 1979–2014 | 1,710 |
| Validation | 2015–2019 | 245 |
| Exploratory test period | 2020–2026 | 319 |

For the final V2 comparison, training and validation data were combined (**1,955 observations**) to fit models for evaluation on 2020–2026. These experiments are **rolling one-season-ahead retrospective evaluations** using actual historical lag values; they are not forecasts generated once in 2019 for the entire following period.

### Model results

**Exploratory test period: 2020–2026** (lower MAE/RMSE and higher R² are better):

| Model | MAE (kg/ha) | RMSE (kg/ha) | R² |
| --- | ---: | ---: | ---: |
| Previous-yield baseline (V1) | 677.90 | 903.50 | 0.0971 |
| Linear Regression V1 | **572.91** | **747.50** | **0.3820** |
| Linear Regression V2 | 577.59 | 749.49 | 0.3787 |
| Tuned Random Forest V1 | 620.84 | 829.88 | 0.2383 |
| Random Forest V2 | 616.05 | 824.87 | 0.2475 |

The dashboard uses **Linear Regression V2**, despite V1's marginally lower test error, because V2 explicitly handles gaps between observed cultivation seasons and provides clearer historical-feature semantics. V1/V2 have differences in usable training observations, so their scores are not a perfectly controlled comparison. The test period was inspected during development and is **not a completely untouched final holdout**.

**Validation period (2015–2019):** Linear Regression V2 achieved **MAE 409.69 kg/ha, RMSE 531.04 kg/ha, R² 0.5681**; Random Forest V2 achieved **MAE 396.33 kg/ha, RMSE 519.78 kg/ha, R² 0.5862**. Validation performance did not reliably translate into improved later-period performance.

## Case study: The 2022 agricultural disruption

Error analysis found particularly large overpredictions around the **2021/2022 Maha** season, coinciding with Sri Lanka's fertilizer restrictions, agricultural input shortages, and economic crisis. This temporal association is consistent with an external shock, but the model results **do not establish causation**.

For **Linear Regression V2**:

| Evaluation subset | Observations | MAE (kg/ha) | RMSE (kg/ha) | R² |
| --- | ---: | ---: | ---: | ---: |
| 2022 only | 49 | 1,254.06 | 1,353.42 | -3.7183 |
| Excluding 2022 | 270 | 454.82 | 575.55 | 0.6026 |
| Full test period | 319 | 577.59 | 749.49 | 0.3787 |

Although 2022 makes up approximately **15.36%** of test observations, it accounts for **33.35%** of total absolute prediction error. Excluding 2022 is a **diagnostic analysis only**, not the project's headline performance measure. A negative 2022 R² indicates performance worse than predicting the mean of the 2022 observations.

Potential future enhancements include rainfall data, fertilizer availability or price information, and more robust evaluation of distribution shifts—provided these inputs are known at the intended forecast date.

## Project structure

```text
.
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
├── models/
│   └── paddy_yield_linear_v2.joblib   # Saved fitted pipeline
├── data/
│   ├── raw/                           # Original source files (if retained)
│   └── processed/
│       ├── paddy_ml_features_v2.csv   # Dashboard historical data
│       ├── linear_v2_errors.csv       # Retrospective model comparison data
│       ├── paddy_train_v2.csv
│       ├── paddy_validation_v2.csv
│       └── paddy_test_v2.csv
├── notebooks/
│   └── 01_paddy_eda.ipynb
└── src/
    ├── historical_features.py
    ├── feature_engineering_v2.py
    ├── prepare_v2_datasets.py
    ├── train_linear_regression_v2.py
    ├── train_random_forest_v2.py
    ├── evaluate_v2_models.py
    └── ...
```

The tree summarizes the principal files used during development; additional data extraction, diagnostics, and experimentation scripts may also be present.

## Run locally

**Requirements:** Python 3.13 (developed locally using Python 3.13.4), Git, and a virtual environment. Use compatible package versions, particularly for loading the serialized scikit-learn model.

```powershell
# Clone your repository (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/sri-lanka-paddy-yield-prediction.git
cd sri-lanka-paddy-yield-prediction

# Create and activate virtual environment on Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Launch dashboard
python -m streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). To avoid load errors, the repository must include `models/paddy_yield_linear_v2.joblib`, `data/processed/paddy_ml_features_v2.csv`, `data/processed/linear_v2_errors.csv`, and `src/historical_features.py`.

### Optional: regenerate the saved model

Provided the processed V2 training and validation CSVs are present:

```powershell
python src/save_model.py
```

This trains the Linear Regression V2 preprocessing/model pipeline on all pre-2020 observations and writes `models/paddy_yield_linear_v2.joblib`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose to deploy an existing app.
3. Select the repository, branch `main`, and entry point `app.py`.
4. Choose **Python 3.13** if available; install dependencies using `requirements.txt`.
5. Check the live dashboard's five tabs and review deployment logs if a dependency or model-loading issue occurs.

**Live app:** https://sri-lanka-paddy-yield-prediction.streamlit.app/

## Limitations and responsible use

- **Forecast-time leakage risk:** Earlier seasons' yields are filtered by season order, but their **publication dates** have not been checked. They may not have been available when a real forecast would have been issued.
- **Sown extent availability:** The model assumes the target-season sown area is known at prediction time; this may only be true after sowing reports are published.
- **Retrospective evaluation:** Historical lag features for test observations use actual previously observed yields, potentially including yields from earlier test-period seasons. Results represent rolling evaluation under availability assumptions, not long-range multi-season forecasting.
- **Distribution shift:** Performance deteriorated sharply in 2022; model predictions may be unreliable during unusual weather, input shortages, or economic conditions.
- **Scope:** The current model estimates **district-level yield per hectare**, not total paddy production or farm-specific yield. District averages are not official national aggregate statistics.
- **Data quality:** Some original workbooks had inconsistent years, area measures, or cross-metric discrepancies; these require further validation.
- **Research use only:** Do not use model outputs alone for financial, agronomic, or public policy decisions.

## Technology stack

**Python 3.13**, Pandas, NumPy, scikit-learn, Streamlit, Matplotlib, Seaborn, OpenPyXL, Joblib, Git, and GitHub.

## Data attribution

Data source: **Department of Census and Statistics (DCS), Sri Lanka**, historical paddy cultivation statistics covering Maha and Yala seasons. Refer to the original DCS publications for official definitions and latest revisions.

---

*Built as a data science and machine learning portfolio project focused on reproducible analysis, practical model evaluation, and transparent communication of limitations.*
