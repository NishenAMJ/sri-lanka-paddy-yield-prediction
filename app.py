
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src.historical_features import calculate_historical_features
# ==========================================
# 1. APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sri Lanka Paddy Yield Dashboard",
    page_icon="🌾",
    layout="wide"
)

DATA_PATH = Path("data/processed/paddy_ml_features_v2.csv")
MODEL_PATH = Path("models/paddy_yield_linear_v2.joblib")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, dtype={"year": str})
    data["district"] = data["district"].str.strip()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

districts = sorted(df["district"].unique())
years = sorted(df["season_year"].unique())

st.title("🌾 Sri Lanka Paddy Yield Dashboard")
st.caption(
    "Historical paddy yield analytics and experimental "
    "machine learning predictions"
)

# ==========================================
# 2. NAVIGATION
# ==========================================

overview_tab, ranking_tab, comparison_tab, performance_tab, prediction_tab = st.tabs([
    "🌍 Islandwide Overview",
    "📊 District Rankings",
    "🔍 District Comparison",
    "📈 Model Performance",
    "🤖 Yield Prediction"
])

# ==========================================
# 3. ISLANDWIDE OVERVIEW
# ==========================================

with overview_tab:

    st.header("Islandwide Paddy Yield Overview")

    st.info(
        "Islandwide summaries use unweighted district "
        "averages, not official national production-weighted "
        "yield statistics."
    )

    selected_year = st.selectbox(
        "Select Year",
        years,
        index=len(years) - 1,
        key="overview_year"
    )

    year_data = df[df["season_year"] == selected_year]

    avg_yield = year_data["average_yield_kg_ha"].mean()
    total_sown = year_data["sown_extent_ha"].sum()
    district_count = year_data["district"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average District Yield",
        f"{avg_yield:,.0f} kg/ha"
    )

    col2.metric(
        "Recorded Sown Extent",
        f"{total_sown:,.0f} ha"
    )

    col3.metric(
        "Districts Covered",
        district_count
    )

    st.caption(
        "Sown extent is summed across available "
        "district-season observations in the selected year."
    )

    st.subheader("Historical Yield Trends")

    yearly_trends = (
        df.groupby(["season_year", "season"])[
            "average_yield_kg_ha"
        ]
        .mean()
        .unstack()
        .sort_index()
    )

    st.line_chart(
        yearly_trends,
        x_label="Year",
        y_label="Mean District Yield (kg/ha)"
    )

    st.subheader("Maha vs Yala")

    season_summary = (
        df.groupby("season")["average_yield_kg_ha"]
        .agg(["mean", "median", "count"])
        .round(2)
    )

    st.dataframe(
        season_summary,
        use_container_width=True
    )

    st.subheader("Yearly Average District Yield")

    yearly_avg = (
        df.groupby("season_year")["average_yield_kg_ha"]
        .mean()
        .sort_index()
    )

    st.line_chart(
        yearly_avg,
        x_label="Year",
        y_label="Mean District Yield (kg/ha)"
    )

    st.warning(
        "Historical note: Sri Lanka's 2021–2022 "
        "fertilizer shortages and economic crisis "
        "coincided with substantial agricultural "
        "disruptions. This dashboard shows historical "
        "patterns; it does not establish causation."
    )

# ==========================================
# 4. DISTRICT RANKINGS
# ==========================================

with ranking_tab:

    st.header("District Yield Rankings")

    col1, col2 = st.columns(2)

    with col1:
        ranking_year = st.selectbox(
            "Year",
            years,
            index=len(years) - 1,
            key="ranking_year"
        )

    with col2:
        ranking_season = st.selectbox(
            "Season",
            ["Maha", "Yala"],
            key="ranking_season"
        )

    ranking_data = df[
        (df["season_year"] == ranking_year) &
        (df["season"] == ranking_season)
    ]

    if ranking_data.empty:
        st.warning("No observations for this year and season.")

    else:
        ranking = (
            ranking_data.groupby("district")[
                "average_yield_kg_ha"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Highest Yield District",
            ranking.index[0],
            f"{ranking.iloc[0]:,.0f} kg/ha",
            delta_color="off"
        )

        col2.metric(
            "Lowest Yield District",
            ranking.index[-1],
            f"{ranking.iloc[-1]:,.0f} kg/ha",
            delta_color="off"
        )

        st.subheader("District Yield Ranking")

        st.bar_chart(
            ranking,
            horizontal=True,
            x_label="Yield (kg/ha)",
            y_label="District",
            height=650
        )

        st.dataframe(
            ranking.rename("Yield (kg/ha)")
            .reset_index(),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# 5. DISTRICT COMPARISON
# ==========================================

with comparison_tab:

    st.header("Compare Districts")

    col1, col2, col3 = st.columns(3)

    with col1:
        district_a = st.selectbox(
            "First District",
            districts,
            key="district_a"
        )

    with col2:
        district_b = st.selectbox(
            "Second District",
            districts,
            index=min(1, len(districts) - 1),
            key="district_b"
        )

    with col3:
        comparison_season = st.selectbox(
            "Cultivation Season",
            ["Maha", "Yala"],
            key="comparison_season"
        )

    if district_a == district_b:
        st.warning("Select two different districts.")

    else:
        comparison_data = df[
            df["district"].isin([district_a, district_b]) &
            (df["season"] == comparison_season)
        ]

        comparison_trends = (
            comparison_data.groupby(
                ["season_year", "district"]
            )["average_yield_kg_ha"]
            .mean()
            .unstack()
            .sort_index()
        )

        st.subheader("Historical Yield Comparison")

        st.line_chart(
            comparison_trends,
            x_label="Year",
            y_label="Yield (kg/ha)"
        )

        # Compare only years reported for both districts
        common_years = comparison_trends.dropna()

        if not common_years.empty:

            avg_a = common_years[district_a].mean()
            avg_b = common_years[district_b].mean()

            col1, col2 = st.columns(2)

            col1.metric(
                f"{district_a} Average",
                f"{avg_a:,.0f} kg/ha"
            )

            col2.metric(
                f"{district_b} Average",
                f"{avg_b:,.0f} kg/ha"
            )

            st.caption(
                f"Comparison uses {len(common_years)} "
                "years where both districts have records."
            )

            st.subheader("Year-by-Year Comparison")

            st.dataframe(
                common_years.round(2),
                use_container_width=True
            )

        else:
            st.warning("No overlapping historical years.")

# ==========================================
# 6. ML YIELD PREDICTION
# ==========================================


# ==========================================
# MODEL PERFORMANCE ANALYSIS
# ==========================================

with performance_tab:

    st.header("Actual vs Predicted Paddy Yield")

    st.caption(
        "Historical model evaluation using "
        "2020–2026 observations."
    )

    @st.cache_data
    def load_predictions():
        return pd.read_csv(
            "data/processed/linear_v2_errors.csv",
            dtype={"year": str}
        )

    results = load_predictions()

    col1, col2 = st.columns(2)

    with col1:
        selected_district = st.selectbox(
            "Select District",
            sorted(results["district"].unique()),
            key="performance_district"
        )

    with col2:
        selected_season = st.selectbox(
            "Select Season",
            ["Maha", "Yala"],
            key="performance_season"
        )

    # Filter data
    filtered = results[
        (results["district"] == selected_district) &
        (results["season"] == selected_season)
    ].copy()

    filtered = filtered.sort_values("season_year")

    if filtered.empty:
        st.warning(
            "No prediction records for this selection."
        )

    else:

        # --------------------------------------
        # PERFORMANCE METRICS
        # --------------------------------------

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        import numpy as np

        actual = filtered["actual_yield"]
        predicted = filtered["predicted_yield"]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean Absolute Error",
            f"{mae:,.2f} kg/ha"
        )

        col2.metric(
            "Root Mean Squared Error",
            f"{rmse:,.2f} kg/ha"
        )

        if len(filtered) >= 2:

            r2 = r2_score(actual, predicted)

            col3.metric(
                "R² Score",
                f"{r2:.3f}"
            )

        else:

            col3.metric(
                "R² Score",
                "N/A"
            )

        # --------------------------------------
        # ACTUAL VS PREDICTED LINE CHART
        # --------------------------------------

        st.subheader("Historical Yield Comparison")

        chart_data = filtered.set_index("season_year")[
            ["actual_yield", "predicted_yield"]
        ]

        st.line_chart(
            chart_data,
            x_label="Year",
            y_label="Paddy Yield (kg/ha)"
        )

        # --------------------------------------
        # ERROR BY YEAR
        # --------------------------------------

        st.subheader("Prediction Error by Year")

        error_chart = filtered.set_index("season_year")[
            "absolute_error"
        ]

        st.bar_chart(
            error_chart,
            x_label="Year",
            y_label="Absolute Error (kg/ha)"
        )

        # --------------------------------------
        # DETAILED RESULTS
        # --------------------------------------

        st.subheader("Detailed Predictions")

        display_data = filtered[
            [
                "year",
                "season",
                "actual_yield",
                "predicted_yield",
                "absolute_error"
            ]
        ].copy()

        display_data = display_data.rename(columns={
            "year": "Year",
            "season": "Season",
            "actual_yield": "Actual Yield (kg/ha)",
            "predicted_yield": "Predicted Yield (kg/ha)",
            "absolute_error": "Absolute Error (kg/ha)"
        })

        st.dataframe(
            display_data.round(2),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------
        # DOWNLOAD RESULTS
        # --------------------------------------

        st.download_button(
            label="Download Prediction Results",
            data=display_data.to_csv(index=False),
            file_name=(
                f"{selected_district}_"
                f"{selected_season}_predictions.csv"
            ),
            mime="text/csv"
        )

    st.info(
        "These are retrospective evaluation results "
        "from Linear Regression V2, not forecasts "
        "generated by the current prediction form. "
        "The model was trained on pre-2020 data, "
        "but forecast-time availability of historical "
        "inputs has not been independently verified."
    )


with prediction_tab:

    st.header("Machine Learning Yield Prediction")

    st.info(
        "Select a district, cultivation season, year, "
        "and sown extent. Historical yield features "
        "are calculated automatically."
    )

    with st.form("prediction_form"):

        district = st.selectbox(
            "District",
            districts
        )

        season = st.selectbox(
            "Cultivation Season",
            ["Maha", "Yala"]
        )

        season_year = st.number_input(
            "Season Year (Maha ending year)",
            min_value=1979,
            max_value=2035,
            value=2025,
            step=1
        )

        sown_extent = st.number_input(
            "Sown Extent (hectares)",
            min_value=0.01,
            value=20000.0
        )

        submitted = st.form_submit_button(
            "Predict Paddy Yield",
            use_container_width=True
        )

    if submitted:

        # Automatically calculate historical features
        history = calculate_historical_features(
            df,
            district,
            season,
            season_year
        )

        input_data = pd.DataFrame([{
            "district": district,
            "season": season,
            "season_year": season_year,
            "sown_extent_ha": sown_extent,
            "season_gap": history["season_gap"],
            "previous_yield": history["previous_yield"],
            "yield_3season_avg": history["yield_3season_avg"]
        }])

        # Display historical values (read-only)
        st.subheader("Historical Data Used")

        st.write(
            "Previous Season Yield:",
            history["previous_yield"]
        )

        st.write(
            "Previous 3-Season Average:",
            history["yield_3season_avg"]
        )

        st.write(
            "Seasons Since Previous Observation:",
            history["season_gap"]
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Yield",
            f"{prediction:,.2f} kg/ha"
        )

        col2.metric(
            "Yield (tonnes/ha)",
            f"{prediction / 1000:.2f}"
        )

        if input_data[
            ["previous_yield", "yield_3season_avg"]
        ].isna().any(axis=None):

            st.warning(
                "Some historical yield data is missing. "
                "The model has used imputed values."
            )

        st.caption(
            "Experimental prediction based on historical "
            "district-level data. Earlier season records "
            "may not have been published at the requested "
            "forecast date."
        )

# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "Sri Lanka Paddy Yield Analytics & Prediction | "
    "District-level historical data | "
    "Linear Regression V2 | "
    "Experimental research and portfolio project"
)
