import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="UrbanEV Decision Support",
    page_icon="🔋",
    layout="wide"
)

BASE = Path(__file__).resolve().parent


# =========================================================
# 2. TITLE
# =========================================================

st.title("🔋 UrbanEV Intelligent Decision-Support System")

st.caption(
    "Predictive congestion monitoring and dynamic pricing "
    "for electric vehicle charging networks"
)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    decision_path = BASE / "intelligent_decision_support.csv"
    pricing_path = BASE / "dynamic_pricing_simulation.csv"
    zones_path = BASE / "zone-information.csv"
    shap_path = BASE / "shap_feature_importance.csv"

    decision = (
        pd.read_csv(decision_path)
        if decision_path.exists()
        else None
    )

    pricing = (
        pd.read_csv(pricing_path)
        if pricing_path.exists()
        else None
    )

    zones = (
        pd.read_csv(zones_path)
        if zones_path.exists()
        else None
    )

    shap = (
        pd.read_csv(shap_path)
        if shap_path.exists()
        else None
    )

    return decision, pricing, zones, shap


try:
    decision, pricing, zones, shap = load_data()

except Exception as e:
    st.error(f"Unable to load project results: {e}")
    st.stop()


if decision is None and pricing is None:
    st.error(
        "Neither decision-support nor pricing results were found. "
        "Check that the CSV files are in the same folder as app.py."
    )
    st.stop()


# =========================================================
# 4. PREPARE DECISION-SUPPORT DATA
# =========================================================

df = decision.copy() if decision is not None else pricing.copy()

if "time" in df.columns:
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

if "zone_id" in df.columns:
    df["zone_id"] = pd.to_numeric(
        df["zone_id"],
        errors="coerce"
    )


# =========================================================
# 5. SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Dashboard Filters")

if "time" in df.columns and df["time"].notna().any():

    valid_dates = df["time"].dropna()

    date_range = st.sidebar.date_input(
        "Select date range",
        value=(
            valid_dates.min().date(),
            valid_dates.max().date()
        )
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

        start_date, end_date = date_range

        df = df[
            df["time"].dt.date.between(
                start_date,
                end_date
            )
        ].copy()


selected_zones = []

if "zone_id" in df.columns:

    zone_values = sorted(
        df["zone_id"].dropna().unique().tolist()
    )

    selected_zones = st.sidebar.multiselect(
        "Select charging zones",
        options=zone_values,
        default=[]
    )

    if selected_zones:
        df = df[
            df["zone_id"].isin(selected_zones)
        ].copy()


# =========================================================
# 6. NETWORK OVERVIEW
# =========================================================

st.subheader("Network Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Records",
    f"{len(df):,}"
)


# Average occupancy

if "occupancy" in df.columns:

    occupancy_values = pd.to_numeric(
        df["occupancy"],
        errors="coerce"
    )

    c2.metric(
        "Average occupancy",
        f"{occupancy_values.mean():.2f}"
    )

else:
    c2.metric(
        "Average occupancy",
        "N/A"
    )


# Predicted congestion probability

prob_col = next(
    (
        col for col in [
            "predicted_congestion_probability",
            "congestion_probability"
        ]
        if col in df.columns
    ),
    None
)

probability_pct = None

if prob_col is not None:

    probability = pd.to_numeric(
        df[prob_col],
        errors="coerce"
    )

    if probability.notna().any():

        # Convert fractional probabilities into percentages.
        # Example: 0.25 becomes 25%.

        if probability.max() <= 1:
            probability_pct = probability * 100
        else:
            probability_pct = probability

        c3.metric(
            "Mean predicted congestion risk",
            f"{probability_pct.mean():.2f}%"
        )

    else:
        c3.metric(
            "Mean predicted congestion risk",
            "N/A"
        )

else:
    c3.metric(
        "Mean predicted congestion risk",
        "N/A"
    )


# Average recommended price

price_col = next(
    (
        col for col in [
            "dynamic_price",
            "recommended_price",
            "recommended_service_price"
        ]
        if col in df.columns
    ),
    None
)

if price_col is not None:

    price_values = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )

    c4.metric(
        "Average recommended price",
        f"{price_values.mean():.3f}"
    )

else:
    c4.metric(
        "Average recommended price",
        "N/A"
    )


# =========================================================
# 7. CONGESTION RISK ANALYSIS
# =========================================================

st.subheader("Congestion Risk Analysis")

if probability_pct is not None:

    df["risk_percentage"] = probability_pct

    def risk_label(probability):

        if pd.isna(probability):
            return "Unknown"

        if probability < 20:
            return "Low"

        elif probability < 40:
            return "Moderate"

        elif probability < 60:
            return "High"

        else:
            return "Very High"


    df["risk_category_display"] = (
        df["risk_percentage"].apply(risk_label)
    )

    risk_order = [
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]

    risk_counts = (
        df["risk_category_display"]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0
        )
    )

    st.write(
        "Distribution of records by predicted congestion-risk category"
    )

    st.bar_chart(risk_counts)


    # Records requiring attention

    attention = df[
        df["risk_category_display"].isin(
            ["High", "Very High"]
        )
    ].copy()

    if "zone_id" in attention.columns:
        attention_count = attention["zone_id"].nunique()
    else:
        attention_count = len(attention)

    st.write(
        f"**Zones requiring attention:** {attention_count:,}"
    )


    if not attention.empty:

        preferred_columns = [
            "time",
            "zone_id",
            "occupancy",
            prob_col,
            "congestion_risk",
            "risk_category_display",
            "service_price",
            "dynamic_price",
            "price_increase_pct",
            "recommended_action",
            "operational_response",
            "risk_priority"
        ]

        visible_columns = [
            col for col in preferred_columns
            if col in attention.columns
        ]

        attention_display = attention[
            visible_columns
        ].copy()

        attention_display = attention_display.sort_values(
            "risk_percentage",
            ascending=False
        )

        st.write("Records requiring attention")

        st.dataframe(
            attention_display,
            use_container_width=True,
            height=350,
            hide_index=True
        )

    else:
        st.info(
            "No records fall into the High or Very High "
            "categories for the selected filters."
        )

else:

    st.warning(
        "Congestion-risk analysis is unavailable because "
        "a probability column was not found."
    )


# =========================================================
# 8. DYNAMIC PRICING ANALYSIS
# =========================================================

st.subheader("Dynamic Pricing")

if (
    price_col is not None
    and "time" in df.columns
):

    price_trend = df.dropna(
        subset=["time"]
    ).copy()

    price_trend[price_col] = pd.to_numeric(
        price_trend[price_col],
        errors="coerce"
    )

    price_trend = price_trend.dropna(
        subset=[price_col]
    )

    if not price_trend.empty:

        price_trend["date"] = (
            price_trend["time"].dt.floor("D")
        )

        daily_price = (
            price_trend
            .groupby("date")[price_col]
            .mean()
        )

        st.write("Average recommended price over time")

        st.line_chart(daily_price)

    else:
        st.info(
            "No valid price records are available "
            "for the selected filters."
        )


if "price_increase_pct" in df.columns:

    price_increase = pd.to_numeric(
        df["price_increase_pct"],
        errors="coerce"
    ).dropna()

    if not price_increase.empty:

        st.write("Price adjustment distribution")

        adjustment_counts = (
            price_increase
            .value_counts()
            .sort_index()
        )

        st.bar_chart(adjustment_counts)


# =========================================================
# 9. CHARGING ZONE INFORMATION AND MAP
# =========================================================

st.subheader("Charging Zone Information")

if zones is not None and not zones.empty:

    # Work on a copy so the original data remains unchanged.
    zone_view = zones.copy()

    # Remove accidental spaces from column names.
    zone_view.columns = (
        zone_view.columns.astype(str).str.strip()
    )

    # Clean the zone identifier.
    if "TAZID" in zone_view.columns:

        zone_view["TAZID"] = pd.to_numeric(
            zone_view["TAZID"],
            errors="coerce"
        )

        # Remove rows that do not have a usable zone ID.
        zone_view = zone_view.dropna(
            subset=["TAZID"]
        ).copy()

    else:

        st.error(
            "The zone-information file does not contain "
            "the required TAZID column."
        )

        zone_view = pd.DataFrame()


    if not zone_view.empty:

        # Clean coordinate and numeric columns.
        numeric_zone_columns = [
            "longitude",
            "latitude",
            "charge_count",
            "area",
            "perimeter"
        ]

        for col in numeric_zone_columns:

            if col in zone_view.columns:

                zone_view[col] = pd.to_numeric(
                    zone_view[col],
                    errors="coerce"
                )


        # Match selected decision-support zones to metadata.
        # The decision dataset uses zone_id.
        # The zone-information dataset uses TAZID.

        if selected_zones:

            selected_ids = pd.to_numeric(
                pd.Series(selected_zones),
                errors="coerce"
            ).dropna()

            zone_view = zone_view[
                zone_view["TAZID"].isin(selected_ids)
            ].copy()


        st.write(
            f"**Charging zones displayed:** {len(zone_view):,}"
        )


        # Display zone metadata.
        st.dataframe(
            zone_view,
            use_container_width=True,
            height=300,
            hide_index=True
        )


        # Display map when coordinates are available.
        if (
            "longitude" in zone_view.columns
            and "latitude" in zone_view.columns
        ):

            map_data = zone_view[
                ["latitude", "longitude"]
            ].copy()

            map_data = map_data.replace(
                [np.inf, -np.inf],
                np.nan
            )

            map_data = map_data.dropna(
                subset=["latitude", "longitude"]
            )

            # Keep only valid geographic coordinates.
            map_data = map_data[
                map_data["latitude"].between(-90, 90)
                & map_data["longitude"].between(-180, 180)
            ]

            if not map_data.empty:

                st.write("### Charging Zone Locations")

                st.map(map_data)

            else:

                st.warning(
                    "No valid latitude and longitude values "
                    "are available for the selected zones."
                )

        else:

            st.warning(
                "The zone-information file does not contain "
                "both longitude and latitude columns."
            )


    else:

        st.warning(
            "No valid charging-zone metadata is available."
        )

else:

    st.warning(
        "Zone-information data could not be loaded. "
        "Check that zone-information.csv is in the same "
        "folder as app.py."
    )


# =========================================================
# 10. MODEL EXPLAINABILITY (SHAP)
# =========================================================

st.subheader("Model Explainability (SHAP)")

if shap is not None and not shap.empty:

    st.write("Feature-importance results")

    st.dataframe(
        shap,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = shap.select_dtypes(
        include=np.number
    ).columns.tolist()

    feature_column = next(
        (
            col for col in shap.columns
            if "feature" in col.lower()
        ),
        shap.columns[0] if len(shap.columns) else None
    )

    importance_column = next(
        (
            col for col in numeric_columns
            if (
                "importance" in col.lower()
                or "shap" in col.lower()
            )
        ),
        numeric_columns[0] if numeric_columns else None
    )

    if feature_column and importance_column:

        plot_data = shap[
            [feature_column, importance_column]
        ].copy()

        plot_data[importance_column] = pd.to_numeric(
            plot_data[importance_column],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        plot_data = plot_data.sort_values(
            importance_column,
            ascending=False
        ).head(15)

        if not plot_data.empty:

            st.write("Top 15 important features")

            st.bar_chart(
                plot_data.set_index(feature_column)
            )

        else:

            st.info(
                "No valid numerical feature-importance "
                "values are available."
            )

else:

    st.info(
        "SHAP feature-importance data is not available."
    )


# =========================================================
# 11. EXPORT FILTERED RESULTS
# =========================================================

st.subheader("Export Results")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download filtered decision-support data",
    data=csv_data,
    file_name="urbanev_decision_support_filtered.csv",
    mime="text/csv"
)


# =========================================================
# 12. DISCLAIMER
# =========================================================

st.info(
    "Pricing recommendations are simulation-based. "
    "They should be validated against driver response and "
    "operational constraints before being used in a live "
    "charging network."
)
