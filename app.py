
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="UrbanEV Decision Support",
    page_icon="🔋",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE
DATASETS = BASE

st.title("🔋 UrbanEV Intelligent Decision-Support System")
st.caption(
    "Predictive congestion monitoring and dynamic pricing "
    "for electric vehicle charging networks"
)

@st.cache_data
def load_data():
    decision_path = BASE / "intelligent_decision_support.csv"
    pricing_path = BASE / "dynamic_pricing_simulation.csv"
    zones_path = BASE / "zone-information.csv"
    shap_path = BASE / "shap_feature_importance.csv"

    decision = pd.read_csv(decision_path) if decision_path.exists() else None
    pricing = pd.read_csv(pricing_path) if pricing_path.exists() else None
    zones = pd.read_csv(zones_path) if zones_path.exists() else None
    shap = pd.read_csv(shap_path) if shap_path.exists() else None

    return decision, pricing, zones, shap

try:
    decision, pricing, zones, shap = load_data()
except Exception as e:
    st.error(f"Unable to load project results: {e}")
    st.stop()

if decision is None and pricing is None:
    st.error("Neither decision-support nor pricing results were found.")
    st.stop()

df = decision.copy() if decision is not None else pricing.copy()

# Parse timestamps where available
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Sidebar filters
st.sidebar.header("Filters")

if "time" in df.columns and df["time"].notna().any():
    valid_dates = df["time"].dropna()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(valid_dates.min().date(), valid_dates.max().date())
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            df["time"].dt.date.between(start_date, end_date)
        ]

if "zone_id" in df.columns:
    zone_values = sorted(df["zone_id"].dropna().unique().tolist())
    selected_zones = st.sidebar.multiselect(
        "Charging zones",
        zone_values,
        default=[]
    )
    if selected_zones:
        df = df[df["zone_id"].isin(selected_zones)]

# Summary metrics
st.subheader("Network Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(df):,}")

if "occupancy" in df.columns:
    c2.metric("Average occupancy", f"{df['occupancy'].mean():.2f}")
else:
    c2.metric("Average occupancy", "N/A")

prob_col = next(
    (c for c in ["predicted_congestion_probability",
                 "congestion_probability"] if c in df.columns),
    None
)

if prob_col:
    probability = pd.to_numeric(df[prob_col], errors="coerce")
    # Detect probabilities stored as fractions versus percentages
    if probability.max() <= 1:
        probability_pct = probability * 100
    else:
        probability_pct = probability

    c3.metric("Mean predicted congestion risk",
              f"{probability_pct.mean():.2f}%")
else:
    probability_pct = None
    c3.metric("Mean predicted congestion risk", "N/A")

price_col = next(
    (c for c in ["dynamic_price", "recommended_price",
                 "recommended_service_price"] if c in df.columns),
    None
)

if price_col:
    c4.metric("Average recommended price",
              f"{pd.to_numeric(df[price_col], errors='coerce').mean():.3f}")
else:
    c4.metric("Average recommended price", "N/A")

# Risk classification
st.subheader("Congestion Risk Analysis")

if probability_pct is not None:
    df["risk_percentage"] = probability_pct

    def risk_label(p):
        if pd.isna(p):
            return "Unknown"
        if p < 20:
            return "Low"
        if p < 40:
            return "Moderate"
        if p < 60:
            return "High"
        return "Very High"

    df["risk_category_display"] = df["risk_percentage"].apply(risk_label)

    risk_counts = (
        df["risk_category_display"]
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Very High"], fill_value=0)
    )

    st.bar_chart(risk_counts)

    attention = df[
        df["risk_category_display"].isin(["High", "Very High"])
    ].copy()

    st.write(f"**Zones requiring attention:** {attention['zone_id'].nunique() if 'zone_id' in attention.columns else len(attention):,}")
    if not attention.empty:
        preferred = [
            "time", "zone_id", "occupancy", prob_col,
            "risk_category", "risk_category_display",
            "dynamic_price", "price_change", "price_increase_pct",
            "recommended_action"
        ]

        visible = [c for c in preferred if c in attention.columns]

        attention_display = attention[visible].copy()
        attention_display["risk_percentage"] = attention["risk_percentage"]

        st.dataframe(
            attention_display.sort_values(
                "risk_percentage", ascending=False
            ),
            use_container_width=True,
            height=350
        )
    
# Pricing analysis
st.subheader("Dynamic Pricing")

if price_col and "time" in df.columns:
    price_trend = df.dropna(subset=["time"]).copy()
    if not price_trend.empty:
        price_trend["hour"] = price_trend["time"].dt.floor("D")
        daily_price = price_trend.groupby("hour")[price_col].mean()
        st.line_chart(daily_price)

if "price_increase_pct" in df.columns:
    st.write("Price adjustment distribution")
    st.bar_chart(df["price_increase_pct"].value_counts().sort_index())

# Spatial information
st.subheader("Charging Zone Information")

if zones is not None:
    st.write("Available zone metadata")
    st.dataframe(zones, use_container_width=True, height=250)

    if {"longitude", "latitude"}.issubset(zones.columns):
        map_data = zones[["latitude", "longitude"]].copy()
        map_data = map_data.apply(pd.to_numeric, errors="coerce").dropna()
        if not map_data.empty:
            st.map(map_data)

# SHAP information
st.subheader("Model Explainability (SHAP)")

if shap is not None:
    st.dataframe(shap, use_container_width=True)
    numeric_cols = shap.select_dtypes(include=np.number).columns.tolist()
    feature_col = next(
        (c for c in shap.columns if "feature" in c.lower()),
        shap.columns[0] if len(shap.columns) else None
    )
    importance_col = next(
        (c for c in numeric_cols if "importance" in c.lower() or "shap" in c.lower()),
        numeric_cols[0] if numeric_cols else None
    )

    if feature_col and importance_col:
        plot_data = shap[[feature_col, importance_col]].dropna()
        plot_data = plot_data.sort_values(importance_col, ascending=False).head(15)
        st.bar_chart(plot_data.set_index(feature_col))

# Export
st.subheader("Export Results")
st.download_button(
    "Download filtered decision-support data",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="urbanev_decision_support_filtered.csv",
    mime="text/csv"
)

st.info(
    "Pricing recommendations are simulation-based. They should be "
    "validated against driver response and operational constraints before "
    "being used in a live charging network."
)
