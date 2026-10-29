import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="UrbanEV | Charging Network Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #176b67 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin: 0; font-size: 2rem;}
    .hero p {color: #e1f5f2; margin: .4rem 0 0 0; font-size: 1rem;}
    div[data-testid="stMetric"] {
        background: #f5f8fa; border: 1px solid #e3ebef;
        padding: 13px 16px; border-radius: 13px;
    }
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    .section-note {color: #526575; font-size: .9rem;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_if_exists(filename):
    path = BASE / filename
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str).str.strip()
        return frame
    except Exception as exc:
        return {"_load_error": str(exc)}

@st.cache_data(show_spinner=False)
def load_project_data():
    return {
        "decision": read_csv_if_exists("intelligent_decision_support.csv"),
        "pricing": read_csv_if_exists("dynamic_pricing_simulation.csv"),
        "zones": read_csv_if_exists("zone-information.csv"),
        "shap": read_csv_if_exists("shap_feature_importance.csv"),
    }

loaded = load_project_data()

def usable_frame(value):
    return isinstance(value, pd.DataFrame) and not value.empty

decision = loaded["decision"]
pricing = loaded["pricing"]
zones = loaded["zones"]
shap_data = loaded["shap"]

# Show file errors without crashing the full app.
for label, value in loaded.items():
    if isinstance(value, dict) and "_load_error" in value:
        st.error(f"Could not read the {label} CSV: {value['_load_error']}")

if not usable_frame(decision) and not usable_frame(pricing):
    st.markdown("""
    <div class="hero">
      <h1>🔋 UrbanEV Charging Network Intelligence</h1>
      <p>Predictive congestion analysis and dynamic pricing decision support</p>
    </div>
    """, unsafe_allow_html=True)
    st.error("The dashboard could not find usable decision-support data.")
    st.markdown(
        "Place the following CSV files in the same folder as `app.py`: "
        "`intelligent_decision_support.csv`, `dynamic_pricing_simulation.csv`, "
        "`zone-information.csv`, and `shap_feature_importance.csv`."
    )
    st.stop()

# Use decision support as the primary table where available.
df = (decision.copy() if usable_frame(decision) else pricing.copy())
df.columns = df.columns.astype(str).str.strip()

# Normalise likely field types.
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

for col in [
    "zone_id", "occupancy", "predicted_congestion_probability",
    "congestion_probability", "congestion_probability_pct",
    "service_price", "dynamic_price", "price_increase_pct",
    "risk_priority", "latitude", "longitude",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create a single display probability in percentage units.
prob_col = next(
    (c for c in ["predicted_congestion_probability", "congestion_probability"]
     if c in df.columns),
    None,
)
if prob_col:
    raw_probability = pd.to_numeric(df[prob_col], errors="coerce")
    if raw_probability.notna().any():
        df["risk_percentage"] = (
            raw_probability * 100 if raw_probability.max() <= 1
            else raw_probability
        )
elif "congestion_probability_pct" in df.columns:
    df["risk_percentage"] = pd.to_numeric(
        df["congestion_probability_pct"], errors="coerce"
    )

if "risk_percentage" in df.columns:
    df["risk_percentage"] = df["risk_percentage"].clip(0, 100)

def risk_label(value):
    if pd.isna(value):
        return "Unknown"
    if value < 20:
        return "Low"
    if value < 40:
        return "Moderate"
    if value < 60:
        return "High"
    return "Very High"

if "risk_percentage" in df.columns:
    df["risk_category_display"] = df["risk_percentage"].map(risk_label)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🔋 UrbanEV Charging Network Intelligence</h1>
  <p>Predictive Congestion Modeling and Dynamic Pricing Optimization for EV Charging Networks</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Decision-support dashboard · UrbanEV zone-level data · "
    "Dynamic prices shown here are simulation-based recommendations."
)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Filter the records shown across the dashboard.")

filtered = df.copy()

if "time" in filtered.columns and filtered["time"].notna().any():
    valid_times = filtered["time"].dropna()
    min_date, max_date = valid_times.min().date(), valid_times.max().date()
    chosen_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(chosen_dates, (tuple, list)) and len(chosen_dates) == 2:
        start_date, end_date = chosen_dates
        filtered = filtered[
            filtered["time"].dt.date.between(start_date, end_date)
        ].copy()
else:
    st.sidebar.info("No valid time column is available for date filtering.")

zone_options = []
if "zone_id" in filtered.columns:
    zone_options = sorted(filtered["zone_id"].dropna().unique().tolist())
    selected_zones = st.sidebar.multiselect(
        "Charging zones",
        options=zone_options,
        default=[],
        help="Leave empty to include all zones in the selected date range.",
    )
    if selected_zones:
        filtered = filtered[filtered["zone_id"].isin(selected_zones)].copy()
else:
    selected_zones = []
    st.sidebar.info("No zone_id column is available for zone filtering.")

if filtered.empty:
    st.warning("No records match these filters. Adjust the date range or zones.")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown("**Risk bands**")
st.sidebar.markdown(
    "- 🟢 Low: <20%\n"
    "- 🟡 Moderate: 20–<40%\n"
    "- 🟠 High: 40–<60%\n"
    "- 🔴 Very high: ≥60%"
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_overview, tab_congestion, tab_pricing, tab_zones, tab_model, tab_data = st.tabs([
    "📊 Overview",
    "🚦 Congestion",
    "💰 Dynamic Pricing",
    "📍 Charging Zones",
    "🧠 Model Insights",
    "📁 Data & Export",
])

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Network overview")
    st.markdown(
        '<p class="section-note">A quick summary of the data currently selected in the sidebar.</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(filtered):,}")
    m2.metric(
        "Charging zones",
        f"{filtered['zone_id'].nunique():,}" if "zone_id" in filtered else "—",
    )
    m3.metric(
        "Mean occupancy",
        f"{filtered['occupancy'].mean():.2f}" if "occupancy" in filtered else "—",
    )
    m4.metric(
        "Time points",
        f"{filtered['time'].nunique():,}" if "time" in filtered else "—",
    )

    st.divider()
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Occupancy over time")
        if {"time", "occupancy"}.issubset(filtered.columns):
            trend = (
                filtered.dropna(subset=["time", "occupancy"])
                .groupby("time", as_index=True)["occupancy"].mean()
                .sort_index()
            )
            if not trend.empty:
                st.line_chart(trend, use_container_width=True)
            else:
                st.info("No occupancy observations are available for this selection.")
        else:
            st.info("The selected data does not contain both time and occupancy.")

    with right:
        st.markdown("#### Congestion risk mix")
        if "risk_category_display" in filtered.columns:
            order = ["Low", "Moderate", "High", "Very High", "Unknown"]
            counts = (
                filtered["risk_category_display"].value_counts()
                .reindex(order, fill_value=0)
            )
            counts = counts[counts > 0]
            st.bar_chart(counts, use_container_width=True)
        else:
            st.info("Congestion probability fields are not available.")

    if "time" in filtered.columns and filtered["time"].notna().any():
        st.caption(
            f"Selected period: {filtered['time'].min():%d %b %Y %H:%M} – "
            f"{filtered['time'].max():%d %b %Y %H:%M}"
        )

# ---------------------------------------------------------
# Congestion analysis
# ---------------------------------------------------------
with tab_congestion:
    st.subheader("Congestion risk analysis")
    if "risk_percentage" not in filtered.columns:
        st.warning(
            "No congestion-probability column was found. Expected "
            "`predicted_congestion_probability`, `congestion_probability`, "
            "or `congestion_probability_pct`."
        )
    else:
        risk = filtered.dropna(subset=["risk_percentage"]).copy()
        if risk.empty:
            st.info("No valid congestion probabilities are available.")
        else:
            high_risk = risk[risk["risk_percentage"] >= 40]
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean predicted risk", f"{risk['risk_percentage'].mean():.1f}%")
            c2.metric("High / very-high risk records", f"{len(high_risk):,}")
            c3.metric(
                "Zones with high risk",
                f"{high_risk['zone_id'].nunique():,}" if "zone_id" in high_risk else "—",
            )

            left, right = st.columns([1, 1.25])
            with left:
                st.markdown("#### Risk category distribution")
                risk_order = ["Low", "Moderate", "High", "Very High"]
                risk_counts = risk["risk_category_display"].value_counts().reindex(
                    risk_order, fill_value=0
                )
                st.bar_chart(risk_counts, use_container_width=True)
            with right:
                st.markdown("#### Predicted risk over time")
                if "time" in risk.columns:
                    risk_trend = risk.dropna(subset=["time"]).groupby("time")[
                        "risk_percentage"
                    ].mean().sort_index()
                    st.line_chart(risk_trend, use_container_width=True)
                else:
                    st.info("No time column is available for a risk trend.")

            st.markdown("#### Records requiring attention")
            display_candidates = [
                "time", "zone_id", "occupancy", "risk_percentage",
                "risk_category_display", "service_price", "dynamic_price",
                "price_increase_pct", "recommended_action", "operational_response",
                "risk_priority",
            ]
            show_cols = [c for c in display_candidates if c in risk.columns]
            if not high_risk.empty and show_cols:
                st.dataframe(
                    high_risk[show_cols].sort_values("risk_percentage", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
                st.download_button(
                    "Download high-risk records",
                    data=high_risk[show_cols].to_csv(index=False).encode("utf-8"),
                    file_name="urbanev_high_risk_records.csv",
                    mime="text/csv",
                )
            else:
                st.success("No records are at or above the 40% high-risk threshold.")

            st.caption(
                "Risk bands are display categories: Low <20%, Moderate 20–<40%, "
                "High 40–<60%, and Very High ≥60%."
            )

# ---------------------------------------------------------
# Dynamic pricing
# ---------------------------------------------------------
with tab_pricing:
    st.subheader("Dynamic pricing analysis")
    required_price_cols = {"service_price", "dynamic_price"}
    if not required_price_cols.issubset(filtered.columns):
        st.warning(
            "Pricing columns are unavailable in the selected main dataset. "
            "The dashboard expects `service_price` and `dynamic_price`."
        )
        if usable_frame(pricing):
            st.info(
                "Pricing data was found in `dynamic_pricing_simulation.csv`. "
                "If it uses a different set of columns, those names must be mapped "
                "to the dashboard fields before this section can display results."
            )
    else:
        price_df = filtered.copy()
        price_df["service_price"] = pd.to_numeric(price_df["service_price"], errors="coerce")
        price_df["dynamic_price"] = pd.to_numeric(price_df["dynamic_price"], errors="coerce")
        valid_prices = price_df.dropna(subset=["service_price", "dynamic_price"])

        if valid_prices.empty:
            st.info("No valid fixed/dynamic price pairs are available.")
        else:
            fixed_mean = valid_prices["service_price"].mean()
            dynamic_mean = valid_prices["dynamic_price"].mean()
            delta_pct = ((dynamic_mean - fixed_mean) / fixed_mean * 100) if fixed_mean else np.nan

            p1, p2, p3 = st.columns(3)
            p1.metric("Average fixed service price", f"{fixed_mean:.4f}")
            p2.metric(
                "Average simulated dynamic price",
                f"{dynamic_mean:.4f}",
                delta=f"{delta_pct:+.2f}%" if pd.notna(delta_pct) else None,
            )
            p3.metric(
                "Maximum dynamic price",
                f"{valid_prices['dynamic_price'].max():.4f}",
            )

            st.markdown("#### Average price comparison")
            comparison = pd.DataFrame(
                {"Average price": [fixed_mean, dynamic_mean]},
                index=["Fixed pricing", "Simulated dynamic pricing"],
            )
            st.bar_chart(comparison, use_container_width=True)

            if "time" in valid_prices.columns:
                st.markdown("#### Price trend over time")
                daily = (
                    valid_prices.dropna(subset=["time"])
                    .set_index("time")[["service_price", "dynamic_price"]]
                    .resample("D").mean().dropna(how="all")
                )
                if not daily.empty:
                    daily.columns = ["Fixed service price", "Dynamic price"]
                    st.line_chart(daily, use_container_width=True)

            if "price_increase_pct" in valid_prices.columns:
                st.markdown("#### Frequency of price adjustments")
                adjustments = pd.to_numeric(
                    valid_prices["price_increase_pct"], errors="coerce"
                ).dropna().round(2)
                if not adjustments.empty:
                    counts = adjustments.value_counts().sort_index()
                    counts.index = [f"{x:g}%" for x in counts.index]
                    st.bar_chart(counts, use_container_width=True)

            if "recommended_action" in valid_prices.columns:
                st.markdown("#### Recommended actions")
                actions = valid_prices["recommended_action"].fillna("Not specified").value_counts()
                st.bar_chart(actions, use_container_width=True)

            st.info(
                "These prices are simulation-based recommendations. The displayed "
                "price change is not evidence of a measured real-world revenue increase "
                "or a verified change in driver behaviour."
            )

# ---------------------------------------------------------
# Zone information and map
# ---------------------------------------------------------
with tab_zones:
    st.subheader("Charging zone explorer")
    if not usable_frame(zones):
        st.warning("Zone metadata is unavailable. Check `zone-information.csv`.")
    else:
        zone_view = zones.copy()
        zone_view.columns = zone_view.columns.astype(str).str.strip()
        if "TAZID" not in zone_view.columns:
            st.error("The zone metadata file needs a `TAZID` column.")
        else:
            zone_view["TAZID"] = pd.to_numeric(zone_view["TAZID"], errors="coerce")
            zone_view = zone_view.dropna(subset=["TAZID"]).copy()

            if selected_zones:
                chosen_ids = pd.to_numeric(pd.Series(selected_zones), errors="coerce").dropna()
                zone_view = zone_view[zone_view["TAZID"].isin(chosen_ids)]

            if "charge_count" in zone_view.columns:
                zone_view["charge_count"] = pd.to_numeric(zone_view["charge_count"], errors="coerce")

            z1, z2 = st.columns(2)
            z1.metric("Zones in view", f"{len(zone_view):,}")
            z2.metric(
                "Total charge count in metadata",
                f"{zone_view['charge_count'].sum():,.0f}" if "charge_count" in zone_view else "—",
            )

            if {"latitude", "longitude"}.issubset(zone_view.columns):
                map_df = zone_view[["latitude", "longitude"]].copy()
                map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
                map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
                map_df = map_df.replace([np.inf, -np.inf], np.nan).dropna()
                map_df = map_df[
                    map_df["latitude"].between(-90, 90)
                    & map_df["longitude"].between(-180, 180)
                ]
                st.markdown("#### Zone locations")
                if not map_df.empty:
                    st.map(map_df, latitude="latitude", longitude="longitude", size=20)
                else:
                    st.warning("No valid coordinates are available for the selected zones.")
            else:
                st.info("The metadata needs `latitude` and `longitude` columns to display a map.")

            st.markdown("#### Zone metadata")
            st.dataframe(zone_view, use_container_width=True, hide_index=True, height=350)

# ---------------------------------------------------------
# Model explainability
# ---------------------------------------------------------
with tab_model:
    st.subheader("Model insights")
    if not usable_frame(shap_data):
        st.info("SHAP feature-importance results are unavailable.")
    else:
        shap_view = shap_data.copy()
        shap_view.columns = shap_view.columns.astype(str).str.strip()
        feature_col = next((c for c in shap_view.columns if "feature" in c.lower()), None)
        importance_col = next(
            (
                c for c in shap_view.select_dtypes(include=np.number).columns
                if "importance" in c.lower() or "shap" in c.lower()
            ),
            None,
        )
        if feature_col and importance_col:
            shap_view[importance_col] = pd.to_numeric(shap_view[importance_col], errors="coerce")
            plot_df = shap_view[[feature_col, importance_col]].dropna()
            plot_df = plot_df.sort_values(importance_col, ascending=False).head(15)
            if not plot_df.empty:
                st.markdown("#### Most influential features")
                st.bar_chart(plot_df.set_index(feature_col), use_container_width=True)
            else:
                st.info("No valid feature-importance values are available to plot.")
        else:
            st.info(
                "Could not identify feature and numeric importance columns. "
                "Check the column names in `shap_feature_importance.csv`."
            )
        with st.expander("View feature-importance table"):
            st.dataframe(shap_view, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Data preview and export
# ---------------------------------------------------------
with tab_data:
    st.subheader("Filtered data and export")
    st.write(f"**Rows:** {len(filtered):,} · **Columns:** {len(filtered.columns):,}")
    search = st.text_input("Search column names", placeholder="e.g. occupancy, price, zone")
    preview_cols = list(filtered.columns)
    if search.strip():
        preview_cols = [c for c in preview_cols if search.lower() in c.lower()]
    if preview_cols:
        st.dataframe(filtered[preview_cols].head(1000), use_container_width=True, hide_index=True)
    else:
        st.info("No column names match that search.")

    st.download_button(
        "Download filtered decision-support data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="urbanev_filtered_decision_support.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "UrbanEV Decision Support · Predictive Congestion Modeling and Dynamic Pricing "
    "Optimization for EV Charging Networks"
)
