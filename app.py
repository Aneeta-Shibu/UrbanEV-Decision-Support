import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Folium is used for the spatial map. The app still runs if it is unavailable.
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="UrbanEV | Smart Charging Network",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        padding: 13px 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 15px;
        background: linear-gradient(115deg, #102d46, #176b72);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin: 0 0 .35rem 0;}
    .hero p {color: #e5f2f4; margin: 0;}
    .small-note {color: #64748b; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# DATA LOADING
# Place the CSV files in the same GitHub folder as app.py.
# ---------------------------------------------------------
FILE_NAMES = {
    "pricing": "dynamic_pricing_simulation.csv",
    "decision": "intelligent_decision_support.csv",
    "pricing_comparison": "final_pricing_comparison.csv",
    "zones": "zone-information.csv",
    "xgb": "spatial_model_test_predictions.csv",
    "sensitivity": "sensitivity_analysis.csv",
    "neighbours": "nearest_alternative_zones.csv",
    "stgnn_metrics": "residual_stgnn_comparison.csv",
    "shap": "shap_feature_importance.csv",
}

@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    try:
        return pd.read_csv(path_string)
    except Exception:
        return None

def load_data(key):
    path = BASE_DIR / FILE_NAMES[key]
    if not path.exists():
        return None
    return load_csv(str(path), path.stat().st_mtime)

D = {key: load_data(key) for key in FILE_NAMES}

pricing = D["pricing"]
decision = D["decision"]
pricing_comparison = D["pricing_comparison"]
zones = D["zones"]
xgb = D["xgb"]
sensitivity = D["sensitivity"]
neighbours = D["neighbours"]
stgnn_metrics = D["stgnn_metrics"]
shap_data = D["shap"]

def find_col(df, candidates):
    if df is None:
        return None
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None

def metric_value(df, candidates, default=np.nan):
    col = find_col(df, candidates)
    if df is None or col is None or df.empty:
        return default
    return pd.to_numeric(df[col], errors="coerce").dropna().mean()

def parse_time(df):
    if df is None or "time" not in df.columns:
        return df
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out

pricing = parse_time(pricing)
decision = parse_time(decision)
xgb = parse_time(xgb)

def money(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"

def pct(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}%"

def section_intro(title, explanation):
    st.title(title)
    st.write(explanation)

def missing_file_message(key):
    filename = FILE_NAMES[key]
    st.info(f"To display this part, add **{filename}** to the same GitHub folder as `app.py`.")

# ---------------------------------------------------------
# SIDEBAR / NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("## ⚡ UrbanEV")
st.sidebar.caption("Smart Charging Network Analytics")
page = st.sidebar.radio(
    "Dashboard sections",
    [
        "1. Dataset Overview",
        "2. Occupancy Analysis",
        "3. Congestion Prediction",
        "4. Occupancy Forecasting",
        "5. Spatial Visualisation",
        "6. Dynamic Pricing",
        "7. Demand Redistribution",
        "8. Model & Scenario Analysis",
    ]
)
st.sidebar.divider()
st.sidebar.caption("UrbanEV Shenzhen dataset")
st.sidebar.caption("September 2022 – February 2023")

st.markdown("""
<div class="hero">
  <h1>UrbanEV Smart Charging Analytics</h1>
  <p>Predictive Congestion Modeling and Dynamic Pricing Optimization for Electric Vehicle Charging Networks</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. DATASET OVERVIEW
# ---------------------------------------------------------
if page == "1. Dataset Overview":
    section_intro(
        "Dataset Overview",
        "Explore the UrbanEV charging network, the study period and the information used to analyse charging activity."
    )

    total_zones = zones["TAZID"].nunique() if zones is not None and "TAZID" in zones.columns else 275
    start_date, end_date = "01 Sep 2022", "28 Feb 2023"
    if pricing is not None and "time" in pricing.columns and pricing["time"].notna().any():
        start_date = pricing["time"].min().strftime("%d %b %Y")
        end_date = pricing["time"].max().strftime("%d %b %Y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Charging zones", f"{total_zones:,}")
    c2.metric("Study period", "6 months")
    c3.metric("Time coverage", f"{start_date} – {end_date}")
    c4.metric("Hourly observations", "4,344")

    st.subheader("What information is included?")
    variable_info = pd.DataFrame({
        "Variable": [
            "Occupancy", "Charging volume", "11 kW charging volume",
            "Charging duration", "Electricity price", "Service price",
            "Zone information"
        ],
        "What it represents": [
            "Number of charging sessions / occupied charging activity in a zone",
            "Charging energy volume recorded for a zone",
            "Charging volume associated with 11 kW charging",
            "Charging duration",
            "Electricity cost",
            "Charging service price",
            "Zone location and geographical characteristics"
        ]
    })
    st.dataframe(variable_info, hide_index=True, use_container_width=True)

    st.subheader("Data files available to this dashboard")
    file_status = pd.DataFrame([
        {"Dataset": FILE_NAMES[k], "Status": "Available" if D[k] is not None else "Not uploaded"}
        for k in FILE_NAMES
    ])
    st.dataframe(file_status, hide_index=True, use_container_width=True)

    st.info(
        "The study combines hourly charging measurements with geographical information "
        "for 275 Shenzhen traffic analysis zones (TAZs)."
    )

# ---------------------------------------------------------
# 2. OCCUPANCY ANALYSIS
# ---------------------------------------------------------
elif page == "2. Occupancy Analysis":
    section_intro(
        "Occupancy Analysis",
        "Understand how charging activity varies over time and across zones. These charts use the occupancy observations available in the uploaded prediction dataset."
    )

    source = xgb if xgb is not None else pricing
    occ_col = find_col(source, ["occupancy"])
    time_col = find_col(source, ["time"])
    zone_col = find_col(source, ["zone_id", "TAZID"])

    if source is None or occ_col is None:
        st.warning("Occupancy data is unavailable. Add spatial_model_test_predictions.csv or dynamic_pricing_simulation.csv.")
    else:
        source = source.copy()
        source[occ_col] = pd.to_numeric(source[occ_col], errors="coerce")
        source = source.dropna(subset=[occ_col])

        a, b, c = st.columns(3)
        a.metric("Average occupancy", f"{source[occ_col].mean():.2f}")
        b.metric("Median occupancy", f"{source[occ_col].median():.2f}")
        c.metric("Maximum occupancy", f"{source[occ_col].max():,.0f}")

        st.subheader("Occupancy distribution")
        fig = px.histogram(
            source, x=occ_col, nbins=45,
            labels={occ_col: "Occupancy", "count": "Number of observations"},
            title="Distribution of charging-zone occupancy"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if time_col and source[time_col].notna().any():
            source["Hour"] = source[time_col].dt.hour
            hourly = source.groupby("Hour", as_index=False)[occ_col].mean()
            st.subheader("Average occupancy by hour")
            fig = px.line(
                hourly, x="Hour", y=occ_col, markers=True,
                labels={"Hour": "Hour of day", occ_col: "Average occupancy"},
                title="Hourly charging pattern"
            )
            st.plotly_chart(fig, use_container_width=True)

            source["Date"] = source[time_col].dt.date
            daily = source.groupby("Date", as_index=False)[occ_col].mean()
            st.subheader("Occupancy over the study test period")
            fig = px.line(
                daily, x="Date", y=occ_col,
                labels={"Date": "Date", occ_col: "Average occupancy"},
                title="Daily average occupancy"
            )
            st.plotly_chart(fig, use_container_width=True)

        if zone_col:
            zone_avg = source.groupby(zone_col, as_index=False)[occ_col].mean()
            zone_avg = zone_avg.sort_values(occ_col, ascending=False).head(15)
            st.subheader("Zones with the highest average occupancy")
            fig = px.bar(
                zone_avg, x=zone_col, y=occ_col,
                labels={zone_col: "Charging zone", occ_col: "Average occupancy"},
                title="Top 15 zones by average occupancy"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: these charts reflect the records present in the uploaded test/prediction dataset, not the complete six-month raw dataset.")

# ---------------------------------------------------------
# 3. CONGESTION PREDICTION
# ---------------------------------------------------------
elif page == "3. Congestion Prediction":
    section_intro(
        "Congestion Prediction",
        "The selected XGBoost model estimates the likelihood of high occupancy in the next hour for each charging zone. Use the filters to inspect the prediction records."
    )

    if xgb is None:
        missing_file_message("xgb")
    else:
        prob_col = find_col(xgb, ["predicted_congestion_probability"])
        actual_col = find_col(xgb, ["actual_high_occupancy"])
        pred_col = find_col(xgb, ["predicted_high_occupancy"])
        zone_col = find_col(xgb, ["zone_id", "TAZID"])
        time_col = find_col(xgb, ["time"])
        occ_col = find_col(xgb, ["occupancy"])

        view = xgb.copy()
        if time_col:
            view[time_col] = pd.to_datetime(view[time_col], errors="coerce")
            dates = view[time_col].dropna()
            if not dates.empty:
                selected_dates = st.date_input(
                    "Select date range",
                    value=(dates.min().date(), dates.max().date()),
                    key="congestion_dates"
                )
                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    view = view[
                        view[time_col].dt.date.between(selected_dates[0], selected_dates[1])
                    ]

        if zone_col:
            zone_options = sorted(view[zone_col].dropna().unique().tolist())
            selected_zone = st.selectbox("Charging zone", ["All zones"] + zone_options, key="congestion_zone")
            if selected_zone != "All zones":
                view = view[view[zone_col] == selected_zone]

        if prob_col:
            view[prob_col] = pd.to_numeric(view[prob_col], errors="coerce")
            avg_prob = view[prob_col].mean() * 100
            high_prob_count = (view[prob_col] >= 0.5).sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Average predicted probability", pct(avg_prob))
            m2.metric("Records with probability ≥ 50%", f"{high_prob_count:,}")
            m3.metric("Records displayed", f"{len(view):,}")

            st.subheader("Predicted congestion probability")
            fig = px.histogram(
                view, x=prob_col, nbins=40,
                labels={prob_col: "Predicted probability"},
                title="Distribution of predicted high-occupancy probability"
            )
            st.plotly_chart(fig, use_container_width=True)

        if actual_col and pred_col:
            actual_rate = pd.to_numeric(view[actual_col], errors="coerce").mean() * 100
            predicted_rate = pd.to_numeric(view[pred_col], errors="coerce").mean() * 100
            compare = pd.DataFrame({
                "Type": ["Actual high occupancy", "Predicted high occupancy"],
                "Percentage": [actual_rate, predicted_rate]
            })
            st.subheader("Actual and predicted high-occupancy records")
            fig = px.bar(compare, x="Type", y="Percentage", text_auto=".2f",
                         labels={"Percentage": "Share of records (%)"})
            st.plotly_chart(fig, use_container_width=True)

        display_cols = [c for c in [
            time_col, zone_col, occ_col, prob_col, pred_col, actual_col
        ] if c is not None]
        st.subheader("Zone-level prediction records")
        st.dataframe(view[display_cols].head(500), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 4. OCCUPANCY FORECASTING
# ---------------------------------------------------------
elif page == "4. Occupancy Forecasting":
    section_intro(
        "Occupancy Forecasting",
        "The Residual ST-GNN uses recent occupancy patterns and spatial connections between nearby zones to estimate occupancy for the next hour."
    )

    st.subheader("Residual ST-GNN test performance")
    st.metric("Mean Absolute Error (MAE)", "1.1300")
    st.metric("Root Mean Squared Error (RMSE)", "2.4041")
    st.metric("R-squared (R²)", "0.9862")
    st.caption("Reported test metrics for the selected Residual ST-GNN model.")

    if stgnn_metrics is not None:
        st.subheader("Model comparison")
        st.dataframe(stgnn_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("residual_stgnn_comparison.csv is not uploaded. The selected model's reported metrics are shown above.")

    st.warning(
        "The current GitHub file list does not include a row-level ST-GNN forecast file. "
        "Therefore this section shows model performance, but cannot yet plot actual versus forecast occupancy over time."
    )

# ---------------------------------------------------------
# 5. SPATIAL VISUALISATION
# ---------------------------------------------------------
elif page == "5. Spatial Visualisation":
    section_intro(
        "Spatial Visualisation",
        "Explore the geographical distribution of charging zones and view occupancy or congestion information at zone level."
    )

    if zones is None:
        missing_file_message("zones")
    else:
        lat_col = find_col(zones, ["latitude", "lat"])
        lon_col = find_col(zones, ["longitude", "lon", "long"])
        zone_col = find_col(zones, ["TAZID", "zone_id"])
        if not lat_col or not lon_col:
            st.error("The zone information file needs latitude and longitude columns.")
        else:
            map_data = zones.copy()
            if zone_col and xgb is not None:
                x_zone = find_col(xgb, ["zone_id", "TAZID"])
                x_occ = find_col(xgb, ["occupancy"])
                x_prob = find_col(xgb, ["predicted_congestion_probability"])
                if x_zone and x_occ:
                    summary = xgb.groupby(x_zone).agg(
                        avg_occupancy=(x_occ, "mean")
                    ).reset_index()
                    if x_prob:
                        probability = xgb.groupby(x_zone)[x_prob].mean().reset_index()
                        summary = summary.merge(probability, on=x_zone, how="left")
                    map_data = map_data.merge(summary, left_on=zone_col, right_on=x_zone, how="left")

            map_data = map_data.dropna(subset=[lat_col, lon_col])
            if FOLIUM_AVAILABLE and not map_data.empty:
                center = [map_data[lat_col].mean(), map_data[lon_col].mean()]
                fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
                for _, row in map_data.iterrows():
                    zone_name = row[zone_col] if zone_col else "Charging zone"
                    occupancy = row.get("avg_occupancy", np.nan)
                    probability = row.get("predicted_congestion_probability", np.nan)
                    if pd.notna(probability):
                        colour = "red" if probability >= 0.5 else "green"
                    else:
                        colour = "blue"
                    popup_text = f"Zone: {zone_name}"
                    if pd.notna(occupancy):
                        popup_text += f"<br>Average occupancy: {occupancy:.2f}"
                    if pd.notna(probability):
                        popup_text += f"<br>Predicted congestion probability: {probability:.3f}"
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=5,
                        color=colour,
                        fill=True,
                        fill_opacity=0.75,
                        popup=folium.Popup(popup_text, max_width=260)
                    ).add_to(fmap)
                st_folium(fmap, use_container_width=True, height=570)
                st.caption("Red markers indicate average predicted congestion probability of 0.50 or higher; green markers are below 0.50. Blue markers indicate no joined prediction.")
            else:
                st.warning("Install folium and streamlit-folium to display the interactive map.")
                st.map(map_data[[lat_col, lon_col]].rename(columns={lat_col: "latitude", lon_col: "longitude"}))
            st.subheader("Charging zone information")
            st.dataframe(zones, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 6. DYNAMIC PRICING
# ---------------------------------------------------------
elif page == "6. Dynamic Pricing":
    section_intro(
        "Dynamic Pricing",
        "Compare the existing service price with the simulated dynamic price. The advanced strategy also incorporates forecast occupancy, relative demand and electricity cost."
    )

    if pricing_comparison is not None:
        st.subheader("Comparison of pricing strategies")
        st.dataframe(pricing_comparison, use_container_width=True, hide_index=True)
        strategy_col = find_col(pricing_comparison, ["Pricing Strategy"])
        mean_col = find_col(pricing_comparison, ["Mean Price"])
        if strategy_col and mean_col:
            fig = px.bar(
                pricing_comparison, x=strategy_col, y=mean_col,
                text_auto=".4f", title="Mean service price by strategy",
                labels={strategy_col: "Pricing strategy", mean_col: "Mean service price"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The comparison uses mean service prices, not total customer prices.")

    if pricing is None:
        missing_file_message("pricing")
    else:
        base_col = find_col(pricing, ["service_price", "base_price"])
        dynamic_col = find_col(pricing, ["dynamic_price"])
        change_col = find_col(pricing, ["price_increase_pct"])
        prob_col = find_col(pricing, ["predicted_congestion_probability"])
        zone_col = find_col(pricing, ["zone_id"])
        time_col = find_col(pricing, ["time"])

        if base_col and dynamic_col:
            p = pricing.copy()
            p[base_col] = pd.to_numeric(p[base_col], errors="coerce")
            p[dynamic_col] = pd.to_numeric(p[dynamic_col], errors="coerce")
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean existing service price", money(p[base_col].mean()))
            m2.metric("Mean dynamic service price", money(p[dynamic_col].mean()))
            m3.metric("Mean price difference", money((p[dynamic_col] - p[base_col]).mean()))

            st.subheader("Price comparison by record")
            sample = p.head(1500)
            long = sample.melt(
                id_vars=[c for c in [time_col, zone_col] if c],
                value_vars=[base_col, dynamic_col],
                var_name="Price type", value_name="Service price"
            )
            fig = px.box(long, x="Price type", y="Service price",
                         title="Distribution of existing and dynamic service prices")
            st.plotly_chart(fig, use_container_width=True)

            if change_col:
                st.subheader("Price adjustment distribution")
                p[change_col] = pd.to_numeric(p[change_col], errors="coerce")
                fig = px.histogram(
                    p, x=change_col, nbins=30,
                    labels={change_col: "Price increase / adjustment"},
                    title="Distribution of dynamic price adjustments"
                )
                st.plotly_chart(fig, use_container_width=True)

            if prob_col and change_col:
                st.subheader("Congestion probability and price adjustment")
                fig = px.scatter(
                    p.sample(min(5000, len(p)), random_state=42),
                    x=prob_col, y=change_col,
                    opacity=0.45,
                    labels={prob_col: "Predicted congestion probability",
                            change_col: "Price adjustment"},
                    title="Relationship between predicted congestion and price adjustment"
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. DEMAND REDISTRIBUTION
# ---------------------------------------------------------
elif page == "7. Demand Redistribution":
    section_intro(
        "Demand Redistribution",
        "This scenario analysis examines how some charging demand could move from highly occupied zones to nearby alternatives with lower predicted occupancy and lower prices."
    )

    if decision is not None:
        risk_col = find_col(decision, ["congestion_risk"])
        action_col = find_col(decision, ["recommended_action"])
        if risk_col:
            st.subheader("Congestion risk categories")
            risk_counts = decision[risk_col].value_counts().rename_axis("Risk category").reset_index(name="Records")
            fig = px.bar(risk_counts, x="Risk category", y="Records",
                         title="Records by congestion risk category")
            st.plotly_chart(fig, use_container_width=True)
        if action_col:
            st.subheader("Recommended operational responses")
            actions = decision[action_col].value_counts().rename_axis("Recommended action").reset_index(name="Records")
            fig = px.bar(actions, x="Records", y="Recommended action", orientation="h",
                         title="Decision-support recommendations")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Decision-support preview")
        cols = [c for c in [
            "time", "zone_id", "occupancy", "congestion_risk",
            "congestion_probability_pct", "service_price", "dynamic_price",
            "recommended_action", "operational_response"
        ] if c in decision.columns]
        st.dataframe(decision[cols].head(300), use_container_width=True, hide_index=True)
    else:
        missing_file_message("decision")

    if neighbours is not None:
        st.subheader("Nearby alternative zones")
        st.dataframe(neighbours.head(200), use_container_width=True, hide_index=True)
    else:
        missing_file_message("neighbours")

    st.info(
        "Demand redistribution results are simulated using assumed price sensitivity and zone-capacity rules. "
        "They should be interpreted as scenario results, not measured changes in actual customer behaviour."
    )

# ---------------------------------------------------------
# 8. MODEL & SCENARIO ANALYSIS
# ---------------------------------------------------------
elif page == "8. Model & Scenario Analysis":
    section_intro(
        "Model & Scenario Analysis",
        "Review the selected prediction models, understand influential features and compare alternative price-sensitivity assumptions."
    )

    st.subheader("XGBoost congestion classification")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Value": [0.985986, 0.881671, 0.987155, 0.931436, 0.999050]
    })
    st.dataframe(metrics.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.caption("Reported test metrics for the safer spatial XGBoost configuration.")

    st.subheader("Residual ST-GNN occupancy forecasting")
    st.dataframe(pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [1.130034, 2.404112, 0.986153]
    }).style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)

    if shap_data is not None:
        st.subheader("Factors influencing XGBoost predictions (SHAP)")
        feature_col = find_col(shap_data, ["feature", "features", "feature_name"])
        importance_col = find_col(shap_data, ["importance", "mean_abs_shap", "shap_importance", "mean_shap_value"])
        if feature_col and importance_col:
            shap_plot = shap_data.sort_values(importance_col, ascending=True)
            fig = px.bar(shap_plot, x=importance_col, y=feature_col, orientation="h",
                         title="SHAP feature importance",
                         labels={importance_col: "Mean absolute SHAP value", feature_col: "Feature"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(shap_data, use_container_width=True, hide_index=True)
    else:
        missing_file_message("shap")

    if sensitivity is not None:
        st.subheader("Price sensitivity scenarios")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)

        sens_col = find_col(sensitivity, ["price_sensitivity", "sensitivity"])
        before_col = find_col(sensitivity, ["congestion_before_pct", "congestion_before", "before_congestion"])
        after_col = find_col(sensitivity, ["congestion_after_pct", "congestion_after", "after_congestion"])
        if sens_col and before_col and after_col:
            chart = sensitivity.melt(
                id_vars=[sens_col], value_vars=[before_col, after_col],
                var_name="Scenario", value_name="Congestion (%)"
            )
            fig = px.line(chart, x=sens_col, y="Congestion (%)", color="Scenario",
                          markers=True, title="Congestion before and after redistribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sensitivity file loaded. Check its column headings if the scenario chart is not displayed.")
    else:
        missing_file_message("sensitivity")

    st.warning(
        "Model metrics describe performance on the reported test data. "
        "Pricing and demand redistribution are policy simulations and do not establish real-world causal effects."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "UrbanEV Smart Charging Analytics | Academic capstone project | "
    "UrbanEV Shenzhen dataset"
)

