import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

st.set_page_config(
    page_title="UrbanEV | Interactive Charging Network",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e6eaf0;
    padding: 12px 14px; border-radius: 12px;
}
.hero {
    padding: 1.2rem 1.5rem; border-radius: 14px;
    background: linear-gradient(115deg, #102d46, #176b72);
    color: white; margin-bottom: 1rem;
}
.hero h1 {color: white; margin: 0 0 .3rem 0;}
.hero p {color: #e5f2f4; margin: 0;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "pricing": "dynamic_pricing_simulation.csv",
    "decision": "intelligent_decision_support.csv",
    "comparison": "final_pricing_comparison.csv",
    "zones": "zone-information.csv",
    "xgb": "spatial_model_test_predictions.csv",
    "sensitivity": "sensitivity_analysis.csv",
    "neighbours": "nearest_alternative_zones.csv",
    "stgnn": "residual_stgnn_comparison.csv",
    "shap": "shap_feature_importance.csv",
}

@st.cache_data(show_spinner=False)
def read_csv(filename, modified):
    try:
        return pd.read_csv(filename)
    except Exception:
        return None

def load(key):
    path = BASE_DIR / FILES[key]
    return read_csv(str(path), path.stat().st_mtime) if path.exists() else None

data = {k: load(k) for k in FILES}
pricing = data["pricing"]
decision = data["decision"]
comparison = data["comparison"]
zones = data["zones"]
xgb = data["xgb"]
sensitivity = data["sensitivity"]
neighbours = data["neighbours"]
stgnn_metrics = data["stgnn"]
shap_data = data["shap"]

def col(df, names):
    if df is None:
        return None
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for name in names:
        key = name.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None

def numeric(df, column):
    if df is not None and column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(dtype=float)

def money(x):
    return "N/A" if pd.isna(x) else f"{x:.4f}"

def pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def add_time(df):
    if df is None:
        return None
    out = df.copy()
    t = col(out, ["time", "timestamp", "datetime"])
    if t:
        out[t] = pd.to_datetime(out[t], errors="coerce")
    return out

pricing = add_time(pricing)
xgb = add_time(xgb)
decision = add_time(decision)

# Standardise the zone identifier for internal joins.
def standard_zone(df):
    if df is None:
        return None
    out = df.copy()
    z = col(out, ["zone_id", "TAZID", "zone", "taz_id"])
    if z and z != "zone_id":
        out = out.rename(columns={z: "zone_id"})
    if "zone_id" in out.columns:
        out["zone_id"] = out["zone_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out

zones = standard_zone(zones)
xgb = standard_zone(xgb)
pricing = standard_zone(pricing)
decision = standard_zone(decision)

# Join zone-level prediction summaries to the geographical information.
map_data = zones.copy() if zones is not None else pd.DataFrame()
if not map_data.empty and "zone_id" in map_data.columns:
    if xgb is not None and "zone_id" in xgb.columns:
        occ = col(xgb, ["occupancy"])
        prob = col(xgb, ["predicted_congestion_probability"])
        pred = col(xgb, ["predicted_high_occupancy"])
        aggs = {}
        if occ: aggs[occ] = "mean"
        if prob: aggs[prob] = "mean"
        if pred: aggs[pred] = "mean"
        if aggs:
            summary = xgb.groupby("zone_id").agg(aggs).reset_index()
            rename = {}
            if occ: rename[occ] = "average_occupancy"
            if prob: rename[prob] = "congestion_probability"
            if pred: rename[pred] = "predicted_high_occupancy"
            summary = summary.rename(columns=rename)
            map_data = map_data.merge(summary, on="zone_id", how="left")

zone_options = []
for source in [zones, xgb, pricing]:
    if source is not None and "zone_id" in source.columns:
        zone_options.extend(source["zone_id"].dropna().astype(str).unique().tolist())
zone_options = sorted(set(zone_options), key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))

st.sidebar.markdown("## ⚡ UrbanEV")
st.sidebar.caption("Interactive EV charging network decision support")
pages = [
    "Interactive Zone Explorer",
    "Network Overview",
    "Congestion Prediction",
    "Occupancy Forecasting",
    "Dynamic Pricing",
    "Demand Redistribution",
    "Model & Scenario Analysis",
]
page = st.sidebar.radio("Application", pages)

if zone_options:
    if "selected_zone" not in st.session_state or st.session_state.selected_zone not in zone_options:
        st.session_state.selected_zone = zone_options[0]
    selected_zone = st.sidebar.selectbox(
        "Select charging zone",
        zone_options,
        index=zone_options.index(st.session_state.selected_zone),
    )
    st.session_state.selected_zone = selected_zone
else:
    selected_zone = None
    st.sidebar.error("No zone IDs found. Check that zone-information.csv or prediction CSV is uploaded.")

st.sidebar.caption("UrbanEV Shenzhen | Sep 2022 – Feb 2023")

st.markdown("""
<div class="hero">
<h1>UrbanEV Smart Charging Network</h1>
<p>Zone-level congestion prediction, occupancy forecasting and dynamic pricing simulation</p>
</div>
""", unsafe_allow_html=True)

def zone_filter(df, zone=selected_zone):
    if df is None or zone is None or "zone_id" not in df.columns:
        return pd.DataFrame()
    return df[df["zone_id"].astype(str) == str(zone)].copy()

def show_zone_metrics():
    zpred = zone_filter(xgb)
    zprice = zone_filter(pricing)
    occupancy_col = col(zpred, ["occupancy"])
    prob_col = col(zpred, ["predicted_congestion_probability"])
    service_col = col(zprice, ["service_price"])
    optimized_col = col(zprice, ["optimized_service_price", "dynamic_price"])
    a, b, c, d = st.columns(4)
    a.metric("Selected zone", str(selected_zone) if selected_zone else "N/A")
    a1 = numeric(zpred, occupancy_col).mean() if occupancy_col else np.nan
    b.metric("Average occupancy", f"{a1:.2f}" if pd.notna(a1) else "N/A")
    p = numeric(zpred, prob_col).mean() * 100 if prob_col else np.nan
    c.metric("Mean congestion probability", pct(p))
    if service_col and optimized_col:
        diff = (numeric(zprice, optimized_col) - numeric(zprice, service_col)).mean()
        d.metric("Mean service-price change", money(diff))
    else:
        d.metric("Pricing records", f"{len(zprice):,}" if not zprice.empty else "N/A")

def draw_zone_map():
    if map_data.empty:
        st.warning("Zone map is unavailable. Upload zone-information.csv with TAZID, latitude and longitude.")
        return
    lat = col(map_data, ["latitude", "lat"])
    lon = col(map_data, ["longitude", "lon", "long"])
    if not lat or not lon:
        st.warning("The zone file must contain latitude and longitude columns.")
        return
    md = map_data.dropna(subset=[lat, lon]).copy()
    if md.empty:
        st.warning("No valid zone coordinates were found.")
        return
    if FOLIUM_AVAILABLE:
        center = [md[lat].mean(), md[lon].mean()]
        fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
        for _, row in md.iterrows():
            zid = str(row["zone_id"])
            prob = row.get("congestion_probability", np.nan)
            occ = row.get("average_occupancy", np.nan)
            colour = "red" if pd.notna(prob) and prob >= 0.5 else "green"
            if pd.isna(prob):
                colour = "blue"
            if zid == str(selected_zone):
                colour = "purple"
            popup = f"Zone: {zid}<br>Average occupancy: {occ:.2f}" if pd.notna(occ) else f"Zone: {zid}"
            if pd.notna(prob):
                popup += f"<br>Mean congestion probability: {prob:.3f}"
            folium.CircleMarker(
                location=[row[lat], row[lon]], radius=7 if zid == str(selected_zone) else 5,
                color=colour, fill=True, fill_opacity=0.8,
                tooltip=f"Zone {zid}", popup=folium.Popup(popup, max_width=260)
            ).add_to(fmap)
        result = st_folium(
            fmap, use_container_width=True, height=560,
            key="urbanev_zone_map", returned_objects=["last_object_clicked"]
        )
        clicked = result.get("last_object_clicked") if isinstance(result, dict) else None
        if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
            distances = (
                (md[lat].astype(float) - float(clicked["lat"])) ** 2
                + (md[lon].astype(float) - float(clicked["lng"])) ** 2
            )
            nearest_index = distances.idxmin()
            clicked_zone = str(md.loc[nearest_index, "zone_id"])
            if clicked_zone in zone_options and clicked_zone != str(selected_zone):
                st.session_state.selected_zone = clicked_zone
                st.rerun()
        st.caption("Purple = selected zone; red = mean predicted congestion probability ≥ 0.50; green = below 0.50; blue = no joined prediction. Click a marker to select its zone.")
    else:
        st.warning("Install folium and streamlit-folium to enable clickable zone selection.")
        st.map(md[[lat, lon]].rename(columns={lat: "latitude", lon: "longitude"}))

if page == "Interactive Zone Explorer":
    st.title("Interactive Zone Explorer")
    st.write("Choose a charging zone using the sidebar dropdown or click a marker on the map. The zone-specific information below updates to your selection.")
    if selected_zone is None:
        st.error("No zones are available. Upload the zone and prediction CSV files.")
    else:
        show_zone_metrics()
        st.subheader("Select a zone on the map")
        draw_zone_map()
        st.divider()
        st.subheader(f"Zone {selected_zone} | Detailed information")
        zinfo = zone_filter(zones)
        if not zinfo.empty:
            st.dataframe(zinfo, use_container_width=True, hide_index=True)
        zpred = zone_filter(xgb)
        if not zpred.empty:
            tcol = col(zpred, ["time"])
            ocol = col(zpred, ["occupancy"])
            pcol = col(zpred, ["predicted_congestion_probability"])
            predcol = col(zpred, ["predicted_high_occupancy"])
            if tcol:
                zpred = zpred.sort_values(tcol)
            if tcol and ocol:
                st.subheader("Observed occupancy over time")
                st.plotly_chart(px.line(zpred, x=tcol, y=ocol, title=f"Zone {selected_zone}: occupancy history"), use_container_width=True)
            if tcol and pcol:
                st.subheader("Next-hour congestion probability")
                chart = zpred.copy()
                chart[pcol] = numeric(chart, pcol) * 100
                st.plotly_chart(px.line(chart, x=tcol, y=pcol, title="Predicted probability of high occupancy", labels={pcol: "Probability (%)"}), use_container_width=True)
            if predcol:
                st.metric("Predicted high-occupancy records", f"{int(numeric(zpred, predcol).sum())}")
        else:
            st.info("No XGBoost prediction records were found for this zone.")
        zprice = zone_filter(pricing)
        base = col(zprice, ["service_price"])
        opt = col(zprice, ["optimized_service_price", "dynamic_price"])
        timec = col(zprice, ["time"])
        if not zprice.empty and base and opt:
            st.subheader("Fixed and optimised service price")
            if timec:
                zprice = zprice.sort_values(timec)
                price_long = zprice.melt(id_vars=[timec], value_vars=[base, opt], var_name="Pricing strategy", value_name="Service price")
                st.plotly_chart(px.line(price_long, x=timec, y="Service price", color="Pricing strategy", title="Zone-level service-price comparison"), use_container_width=True)
            st.dataframe(zprice[[c for c in [timec, base, opt, col(zprice, ["price_adjustment"])] if c]], use_container_width=True, hide_index=True)

elif page == "Network Overview":
    st.title("Network Overview")
    st.write("A network-level view of the UrbanEV charging zones and observed test-period activity.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Charging zones", f"{zones['zone_id'].nunique():,}" if zones is not None and "zone_id" in zones else "N/A")
    c2.metric("Prediction records", f"{len(xgb):,}" if xgb is not None else "N/A")
    c3.metric("Pricing simulation records", f"{len(pricing):,}" if pricing is not None else "N/A")
    if xgb is not None:
        occ = col(xgb, ["occupancy"])
        timec = col(xgb, ["time"])
        zc = col(xgb, ["zone_id"])
        if occ:
            st.subheader("Occupancy distribution")
            st.plotly_chart(px.histogram(xgb, x=occ, nbins=45), use_container_width=True)
        if occ and timec:
            hourly = xgb.assign(hour=xgb[timec].dt.hour).groupby("hour", as_index=False)[occ].mean()
            st.subheader("Average occupancy by hour")
            st.plotly_chart(px.line(hourly, x="hour", y=occ, markers=True), use_container_width=True)
        if occ and zc:
            top = xgb.groupby(zc)[occ].mean().nlargest(15).reset_index()
            st.subheader("15 zones with highest average occupancy")
            st.plotly_chart(px.bar(top, x=zc, y=occ), use_container_width=True)
    st.subheader("Zone map")
    draw_zone_map()

elif page == "Congestion Prediction":
    st.title("Zone-level Congestion Prediction")
    st.write("XGBoost estimates the probability that occupancy in the selected zone will be high in the next hour.")
    view = zone_filter(xgb)
    if view.empty:
        st.warning("No prediction data found for the selected zone. Check spatial_model_test_predictions.csv.")
    else:
        prob = col(view, ["predicted_congestion_probability"])
        actual = col(view, ["actual_high_occupancy"])
        pred = col(view, ["predicted_high_occupancy"])
        timec = col(view, ["time"])
        occ = col(view, ["occupancy"])
        if prob:
            mean_prob = numeric(view, prob).mean() * 100
            st.metric("Mean predicted congestion probability", pct(mean_prob))
            if timec:
                chart = view.copy()
                chart[prob] = numeric(chart, prob) * 100
                st.plotly_chart(px.line(chart, x=timec, y=prob, title=f"Zone {selected_zone}: congestion probability", labels={prob: "Probability (%)"}), use_container_width=True)
        a, b = st.columns(2)
        if actual:
            a.metric("Actual high-occupancy share", pct(numeric(view, actual).mean() * 100))
        if pred:
            b.metric("Predicted high-occupancy share", pct(numeric(view, pred).mean() * 100))
        if timec and occ:
            st.plotly_chart(px.line(view, x=timec, y=occ, title="Observed occupancy for selected zone"), use_container_width=True)
        st.subheader("Prediction records")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)

elif page == "Occupancy Forecasting":
    st.title("Next-hour Occupancy Forecast")
    st.write("The Residual ST-GNN model uses the recent occupancy sequence and spatial relationships between zones.")
    st.info("The current project files include the Residual ST-GNN model and its evaluation metrics, but do not yet include a zone-by-zone forecast CSV. The app will not invent a forecast for the selected zone.")
    if stgnn_metrics is not None:
        st.subheader("ST-GNN model comparison")
        st.dataframe(stgnn_metrics, use_container_width=True, hide_index=True)
    st.subheader("Selected Residual ST-GNN test performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", "1.1300")
    m2.metric("RMSE", "2.4041")
    m3.metric("R²", "0.9862")
    st.caption("To display a zone-specific actual-versus-forecast chart, add the exported stgnn_occupancy_forecasts.csv with time, zone_id, actual_occupancy and predicted_occupancy columns.")

elif page == "Dynamic Pricing":
    st.title("Interactive Dynamic Pricing")
    st.write("Inspect fixed and optimised service prices for the selected charging zone.")
    view = zone_filter(pricing)
    base = col(view, ["service_price"])
    optimized = col(view, ["optimized_service_price", "dynamic_price"])
    total = col(view, ["optimized_total_price"])
    adjustment = col(view, ["price_adjustment", "price_increase_pct"])
    timec = col(view, ["time"])
    if view.empty or not base or not optimized:
        st.warning("Selected-zone pricing data is unavailable. Check dynamic_pricing_simulation.csv and its service_price / optimized_service_price columns.")
    else:
        old_mean = numeric(view, base).mean()
        new_mean = numeric(view, optimized).mean()
        change = ((new_mean / old_mean) - 1) * 100 if old_mean else np.nan
        a, b, c = st.columns(3)
        a.metric("Mean fixed service price", money(old_mean))
        b.metric("Mean optimised service price", money(new_mean))
        c.metric("Mean price change", pct(change))
        if timec:
            view = view.sort_values(timec)
            long = view.melt(id_vars=[timec], value_vars=[base, optimized], var_name="Strategy", value_name="Service price")
            st.plotly_chart(px.line(long, x=timec, y="Service price", color="Strategy", title=f"Zone {selected_zone}: service-price comparison"), use_container_width=True)
        if adjustment:
            st.plotly_chart(px.histogram(view, x=adjustment, nbins=30, title="Price adjustment distribution for selected zone"), use_container_width=True)
        if total:
            st.metric("Mean optimised total price (electricity + service)", money(numeric(view, total).mean()))
        st.subheader("Pricing records")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    if comparison is not None:
        st.subheader("Network-wide pricing strategy comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        strategy = col(comparison, ["Pricing Strategy"])
        meanprice = col(comparison, ["Mean Price"])
        if strategy and meanprice:
            st.plotly_chart(px.bar(comparison, x=strategy, y=meanprice, title="Mean service price across strategies"), use_container_width=True)

elif page == "Demand Redistribution":
    st.title("Nearby Alternative Zones & Demand Redistribution")
    st.write("Explore the simulated movement of demand away from a congested zone toward nearby alternatives.")
    view = zone_filter(pricing)
    if not view.empty:
        origin = col(view, ["origin_zone"])
        outgoing = col(view, ["outgoing_volume", "actual_outgoing_volume"])
        incoming = col(view, ["incoming_volume"])
        adjusted = col(view, ["adjusted_volume"])
        a, b, c = st.columns(3)
        a.metric("Mean outgoing volume", money(numeric(view, outgoing).mean()) if outgoing else "N/A")
        b.metric("Mean incoming volume", money(numeric(view, incoming).mean()) if incoming else "N/A")
        c.metric("Mean adjusted volume", money(numeric(view, adjusted).mean()) if adjusted else "N/A")
        st.subheader("Selected-zone redistribution records")
        useful = [c for c in [col(view, ["time"]), "zone_id", origin, outgoing, incoming, adjusted] if c and c in view.columns]
        st.dataframe(view[useful].head(500), use_container_width=True, hide_index=True)
    if neighbours is not None:
        st.subheader("Nearest alternative-zone table")
        nview = standard_zone(neighbours)
        st.dataframe(nview.head(300), use_container_width=True, hide_index=True)
    else:
        st.info("nearest_alternative_zones.csv is not available. The app can still show redistribution fields included in dynamic_pricing_simulation.csv.")
    st.warning("Demand redistribution and congestion changes are scenario simulations based on assumed price sensitivity and capacity rules, not measured customer behaviour.")

elif page == "Model & Scenario Analysis":
    st.title("Model & Scenario Analysis")
    st.write("Review the reported model performance, influential features and alternative price-sensitivity scenarios.")
    st.subheader("XGBoost congestion classification")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Value": [0.985986, 0.881671, 0.987155, 0.931436, 0.999050]
    })
    st.dataframe(metrics.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.subheader("Residual ST-GNN occupancy forecasting")
    st.dataframe(pd.DataFrame({"Metric": ["MAE", "RMSE", "R²"], "Value": [1.130034, 2.404112, 0.986153]}).style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
    if shap_data is not None:
        feature = col(shap_data, ["feature", "features", "feature_name"])
        importance = col(shap_data, ["importance", "mean_abs_shap", "shap_importance", "mean_shap_value"])
        st.subheader("XGBoost feature importance (SHAP)")
        if feature and importance:
            plot_data = shap_data.sort_values(importance, ascending=True)
            st.plotly_chart(px.bar(plot_data, x=importance, y=feature, orientation="h"), use_container_width=True)
        else:
            st.dataframe(shap_data, use_container_width=True, hide_index=True)
    if sensitivity is not None:
        st.subheader("Price-sensitivity scenarios")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        sens = col(sensitivity, ["price_sensitivity", "sensitivity"])
        before = col(sensitivity, ["congestion_before_pct", "congestion_before", "before_congestion"])
        after = col(sensitivity, ["congestion_after_pct", "congestion_after", "after_congestion"])
        if sens and before and after:
            melted = sensitivity.melt(id_vars=[sens], value_vars=[before, after], var_name="Scenario", value_name="Congestion")
            st.plotly_chart(px.line(melted, x=sens, y="Congestion", color="Scenario", markers=True), use_container_width=True)
    st.warning("Pricing and demand redistribution are scenario simulations. Reported predictive metrics should be interpreted with the model-validation limitations discussed in the project.")

st.divider()
st.caption("UrbanEV Smart Charging Network | Academic capstone | UrbanEV Shenzhen dataset")
