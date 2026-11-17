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
    page_title="UrbanEV | Smart Charging Network",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# Supports the uploaded filenames with or without "(1)" suffixes.
FILE_CANDIDATES = {
    "pricing": ["dynamic_pricing_simulation.csv"],
    "decision": ["intelligent_decision_support.csv"],
    "comparison": ["final_pricing_comparison.csv", "final_pricing_comparison (1).csv"],
    "zones": ["zone-information.csv", "zone_information.csv"],
    "predictions": ["spatial_model_test_predictions.csv",
                    "spatial_model_test_predictions (1).csv"],
    "sensitivity": ["sensitivity_analysis.csv", "sensitivity_analysis (1).csv"],
    "alternatives": ["nearest_alternative_zones.csv",
                     "nearest_alternative_zones (1).csv"],
    "neighbours": ["zone_neighbours.csv"],
    "stgnn": ["residual_stgnn_comparison.csv",
              "residual_stgnn_comparison (1).csv"],
    "shap": ["shap_feature_importance.csv"],
}

@st.cache_data(show_spinner=False)
def read_csv(path, modified):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return None

def find_file(key):
    for filename in FILE_CANDIDATES[key]:
        path = BASE_DIR / filename
        if path.exists():
            return path
    # Fallback: match a candidate's base name when the upload tool changes suffixes.
    for candidate in FILE_CANDIDATES[key]:
        stem = Path(candidate).stem.split(" (")[0]
        matches = sorted(BASE_DIR.glob(stem + "*.csv"))
        if matches:
            return matches[0]
    return None

def load_data(key):
    path = find_file(key)
    return read_csv(str(path), path.stat().st_mtime) if path else None

data = {key: load_data(key) for key in FILE_CANDIDATES}
pricing = data["pricing"]
decision = data["decision"]
comparison = data["comparison"]
zones = data["zones"]
predictions = data["predictions"]
sensitivity = data["sensitivity"]
alternatives = data["alternatives"]
neighbours = data["neighbours"]
stgnn = data["stgnn"]
shap_data = data["shap"]

def norm(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def col(df, names):
    if df is None:
        return None
    lookup = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in lookup:
            return lookup[norm(name)]
    return None

def num(series):
    return pd.to_numeric(series, errors="coerce")

def money(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"

def pct(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}%"

def clean_zone(df):
    if df is None:
        return None
    out = df.copy()
    zone_col = col(out, ["zone_id", "tazid", "zone", "taz_id"])
    if zone_col and zone_col != "zone_id":
        out = out.rename(columns={zone_col: "zone_id"})
    if "zone_id" in out.columns:
        out["zone_id"] = (
            out["zone_id"].astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        out = out.dropna(subset=["zone_id"])
        out = out[out["zone_id"] != ""]
    time_col = col(out, ["time", "timestamp", "datetime"])
    if time_col:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out

pricing = clean_zone(pricing)
decision = clean_zone(decision)
predictions = clean_zone(predictions)
zones = clean_zone(zones)
alternatives = clean_zone(alternatives)
neighbours = clean_zone(neighbours)

def safe_mean(df, column):
    return num(df[column]).mean() if df is not None and column else np.nan

def safe_zone_filter(df, zone):
    if df is None or zone is None or "zone_id" not in df.columns:
        return pd.DataFrame()
    return df[df["zone_id"].astype(str) == str(zone)].copy()

def sorted_zone_ids():
    ids = set()
    for frame in [predictions, pricing, decision, zones, alternatives, neighbours]:
        if frame is not None and "zone_id" in frame.columns:
            ids.update(frame["zone_id"].dropna().astype(str).tolist())
    return sorted(ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

zone_options = sorted_zone_ids()

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e6eaf0;
    padding: 12px 14px; border-radius: 12px;
}
.hero {
    padding: 1.25rem 1.5rem; border-radius: 14px;
    background: linear-gradient(115deg, #102d46, #176b72);
    color: white; margin-bottom: 1rem;
}
.hero h1 {color: white; margin: 0 0 .3rem 0;}
.hero p {color: #e5f2f4; margin: 0;}
</style>
""", unsafe_allow_html=True)

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
    if st.session_state.get("selected_zone") not in zone_options:
        st.session_state.selected_zone = zone_options[0]
    selected_zone = st.sidebar.selectbox(
        "Select charging zone", zone_options,
        index=zone_options.index(st.session_state.selected_zone)
    )
    st.session_state.selected_zone = selected_zone
else:
    selected_zone = None
    st.sidebar.error("No zone IDs found. Check the uploaded prediction or pricing CSV files.")

st.sidebar.caption("UrbanEV Shenzhen | Jan–Feb 2023")
st.markdown("""
<div class="hero">
<h1>UrbanEV Smart Charging Network</h1>
<p>Zone-level congestion prediction, occupancy analysis and dynamic pricing decision support</p>
</div>
""", unsafe_allow_html=True)

def show_zone_metrics():
    zp = safe_zone_filter(predictions, selected_zone)
    zr = safe_zone_filter(pricing, selected_zone)
    occ = col(zp, ["occupancy"])
    probability = col(zp, ["predicted_congestion_probability"])
    base = col(zr, ["service_price", "base_price"])
    dynamic = col(zr, ["dynamic_price", "optimized_service_price"])
    a, b, c, d = st.columns(4)
    a.metric("Selected zone", selected_zone or "N/A")
    average_occupancy = safe_mean(zp, occ) if occ else np.nan
    b.metric("Average occupancy", f"{average_occupancy:.2f}" if pd.notna(average_occupancy) else "N/A")
    mean_prob = safe_mean(zp, probability) * 100 if probability else np.nan
    c.metric("Mean congestion probability", pct(mean_prob))
    if base and dynamic:
        change = safe_mean(zr, dynamic) - safe_mean(zr, base)
        d.metric("Mean service-price change", money(change))
    else:
        d.metric("Pricing records", f"{len(zr):,}" if not zr.empty else "N/A")

def draw_zone_map():
    if zones is None or zones.empty:
        st.info("Map unavailable: zone-information.csv with zone_id, latitude and longitude was not included. The neighbour files contain distances, not coordinates.")
        return
    lat = col(zones, ["latitude", "lat"])
    lon = col(zones, ["longitude", "lon", "long"])
    if not lat or not lon:
        st.info("Map unavailable: the zone-information file needs latitude and longitude columns.")
        return
    md = zones.copy()
    md[lat] = num(md[lat])
    md[lon] = num(md[lon])
    md = md.dropna(subset=[lat, lon])
    md = md[md[lat].between(-90, 90) & md[lon].between(-180, 180)]
    if md.empty:
        st.warning("No valid zone coordinates were found.")
        return
    if not FOLIUM_AVAILABLE:
        st.map(md[[lat, lon]].rename(columns={lat: "latitude", lon: "longitude"}))
        st.caption("Install folium and streamlit-folium to enable clickable zone markers.")
        return
    center = [md[lat].mean(), md[lon].mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    prob_col = col(predictions, ["predicted_congestion_probability"])
    occ_col = col(predictions, ["occupancy"])
    summary = None
    if predictions is not None and "zone_id" in predictions.columns:
        aggregations = {}
        if prob_col: aggregations[prob_col] = "mean"
        if occ_col: aggregations[occ_col] = "mean"
        if aggregations:
            summary = predictions.groupby("zone_id").agg(aggregations).reset_index()
            summary = summary.rename(columns={
                prob_col: "mean_probability" if prob_col else prob_col,
                occ_col: "mean_occupancy" if occ_col else occ_col
            })
            md = md.merge(summary, on="zone_id", how="left")
    for _, row in md.iterrows():
        zid = str(row["zone_id"])
        probability = row.get("mean_probability", np.nan)
        colour = "purple" if zid == str(selected_zone) else (
            "red" if pd.notna(probability) and probability >= .5 else
            "green" if pd.notna(probability) else "blue"
        )
        popup = f"Zone: {zid}"
        if pd.notna(row.get("mean_occupancy", np.nan)):
            popup += f"<br>Mean occupancy: {row['mean_occupancy']:.2f}"
        if pd.notna(probability):
            popup += f"<br>Mean congestion probability: {probability:.3f}"
        folium.CircleMarker(
            [row[lat], row[lon]], radius=8 if zid == str(selected_zone) else 5,
            color=colour, fill=True, fill_opacity=.8,
            tooltip=f"Zone {zid}", popup=folium.Popup(popup, max_width=260)
        ).add_to(fmap)
    result = st_folium(fmap, use_container_width=True, height=540,
                       key="urbanev_map", returned_objects=["last_object_clicked"])
    clicked = result.get("last_object_clicked") if isinstance(result, dict) else None
    if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
        distances = ((md[lat] - float(clicked["lat"])) ** 2 +
                     (md[lon] - float(clicked["lng"])) ** 2)
        clicked_zone = str(md.loc[distances.idxmin(), "zone_id"])
        if clicked_zone in zone_options and clicked_zone != selected_zone:
            st.session_state.selected_zone = clicked_zone
            st.rerun()
    st.caption("Purple = selected; red = mean predicted congestion probability ≥ 0.50; green = below 0.50; blue = no prediction.")

if page == "Interactive Zone Explorer":
    st.title("Interactive Zone Explorer")
    st.write("Select a zone from the sidebar. Zone-level prediction, pricing and decision-support information will update.")
    if selected_zone is None:
        st.error("No charging zones are available.")
    else:
        show_zone_metrics()
        st.subheader("Charging-zone map")
        draw_zone_map()
        st.divider()
        zp = safe_zone_filter(predictions, selected_zone)
        zd = safe_zone_filter(decision, selected_zone)
        zr = safe_zone_filter(pricing, selected_zone)
        if not zp.empty:
            t = col(zp, ["time"])
            o = col(zp, ["occupancy"])
            pr = col(zp, ["predicted_congestion_probability"])
            if t:
                zp = zp.sort_values(t)
            if t and o:
                st.subheader("Observed occupancy over time")
                st.plotly_chart(px.line(zp, x=t, y=o, title=f"Zone {selected_zone}: observed occupancy"), use_container_width=True)
            if t and pr:
                chart = zp.copy()
                chart[pr] = num(chart[pr]) * 100
                st.subheader("Predicted congestion probability")
                st.plotly_chart(px.line(chart, x=t, y=pr, labels={pr: "Probability (%)"}), use_container_width=True)
            st.subheader("Prediction records")
            st.dataframe(zp.head(500), use_container_width=True, hide_index=True)
        else:
            st.info("No prediction records are available for this zone.")
        if not zd.empty:
            st.subheader("Intelligent decision-support recommendations")
            st.dataframe(zd.head(200), use_container_width=True, hide_index=True)
        if not zr.empty:
            base = col(zr, ["service_price", "base_price"])
            dyn = col(zr, ["dynamic_price", "optimized_service_price"])
            t = col(zr, ["time"])
            if base and dyn:
                st.subheader("Fixed and dynamic service-price comparison")
                if t:
                    long = zr.sort_values(t).melt(
                        id_vars=[t], value_vars=[base, dyn],
                        var_name="Pricing strategy", value_name="Service price"
                    )
                    st.plotly_chart(px.line(long, x=t, y="Service price",
                                            color="Pricing strategy"), use_container_width=True)
                st.dataframe(zr[[c for c in [t, base, dyn, col(zr, ["price_change", "price_increase_pct"])] if c]], use_container_width=True, hide_index=True)

elif page == "Network Overview":
    st.title("Network Overview")
    st.write("Network-level summary of the uploaded test-period prediction and pricing records.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Charging zones", f"{len(zone_options):,}")
    c2.metric("Prediction records", f"{len(predictions):,}" if predictions is not None else "N/A")
    c3.metric("Pricing records", f"{len(pricing):,}" if pricing is not None else "N/A")
    if predictions is not None:
        occ = col(predictions, ["occupancy"])
        t = col(predictions, ["time"])
        z = col(predictions, ["zone_id"])
        if occ:
            st.subheader("Occupancy distribution")
            st.plotly_chart(px.histogram(predictions, x=occ, nbins=45), use_container_width=True)
        if occ and t:
            hourly = predictions.assign(hour=predictions[t].dt.hour).groupby("hour", as_index=False)[occ].mean()
            st.subheader("Average occupancy by hour")
            st.plotly_chart(px.line(hourly, x="hour", y=occ, markers=True), use_container_width=True)
        if occ and z:
            top = predictions.groupby(z)[occ].mean().nlargest(15).reset_index()
            st.subheader("15 zones with highest average occupancy")
            st.plotly_chart(px.bar(top, x=z, y=occ), use_container_width=True)
    st.subheader("Zone map")
    draw_zone_map()

elif page == "Congestion Prediction":
    st.title("Zone-level Congestion Prediction")
    st.write("The uploaded XGBoost prediction file contains predicted congestion probabilities and high-occupancy labels.")
    view = safe_zone_filter(predictions, selected_zone)
    if view.empty:
        st.warning("No prediction records found for the selected zone.")
    else:
        probability = col(view, ["predicted_congestion_probability"])
        actual = col(view, ["actual_high_occupancy"])
        predicted = col(view, ["predicted_high_occupancy"])
        t = col(view, ["time"])
        occ = col(view, ["occupancy"])
        a, b, c = st.columns(3)
        a.metric("Mean congestion probability", pct(safe_mean(view, probability) * 100 if probability else np.nan))
        b.metric("Actual high-occupancy share", pct(safe_mean(view, actual) * 100 if actual else np.nan))
        c.metric("Predicted high-occupancy share", pct(safe_mean(view, predicted) * 100 if predicted else np.nan))
        if t and probability:
            chart = view.sort_values(t).copy()
            chart[probability] = num(chart[probability]) * 100
            st.plotly_chart(px.line(chart, x=t, y=probability,
                                    labels={probability: "Probability (%)"},
                                    title=f"Zone {selected_zone}: predicted congestion probability"),
                            use_container_width=True)
        if t and occ:
            st.plotly_chart(px.line(view.sort_values(t), x=t, y=occ,
                                    title="Observed occupancy"), use_container_width=True)
        st.subheader("Prediction records")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)

elif page == "Occupancy Forecasting":
    st.title("Occupancy Forecasting")
    st.write("Residual ST-GNN model performance based on the supplied model-comparison results.")
    if stgnn is None:
        st.warning("residual_stgnn_comparison.csv was not found.")
    else:
        st.dataframe(stgnn, use_container_width=True, hide_index=True)
        model_col = col(stgnn, ["model", "unnamed:_0", "unnamed_0"])
        mae_col = col(stgnn, ["mae"])
        rmse_col = col(stgnn, ["rmse"])
        r2_col = col(stgnn, ["r2", "r_2"])
        if model_col:
            residual = stgnn[stgnn[model_col].astype(str).str.contains("Residual ST-GNN", case=False)]
            if not residual.empty:
                row = residual.iloc[0]
                a, b, c = st.columns(3)
                a.metric("Residual ST-GNN MAE", f"{row[mae_col]:.4f}" if mae_col else "N/A")
                b.metric("Residual ST-GNN RMSE", f"{row[rmse_col]:.4f}" if rmse_col else "N/A")
                c.metric("Residual ST-GNN R²", f"{row[r2_col]:.4f}" if r2_col else "N/A")
    st.info("The supplied files contain model-comparison metrics, but no zone-by-zone actual-versus-forecast time series. A zone-specific forecast chart is therefore not displayed.")

elif page == "Dynamic Pricing":
    st.title("Interactive Dynamic Pricing")
    st.write("Compare fixed service prices with dynamic prices for the selected charging zone.")
    view = safe_zone_filter(pricing, selected_zone)
    base = col(view, ["service_price", "base_price"])
    dynamic = col(view, ["dynamic_price", "optimized_service_price"])
    t = col(view, ["time"])
    adjustment = col(view, ["price_increase_pct", "price_change"])
    if view.empty or not base or not dynamic:
        st.warning("Pricing data or required price columns are unavailable for this zone.")
    else:
        old_mean = num(view[base]).mean()
        new_mean = num(view[dynamic]).mean()
        change = ((new_mean - old_mean) / old_mean * 100
                  if pd.notna(old_mean) and old_mean != 0 and pd.notna(new_mean)
                  else np.nan)
        a, b, c = st.columns(3)
        a.metric("Mean fixed service price", money(old_mean))
        b.metric("Mean dynamic service price", money(new_mean))
        c.metric("Mean price change", pct(change))
        if t:
            long = view.sort_values(t).melt(
                id_vars=[t], value_vars=[base, dynamic],
                var_name="Pricing strategy", value_name="Service price"
            )
            st.plotly_chart(px.line(long, x=t, y="Service price", color="Pricing strategy",
                                    title=f"Zone {selected_zone}: price comparison"),
                            use_container_width=True)
        if adjustment:
            st.plotly_chart(px.histogram(view, x=adjustment, nbins=30,
                                         title="Price adjustment distribution"),
                            use_container_width=True)
        st.subheader("Pricing records")
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    if comparison is not None:
        st.subheader("Network-wide pricing strategy comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        strategy = col(comparison, ["pricing_strategy"])
        mean_price = col(comparison, ["mean_price"])
        if strategy and mean_price:
            st.plotly_chart(px.bar(comparison, x=strategy, y=mean_price,
                                   title="Mean service price across pricing strategies"),
                            use_container_width=True)

elif page == "Demand Redistribution":
    st.title("Nearby Alternative Zones & Demand Redistribution")
    st.write("Review nearby alternative charging zones and the supplied price-sensitivity scenarios.")
    if alternatives is not None:
        st.subheader("Nearest alternative zones")
        alt = alternatives.copy()
        source_zone = col(alt, ["zone_id"])
        alt_zone = col(alt, ["alternative_zone", "neighbour_zone_id", "neighbor_zone_id"])
        distance = col(alt, ["distance_km"])
        if source_zone and alt_zone:
            alt[source_zone] = alt[source_zone].astype(str).str.replace(r"\.0$", "", regex=True)
            selected_alternatives = alt[alt[source_zone] == str(selected_zone)] if selected_zone else alt
            if selected_alternatives.empty:
                st.info("No alternative-zone records were found for the selected zone.")
            else:
                st.dataframe(selected_alternatives.sort_values(distance) if distance else selected_alternatives,
                             use_container_width=True, hide_index=True)
        else:
            st.dataframe(alt.head(300), use_container_width=True, hide_index=True)
    elif neighbours is not None:
        st.subheader("Neighbouring zones")
        st.dataframe(neighbours.head(300), use_container_width=True, hide_index=True)
    else:
        st.info("No nearest-alternative or neighbour-zone CSV was found.")
    if sensitivity is not None:
        st.subheader("Price-sensitivity scenarios")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        sens = col(sensitivity, ["price_sensitivity"])
        before = col(sensitivity, ["congestion_before_pct"])
        after = col(sensitivity, ["congestion_after_pct"])
        if sens and before and after:
            long = sensitivity.melt(id_vars=[sens], value_vars=[before, after],
                                    var_name="Scenario", value_name="Congestion (%)")
            st.plotly_chart(px.line(long, x=sens, y="Congestion (%)", color="Scenario",
                                    markers=True, title="Congestion before and after by price sensitivity"),
                            use_container_width=True)
    st.warning("The supplied scenario file reports simulated demand transfer and congestion changes. These are scenario results, not measured customer behaviour. The pricing and prediction files do not contain zone-level incoming/outgoing volume records.")

elif page == "Model & Scenario Analysis":
    st.title("Model & Scenario Analysis")
    st.write("Model metrics are calculated from the supplied prediction records or read from the supplied comparison files.")
    st.subheader("XGBoost congestion classification")
    if predictions is not None:
        actual_col = col(predictions, ["actual_high_occupancy"])
        pred_col = col(predictions, ["predicted_high_occupancy"])
        prob_col = col(predictions, ["predicted_congestion_probability"])
        if actual_col and pred_col:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            valid = predictions[[actual_col, pred_col]].dropna()
            y_true = num(valid[actual_col]).astype(int)
            y_pred = num(valid[pred_col]).astype(int)
            rows = [
                ("Accuracy", accuracy_score(y_true, y_pred)),
                ("Precision", precision_score(y_true, y_pred, zero_division=0)),
                ("Recall", recall_score(y_true, y_pred, zero_division=0)),
                ("F1-score", f1_score(y_true, y_pred, zero_division=0)),
            ]
            if prob_col:
                auc_data = predictions[[actual_col, prob_col]].dropna()
                try:
                    auc = roc_auc_score(auc_data[actual_col].astype(int), num(auc_data[prob_col]))
                    rows.append(("ROC-AUC", auc))
                except ValueError:
                    pass
            metrics = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(metrics.style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
        else:
            st.warning("Actual and predicted high-occupancy columns are required to calculate classification metrics.")
    else:
        st.warning("Prediction CSV not found; XGBoost metrics cannot be calculated.")

    st.subheader("Residual ST-GNN occupancy forecasting")
    if stgnn is not None:
        st.dataframe(stgnn, use_container_width=True, hide_index=True)
    else:
        st.info("ST-GNN comparison file is unavailable.")

    if shap_data is not None:
        feature = col(shap_data, ["feature", "features", "feature_name"])
        importance = col(shap_data, ["importance", "mean_abs_shap", "shap_importance", "mean_shap_value"])
        st.subheader("XGBoost feature importance (SHAP)")
        if feature and importance:
            plot = shap_data.copy()
            plot[importance] = num(plot[importance])
            plot = plot.dropna(subset=[importance]).sort_values(importance, ascending=True)
            st.plotly_chart(px.bar(plot, x=importance, y=feature, orientation="h",
                                   title="Mean absolute SHAP feature importance"),
                            use_container_width=True)
        else:
            st.dataframe(shap_data, use_container_width=True, hide_index=True)

    if sensitivity is not None:
        st.subheader("Price-sensitivity scenarios")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        sens = col(sensitivity, ["price_sensitivity"])
        before = col(sensitivity, ["congestion_before_pct"])
        after = col(sensitivity, ["congestion_after_pct"])
        if sens and before and after:
            long = sensitivity.melt(id_vars=[sens], value_vars=[before, after],
                                    var_name="Scenario", value_name="Congestion (%)")
            st.plotly_chart(px.line(long, x=sens, y="Congestion (%)",
                                    color="Scenario", markers=True),
                            use_container_width=True)

st.divider()
st.caption("UrbanEV Smart Charging Network | Academic capstone | UrbanEV Shenzhen dataset")
