from pathlib import Path

code = r'''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

try:
    import folium
    from streamlit_folium import st_folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False

# ---------------------------------------------------------
# PAGE + VISUAL STYLE
# ---------------------------------------------------------
st.set_page_config(
    page_title="UrbanEV | Smart Charging",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e8eef7; }
[data-testid="stHeader"] { background: rgba(11,18,32,0.92); }
[data-testid="stSidebar"] { background: #111c2d; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1500px; }
h1,h2,h3 { color: #f4f8fc; }
p, label, .stMarkdown { color: #d5deea; }
div[data-testid="stMetric"] {
    background: #142238; border: 1px solid #263b54;
    padding: 15px 16px; border-radius: 14px;
}
div[data-testid="stMetricLabel"] { color: #aebfd1; }
div[data-testid="stMetricValue"] { color: #f4f8fc; }
div[data-testid="stMetricDelta"] { color: #35d0a0; }
.stSelectbox div[data-baseweb="select"] > div,
.stDateInput div[data-baseweb="input"] {
    background: #142238; border-color: #31465f; border-radius: 10px;
}
.hero {
    background: linear-gradient(115deg,#10283e 0%,#124b50 58%,#12645c 100%);
    padding: 1.35rem 1.6rem; border: 1px solid #28666b;
    border-radius: 18px; margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; color: #ffffff; font-size: 2rem; }
.hero p { margin: .35rem 0 0 0; color: #d5f2ec; }
.panel {
    background: #142238; border: 1px solid #263b54;
    border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
}
.pill {
    display: inline-block; padding: 5px 10px; border-radius: 20px;
    background: #173e3b; color: #75e3c4; font-size: .82rem;
    border: 1px solid #27665b;
}
div.stButton > button {
    background: #13a88a; color: white; border: 0;
    border-radius: 10px; font-weight: 650; min-height: 2.7rem;
}
div.stButton > button:hover { background: #0e8c74; color: white; border: 0; }
[data-testid="stDataFrame"] { border: 1px solid #263b54; border-radius: 10px; }
hr { border-color: #263b54; }
.small-muted { color: #9aadc2; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
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
def read_file(path, modified):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load(key):
    path = ROOT / FILES[key]
    return read_file(str(path), path.stat().st_mtime) if path.exists() else None

data = {k: load(k) for k in FILES}
pricing = data["pricing"]
xgb = data["xgb"]
zones = data["zones"]
neighbours = data["neighbours"]
comparison = data["comparison"]
sensitivity = data["sensitivity"]
stgnn = data["stgnn"]

def find_col(df, choices):
    if df is None:
        return None
    normal = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for item in choices:
        key = item.lower().replace(" ", "_")
        if key in normal:
            return normal[key]
    return None

def prep(df):
    if df is None:
        return None
    out = df.copy()
    z = find_col(out, ["zone_id", "TAZID", "taz_id", "zone"])
    if z and z != "zone_id":
        out = out.rename(columns={z: "zone_id"})
    if "zone_id" in out:
        out["zone_id"] = out["zone_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    t = find_col(out, ["time", "timestamp", "datetime"])
    if t:
        out[t] = pd.to_datetime(out[t], errors="coerce")
    return out

pricing, xgb, zones, neighbours = map(prep, [pricing, xgb, zones, neighbours])

zone_ids = set()
for frame in [zones, xgb, pricing]:
    if frame is not None and "zone_id" in frame:
        zone_ids.update(frame["zone_id"].dropna().astype(str).tolist())
zone_ids = sorted(zone_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

if "active_zone" not in st.session_state and zone_ids:
    st.session_state.active_zone = zone_ids[0]
if st.session_state.get("active_zone") not in zone_ids and zone_ids:
    st.session_state.active_zone = zone_ids[0]

def zone_rows(df, zone=None):
    zone = str(zone or st.session_state.get("active_zone", ""))
    if df is None or "zone_id" not in df:
        return pd.DataFrame()
    return df[df["zone_id"].astype(str) == zone].copy()

def mean_num(df, column):
    if df is None or column is None or column not in df:
        return np.nan
    return pd.to_numeric(df[column], errors="coerce").mean()

def money(x):
    return "Not available" if pd.isna(x) else f"¥{x:.3f}"

def pct(x):
    return "Not available" if pd.isna(x) else f"{x:.1f}%"

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value, delta) in zip(cols, items):
        c.metric(label, value, delta)

def get_map_frame():
    if zones is None or zones.empty or "zone_id" not in zones:
        return pd.DataFrame()
    m = zones.copy()
    lat = find_col(m, ["latitude", "lat"])
    lon = find_col(m, ["longitude", "lon", "long"])
    if not lat or not lon:
        return pd.DataFrame()
    if xgb is not None and "zone_id" in xgb:
        occ = find_col(xgb, ["occupancy"])
        prob = find_col(xgb, ["predicted_congestion_probability"])
        aggregations = {}
        if occ: aggregations[occ] = "mean"
        if prob: aggregations[prob] = "mean"
        if aggregations:
            summary = xgb.groupby("zone_id").agg(aggregations).reset_index()
            rename = {}
            if occ: rename[occ] = "mean_occupancy"
            if prob: rename[prob] = "mean_risk"
            m = m.merge(summary.rename(columns=rename), on="zone_id", how="left")
    return m.dropna(subset=[lat, lon]), lat, lon

def render_map(height=520):
    result = get_map_frame()
    if isinstance(result, tuple):
        m, lat, lon = result
    else:
        m = result
        lat = lon = None
    if m.empty:
        st.info("Map coordinates are unavailable. Check zone-information.csv contains latitude and longitude.")
        return
    if not HAS_MAP:
        st.warning("For clickable map selection, add folium and streamlit-folium to requirements.txt.")
        st.map(m[[lat, lon]].rename(columns={lat: "latitude", lon: "longitude"}))
        return
    center = [m[lat].astype(float).mean(), m[lon].astype(float).mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")
    for _, row in m.iterrows():
        zid = str(row["zone_id"])
        risk = row.get("mean_risk", np.nan)
        occupancy = row.get("mean_occupancy", np.nan)
        colour = "purple" if zid == str(st.session_state.active_zone) else (
            "red" if pd.notna(risk) and risk >= .5 else ("green" if pd.notna(risk) else "blue")
        )
        popup = f"<b>Zone {zid}</b>"
        if pd.notna(occupancy): popup += f"<br>Mean occupancy: {occupancy:.2f}"
        if pd.notna(risk): popup += f"<br>Mean congestion probability: {risk:.3f}"
        folium.CircleMarker(
            [float(row[lat]), float(row[lon])],
            radius=8 if zid == str(st.session_state.active_zone) else 5,
            color=colour, fill=True, fill_opacity=.85,
            tooltip=f"Select Zone {zid}",
            popup=folium.Popup(popup, max_width=250)
        ).add_to(fmap)
    result = st_folium(fmap, height=height, use_container_width=True,
                       key="urban_ev_click_map", returned_objects=["last_object_clicked"])
    click = result.get("last_object_clicked") if isinstance(result, dict) else None
    if click and click.get("lat") is not None and click.get("lng") is not None:
        distances = ((m[lat].astype(float) - float(click["lat"])) ** 2
                     + (m[lon].astype(float) - float(click["lng"])) ** 2)
        nearest = str(m.loc[distances.idxmin(), "zone_id"])
        if nearest in zone_ids and nearest != st.session_state.active_zone:
            st.session_state.active_zone = nearest
            st.rerun()
    st.caption("Purple: selected zone · Red: higher predicted congestion risk · Green: lower risk · Blue: prediction unavailable. Click a marker to select it.")

def zone_selector():
    if not zone_ids:
        st.error("Zone IDs were not found. Upload zone-information.csv and the prediction/pricing files.")
        return None
    current = st.session_state.active_zone
    choice = st.selectbox("Choose a charging zone", zone_ids,
                          index=zone_ids.index(current), key="zone_picker")
    st.session_state.active_zone = choice
    return choice

def zone_snapshot(zone):
    pred = zone_rows(xgb, zone)
    price = zone_rows(pricing, zone)
    occ_c = find_col(pred, ["occupancy"])
    risk_c = find_col(pred, ["predicted_congestion_probability"])
    base_c = find_col(price, ["service_price"])
    opt_c = find_col(price, ["optimized_service_price", "dynamic_price"])
    occ = mean_num(pred, occ_c)
    risk = mean_num(pred, risk_c)
    base = mean_num(price, base_c)
    opt = mean_num(price, opt_c)
    return pred, price, occ, risk, base, opt

# ---------------------------------------------------------
# APP NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("## ⚡ UrbanEV")
st.sidebar.caption("Smart charging network")
experience = st.sidebar.radio("Choose experience", ["🚗 EV Driver", "🏙️ Network Operator"])
st.sidebar.divider()
st.sidebar.markdown("**Charging zone**")
selected = zone_selector()
st.sidebar.caption("Shenzhen · UrbanEV research dataset")
st.sidebar.caption("Historical study period: Sep 2022 – Feb 2023")

st.markdown("""
<div class="hero">
  <h1>UrbanEV Smart Charging</h1>
  <p>Find suitable charging zones. Understand congestion. Explore smarter pricing.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# EV DRIVER EXPERIENCE
# ---------------------------------------------------------
if experience == "🚗 EV Driver":
    st.markdown("### Find a charging zone")
    st.write("Choose a zone to explore its charging activity, congestion outlook, prices and nearby alternatives.")
    selected = zone_selector() if selected is None else selected

    if selected is not None:
        pred, price, occ, risk, base, opt = zone_snapshot(selected)
        risk_pct = risk * 100 if pd.notna(risk) else np.nan
        status = ("High expected congestion" if pd.notna(risk) and risk >= .5
                  else "Lower expected congestion" if pd.notna(risk)
                  else "Prediction unavailable")
        st.markdown(f'<span class="pill">Selected charging zone: {selected}</span> &nbsp; <span class="pill">{status}</span>', unsafe_allow_html=True)
        st.write("")
        metric_row([
            ("Observed average occupancy", f"{occ:.1f}" if pd.notna(occ) else "N/A", None),
            ("Next-hour congestion likelihood", pct(risk_pct), None),
            ("Current service price", money(base), None),
            ("Simulated optimised price", money(opt), None),
        ])

        left, right = st.columns([1.15, .85], gap="large")
        with left:
            st.markdown("#### Charging zone map")
            render_map(510)
        with right:
            st.markdown("#### Is this zone a suitable choice?")
            if pd.isna(risk):
                st.info("There is no linked congestion prediction for this zone in the current prediction file.")
            elif risk >= .5:
                st.warning("This zone has an elevated predicted chance of high occupancy. Consider checking nearby alternatives.")
            else:
                st.success("The model indicates a lower predicted chance of high occupancy for this zone.")
            if pd.notna(base) and pd.notna(opt):
                change = ((opt / base) - 1) * 100 if base else np.nan
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Service price comparison**")
                st.write(f"Existing service price: **{money(base)}**")
                st.write(f"Simulated optimised service price: **{money(opt)}**")
                st.write(f"Price change in this simulation: **{pct(change)}**")
                st.markdown('<span class="small-muted">These are simulated service prices, not live charging tariffs.</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Pricing information is not available for this zone.")

        st.divider()
        st.markdown("#### Zone activity")
        if not pred.empty:
            t = find_col(pred, ["time"])
            occ_c = find_col(pred, ["occupancy"])
            prob_c = find_col(pred, ["predicted_congestion_probability"])
            if t:
                pred = pred.sort_values(t)
            if t and occ_c:
                st.plotly_chart(px.line(pred, x=t, y=occ_c,
                    title=f"Observed occupancy · Zone {selected}",
                    labels={t: "Time", occ_c: "Occupancy"},
                    template="plotly_dark"), use_container_width=True)
            if t and prob_c:
                chart = pred.copy()
                chart["Congestion likelihood (%)"] = pd.to_numeric(chart[prob_c], errors="coerce") * 100
                st.plotly_chart(px.line(chart, x=t, y="Congestion likelihood (%)",
                    title="Predicted chance of high occupancy in the next hour",
                    template="plotly_dark"), use_container_width=True)
        else:
            st.info("No zone-level prediction records are available for the selected zone.")

        st.divider()
        st.markdown("#### Nearby alternatives")
        if neighbours is not None and not neighbours.empty:
            n = neighbours.copy()
            cols_lower = {str(c).lower().replace(" ", "_"): c for c in n.columns}
            origin_c = next((cols_lower[k] for k in ["zone_id", "origin_zone", "source_zone", "tazid"] if k in cols_lower), None)
            if origin_c:
                n[origin_c] = n[origin_c].astype(str).str.replace(r"\.0$", "", regex=True)
                alternatives = n[n[origin_c] == str(selected)]
            else:
                alternatives = pd.DataFrame()
            if not alternatives.empty:
                st.write("Nearby zones listed for this selected zone:")
                st.dataframe(alternatives, use_container_width=True, hide_index=True)
            else:
                st.info("No matching nearby-zone records were found for this selected zone in nearest_alternative_zones.csv.")
        else:
            st.info("The nearest-alternative-zone file is not available yet.")
        st.caption("Congestion estimates and price comparisons are based on historical model outputs and simulations. They are not live charger availability or live tariffs.")

# ---------------------------------------------------------
# NETWORK OPERATOR EXPERIENCE
# ---------------------------------------------------------
else:
    st.markdown("### Charging network control centre")
    st.write("Monitor network conditions, inspect individual zones and explore pricing and demand-management scenarios.")
    if selected is not None:
        pred, price, occ, risk, base, opt = zone_snapshot(selected)

    all_occ = find_col(xgb, ["occupancy"])
    all_prob = find_col(xgb, ["predicted_congestion_probability"])
    all_zone = "zone_id" if xgb is not None and "zone_id" in xgb else None
    total_zones = zones["zone_id"].nunique() if zones is not None and "zone_id" in zones else len(zone_ids)
    mean_occ = mean_num(xgb, all_occ)
    mean_risk = mean_num(xgb, all_prob)
    high_risk_zones = "N/A"
    if xgb is not None and all_zone and all_prob:
        risk_by_zone = xgb.groupby(all_zone)[all_prob].mean()
        high_risk_zones = int((risk_by_zone >= .5).sum())
    metric_row([
        ("Charging zones", f"{total_zones:,}", None),
        ("Prediction records", f"{len(xgb):,}" if xgb is not None else "N/A", None),
        ("Network average occupancy", f"{mean_occ:.1f}" if pd.notna(mean_occ) else "N/A", None),
        ("Zones with mean risk ≥ 50%", f"{high_risk_zones:,}" if isinstance(high_risk_zones, int) else high_risk_zones, None),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(["Live-style network view", "Zone operations", "Pricing controls", "Scenario analysis"])

    with tab1:
        st.markdown("#### Network congestion map")
        render_map(550)
        if xgb is not None and all_zone and all_prob:
            risk_by_zone = xgb.groupby(all_zone)[all_prob].mean().reset_index()
            risk_by_zone["Risk category"] = np.where(risk_by_zone[all_prob] >= .5, "Higher risk", "Lower risk")
            st.markdown("#### Zones to review")
            st.dataframe(risk_by_zone.sort_values(all_prob, ascending=False).head(15),
                         use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f"#### Zone {selected} operations")
        if selected is not None:
            pred, price, occ, risk, base, opt = zone_snapshot(selected)
            metric_row([
                ("Average occupancy", f"{occ:.2f}" if pd.notna(occ) else "N/A", None),
                ("Mean congestion probability", pct(risk * 100 if pd.notna(risk) else np.nan), None),
                ("Existing service price", money(base), None),
                ("Optimised service price", money(opt), None),
            ])
            if not pred.empty:
                time_c = find_col(pred, ["time"])
                occ_c = find_col(pred, ["occupancy"])
                prob_c = find_col(pred, ["predicted_congestion_probability"])
                if time_c and occ_c:
                    pred = pred.sort_values(time_c)
                    st.plotly_chart(px.line(pred, x=time_c, y=occ_c,
                        title="Selected-zone occupancy trend", template="plotly_dark"),
                        use_container_width=True)
                if time_c and prob_c:
                    st.plotly_chart(px.line(pred, x=time_c, y=prob_c,
                        title="Selected-zone predicted congestion probability", template="plotly_dark"),
                        use_container_width=True)
                st.dataframe(pred.head(300), use_container_width=True, hide_index=True)
            else:
                st.info("No prediction records are available for this zone.")

    with tab3:
        st.markdown("#### Pricing strategy comparison")
        if comparison is not None:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            strategy = find_col(comparison, ["Pricing Strategy"])
            avg_price = find_col(comparison, ["Mean Price"])
            if strategy and avg_price:
                st.plotly_chart(px.bar(comparison, x=strategy, y=avg_price,
                    title="Average service price by strategy", template="plotly_dark"),
                    use_container_width=True)
        if pricing is not None and selected is not None:
            p = zone_rows(pricing, selected)
            base_c = find_col(p, ["service_price"])
            opt_c = find_col(p, ["optimized_service_price", "dynamic_price"])
            time_c = find_col(p, ["time"])
            if base_c and opt_c and not p.empty:
                st.markdown(f"#### Zone {selected}: pricing record")
                if time_c:
                    long = p.melt(id_vars=[time_c], value_vars=[base_c, opt_c],
                                  var_name="Strategy", value_name="Service price")
                    st.plotly_chart(px.line(long, x=time_c, y="Service price",
                        color="Strategy", template="plotly_dark"),
                        use_container_width=True)
                st.dataframe(p.head(250), use_container_width=True, hide_index=True)
        st.caption("The pricing logic is a simulated policy based on congestion, occupancy, relative demand and electricity cost. It is not a live tariff controller.")

    with tab4:
        st.markdown("#### Price sensitivity scenario explorer")
        st.write("Adjust the assumed price sensitivity to inspect the precomputed scenarios from the project.")
        if sensitivity is not None:
            sens_c = find_col(sensitivity, ["price_sensitivity", "sensitivity"])
            before_c = find_col(sensitivity, ["congestion_before_pct", "congestion_before", "before_congestion"])
            after_c = find_col(sensitivity, ["congestion_after_pct", "congestion_after", "after_congestion"])
            if sens_c:
                values = sorted(pd.to_numeric(sensitivity[sens_c], errors="coerce").dropna().unique())
                if values:
                    chosen = st.select_slider("Assumed price sensitivity", options=values, value=min(values, key=lambda v: abs(v - .5)))
                    scenario = sensitivity[pd.to_numeric(sensitivity[sens_c], errors="coerce") == chosen]
                    st.dataframe(scenario, use_container_width=True, hide_index=True)
            st.dataframe(sensitivity, use_container_width=True, hide_index=True)
            if sens_c and before_c and after_c:
                melted = sensitivity.melt(id_vars=[sens_c], value_vars=[before_c, after_c],
                                          var_name="Measure", value_name="Congestion (%)")
                st.plotly_chart(px.line(melted, x=sens_c, y="Congestion (%)",
                    color="Measure", markers=True, template="plotly_dark",
                    title="Congestion before and after simulated redistribution"),
                    use_container_width=True)
        else:
            st.info("sensitivity_analysis.csv is not available.")
        st.warning("Scenario outputs are simulations based on assumed customer price sensitivity and zone-capacity rules. They should not be interpreted as observed real-world outcomes.")

st.divider()
st.caption("UrbanEV · Interactive EV charging and network decision-support prototype · Historical UrbanEV Shenzhen data")
'''
st.download_button(
    label="Download Interactive App Code",
    data=code,
    file_name="app_user_friendly.py",
    mime="text/x-python"
)
