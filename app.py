"""
EV Charging Network Intelligence Dashboard
Run: streamlit run app.py

Place the CSV files in the same GitHub repository/folder as this app.py.
The app uses supplied prediction/pricing outputs. It can optionally load model
artifacts later, but does not claim to execute XGBoost/ST-GNN without them.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="VoltIQ | EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#101719; --panel:#182326; --panel2:#1d2b2e; --teal:#22c7b8; --green:#b7f36b; --muted:#9aadae; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#0d1517 0%,#111d20 55%,#102322 100%); color:#eef7f5; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f2fbf8; }
[data-testid="stSidebar"] { background:#111d20; border-right:1px solid #263a3c; }
[data-testid="stMetric"] { background:#192629; border:1px solid #294144; padding:15px 18px; border-radius:15px; }
[data-testid="stMetricLabel"] { color:#9aadae !important; }
[data-testid="stMetricValue"] { color:#f4fffc !important; }
.stButton>button { background:linear-gradient(90deg,#16a99d,#80d957); color:#071615; border:0; border-radius:10px; font-weight:700; }
div[data-testid="stTabs"] button { color:#c9d8d6; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#72e2d4; border-bottom-color:#22c7b8; }
div[data-testid="stAlert"] { border-radius:12px; }
.small-muted { color:#9aadae; font-size:0.88rem; }
.hero { padding:22px 24px; border:1px solid #294144; border-radius:18px; background:linear-gradient(110deg,#1b3032,#172426); margin-bottom:18px; }
.pill { display:inline-block; padding:5px 10px; border-radius:20px; background:#183e3a; color:#80eadc; font-size:12px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).resolve().parent

FILE_CANDIDATES = {
    "dynamic": ["dynamic_pricing_simulation.csv"],
    "pricing": ["final_pricing_comparison.csv", "final_pricing_comparison (1).csv"],
    "decision": ["intelligent_decision_support.csv"],
    "alternatives": ["nearest_alternative_zones.csv", "nearest_alternative_zones (1).csv"],
    "model_comparison": ["residual_stgnn_comparison.csv", "residual_stgnn_comparison (1).csv"],
    "sensitivity": ["sensitivity_analysis.csv", "sensitivity_analysis (1).csv"],
    "shap": ["shap_feature_importance.csv"],
    "predictions": ["spatial_model_test_predictions.csv", "spatial_model_test_predictions (1).csv"],
}

@st.cache_data(show_spinner=False)
def load_csv(candidates):
    for filename in candidates:
        path = DATA_DIR / filename
        if path.exists():
            try:
                return pd.read_csv(path), filename
            except Exception as exc:
                st.warning(f"Could not read {filename}: {exc}")
    return None, None

def load_data():
    loaded = {}
    for key, candidates in FILE_CANDIDATES.items():
        loaded[key] = load_csv(candidates)
    return loaded

DATA = load_data()

def df_for(key):
    return DATA[key][0]

def numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def fmt_price(value):
    return f"{float(value):.3f}" if pd.notna(value) else "—"

def risk_label(prob):
    if prob >= .80: return "Very High"
    if prob >= .60: return "High"
    if prob >= .35: return "Moderate"
    return "Low"

def risk_color(prob):
    if prob >= .80: return "#ff6b6b"
    if prob >= .60: return "#ffad5c"
    if prob >= .35: return "#f4d35e"
    return "#8be28b"

def latest_snapshot(df, timestamp=None):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        times = out["time"].dropna()
        if timestamp is not None:
            out = out[out["time"] == pd.Timestamp(timestamp)]
        elif not times.empty:
            out = out[out["time"] == times.max()]
    return out

def get_zone_snapshot(zone_id, timestamp=None):
    source = df_for("dynamic")
    if source is None:
        source = df_for("predictions")
    snap = latest_snapshot(source, timestamp)
    if snap.empty:
        return pd.DataFrame()
    snap["zone_id"] = numeric(snap["zone_id"]).astype("Int64")
    return snap[snap["zone_id"] == int(zone_id)].copy()

def get_zone_alternatives(zone_id, timestamp=None):
    alt = df_for("alternatives")
    snap = latest_snapshot(df_for("dynamic"), timestamp)
    if alt is None or snap.empty:
        return pd.DataFrame()
    alt = alt.copy()
    alt["zone_id"] = numeric(alt["zone_id"]).astype("Int64")
    alt["alternative_zone"] = numeric(alt["alternative_zone"]).astype("Int64")
    alt = alt[alt["zone_id"] == int(zone_id)].copy()
    snap["zone_id"] = numeric(snap["zone_id"]).astype("Int64")
    info = snap[["zone_id", "occupancy", "predicted_congestion_probability",
                 "service_price", "dynamic_price"]].drop_duplicates("zone_id")
    info = info.rename(columns={
        "zone_id": "alternative_zone",
        "occupancy": "alt_occupancy",
        "predicted_congestion_probability": "alt_congestion",
        "service_price": "alt_fixed_price",
        "dynamic_price": "alt_dynamic_price",
    })
    return alt.merge(info, on="alternative_zone", how="left").sort_values("distance_km")

def build_network_figure(zone_id, alternatives):
    """A schematic network diagram, not a geographic map (source CSVs lack coordinates)."""
    nodes = [int(zone_id)]
    if not alternatives.empty:
        nodes += [int(x) for x in alternatives["alternative_zone"].dropna().head(7).tolist()]
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, max(n, 2), endpoint=False)
    coords = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}
    fig = go.Figure()
    for _, row in alternatives.head(7).iterrows():
        target = int(row["alternative_zone"])
        if target in coords:
            x0, y0 = coords[int(zone_id)]
            x1, y1 = coords[target]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color="#365456", width=1.5), hoverinfo="skip", showlegend=False))
    snap = get_zone_snapshot(zone_id)
    prob = float(snap["predicted_congestion_probability"].iloc[0]) if not snap.empty else 0
    colors = ["#ff6b6b" if node == int(zone_id) and prob >= .6 else
              "#22c7b8" if node == int(zone_id) else "#b7f36b" for node in nodes]
    labels = ["Selected" if node == int(zone_id) else f"Zone {node}" for node in nodes]
    fig.add_trace(go.Scatter(
        x=[coords[node][0] for node in nodes], y=[coords[node][1] for node in nodes],
        mode="markers+text", text=labels, textposition="bottom center",
        marker=dict(size=[28 if node == int(zone_id) else 19 for node in nodes],
                    color=colors, line=dict(color="#eafff8", width=1)),
        customdata=nodes, hovertemplate="Zone %{customdata}<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(height=340, margin=dict(l=5,r=5,t=10,b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        annotations=[dict(text="Network relationship view · not to geographic scale",
                          x=.5,y=1.04,xref="paper",yref="paper",showarrow=False,
                          font=dict(color="#91aaa8",size=11))])
    return fig

# ----------------------------- Header ----------------------------------------
st.markdown("""
<div class="hero">
  <div class="pill">EV CHARGING NETWORK INTELLIGENCE</div>
  <h1 style="margin:10px 0 4px 0;">VoltIQ <span style="color:#22c7b8;">/</span> Charging, made smarter.</h1>
  <div class="small-muted">Find available charging zones, compare prices and understand network demand.</div>
</div>
""", unsafe_allow_html=True)

missing = [key for key, (df, _) in DATA.items() if df is None]
if missing:
    st.error("Some data files are missing: " + ", ".join(missing) +
             ". Add the uploaded CSVs beside app.py in your GitHub repository.")
    st.stop()

dynamic = df_for("dynamic").copy()
dynamic["time"] = pd.to_datetime(dynamic["time"], errors="coerce")
dynamic["zone_id"] = numeric(dynamic["zone_id"]).astype(int)
all_times = sorted(dynamic["time"].dropna().unique())
all_zones = sorted(dynamic["zone_id"].unique())

# Sidebar navigation and global controls
with st.sidebar:
    st.markdown("## ⚡ VoltIQ")
    st.caption("Charging network console")
    experience = st.radio("Choose experience", ["EV Driver", "Network Operator"])
    st.divider()
    st.markdown("**Network snapshot**")
    default_time = all_times[-1] if all_times else None
    selected_time = st.select_slider(
        "Historical time",
        options=all_times,
        value=default_time,
        format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y · %H:%M")
    ) if all_times else None
    st.caption("The dataset covers Jan–Feb 2023. This selector changes the displayed historical snapshot.")
    st.divider()
    st.caption("Data-powered demo · price values shown in the units provided by the source data")

snapshot = latest_snapshot(dynamic, selected_time)
snapshot["zone_id"] = numeric(snapshot["zone_id"]).astype(int)
decision = latest_snapshot(df_for("decision"), selected_time)
if not decision.empty:
    decision["zone_id"] = numeric(decision["zone_id"]).astype(int)

# ----------------------------- DRIVER ----------------------------------------
if experience == "EV Driver":
    st.subheader("EV Driver")
    st.write("Choose a charging zone to check its current recorded occupancy, predicted congestion and price.")
    left, right = st.columns([1, 2])
    with left:
        selected_zone = st.selectbox("Select charging zone", all_zones, index=0)
        zone_rows = snapshot[snapshot["zone_id"] == selected_zone]
        if zone_rows.empty:
            st.warning("No record for this zone at the selected time.")
            st.stop()
        z = zone_rows.iloc[0]
        st.markdown("**Selected zone status**")
        prob = float(z["predicted_congestion_probability"])
        st.markdown(f"<span class='pill' style='background:{risk_color(prob)}22;color:{risk_color(prob)}'>{risk_label(prob)} congestion risk</span>", unsafe_allow_html=True)
        st.metric("Predicted congestion", f"{prob*100:.1f}%")
        st.metric("Recorded occupancy", f"{float(z['occupancy']):,.0f}")
        st.metric("Dynamic service price", fmt_price(z.get("dynamic_price", np.nan)))
        st.metric("Fixed service price", fmt_price(z.get("service_price", np.nan)))
    with right:
        st.markdown("#### Nearby charging network")
        alternatives = get_zone_alternatives(selected_zone, selected_time)
        st.plotly_chart(build_network_figure(selected_zone, alternatives), use_container_width=True, config={"displayModeBar": False})
        st.caption("A schematic of connected alternatives. Exact map pins are not available because latitude/longitude fields are not present in the uploaded files.")

    st.markdown("### Better nearby alternatives")
    alternatives = get_zone_alternatives(selected_zone, selected_time)
    if alternatives.empty:
        st.info("No alternative-zone records are available for this selection.")
    else:
        alternatives = alternatives.dropna(subset=["distance_km"])
        alternatives["alt_congestion"] = numeric(alternatives["alt_congestion"], 1)
        alternatives["alt_dynamic_price"] = numeric(alternatives["alt_dynamic_price"], np.nan)
        alternatives["alt_fixed_price"] = numeric(alternatives["alt_fixed_price"], np.nan)
        # Transparent multi-factor ranking: normalized congestion, distance and dynamic price.
        def minmax(s):
            s = numeric(s, 0)
            span = s.max() - s.min()
            return (s - s.min()) / span if span > 0 else pd.Series(0.0, index=s.index)
        alternatives["recommendation_score"] = (
            0.55 * (1 - alternatives["alt_congestion"].clip(0,1)) +
            0.25 * (1 - minmax(alternatives["distance_km"])) +
            0.20 * (1 - minmax(alternatives["alt_dynamic_price"]))
        )
        alternatives = alternatives.sort_values("recommendation_score", ascending=False)
        best = alternatives.iloc[0]
        st.success(f"Recommended alternative: Zone {int(best['alternative_zone'])} · "
                   f"{best['distance_km']:.2f} km away · "
                   f"{risk_label(float(best['alt_congestion']))} congestion risk.")
        display = alternatives.head(8).copy()
        display["Zone"] = display["alternative_zone"].astype(int).map(lambda x: f"Zone {x}")
        display["Distance"] = display["distance_km"].map(lambda x: f"{x:.2f} km")
        display["Congestion risk"] = display["alt_congestion"].map(lambda x: risk_label(float(x)))
        display["Fixed price"] = display["alt_fixed_price"].map(fmt_price)
        display["Dynamic price"] = display["alt_dynamic_price"].map(fmt_price)
        st.dataframe(display[["Zone","Distance","Congestion risk","Fixed price","Dynamic price"]],
                     use_container_width=True, hide_index=True)
        st.caption("Recommendation ranking weights: congestion 55%, distance 25%, dynamic price 20%. Scores are normalized within the available alternatives for this selected zone.")

    st.markdown("### Price comparison")
    price_cols = st.columns(2)
    with price_cols[0]:
        st.metric("Fixed price", fmt_price(z.get("service_price", np.nan)))
    with price_cols[1]:
        st.metric("Optimised dynamic price", fmt_price(z.get("dynamic_price", np.nan)),
                  delta=f"{(float(z.get('dynamic_price',0))-float(z.get('service_price',0))):+.3f}")
    st.caption("Prices are shown in the same units as the supplied dataset; no currency is specified in the source files.")

# ----------------------------- OPERATOR --------------------------------------
else:
    st.subheader("Network Operator")
    st.write("Monitor demand across all charging zones, identify hotspots and test pricing sensitivity.")
    avg_occ = float(snapshot["occupancy"].mean())
    avg_prob = float(snapshot["predicted_congestion_probability"].mean())
    high_risk = int((snapshot["predicted_congestion_probability"] >= .6).sum())
    avg_price = float(snapshot["dynamic_price"].mean())
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Active zones", f"{snapshot['zone_id'].nunique():,}")
    k2.metric("Average occupancy", f"{avg_occ:,.1f}")
    k3.metric("High-risk zones", f"{high_risk:,}")
    k4.metric("Average dynamic price", fmt_price(avg_price))

    st.markdown("### Network congestion overview")
    c1,c2 = st.columns([1.4,1])
    with c1:
        chart_df = snapshot.sort_values("predicted_congestion_probability", ascending=False).head(20).copy()
        chart_df["Zone"] = chart_df["zone_id"].astype(str)
        fig = px.bar(chart_df, x="Zone", y="predicted_congestion_probability",
                     color="predicted_congestion_probability",
                     color_continuous_scale=["#22c7b8","#f4d35e","#ff6b6b"],
                     labels={"predicted_congestion_probability":"Congestion probability"})
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#dce9e6", coloraxis_showscale=False,
                          margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        risk_counts = pd.Series({
            "Low": int((snapshot["predicted_congestion_probability"] < .35).sum()),
            "Moderate": int(((snapshot["predicted_congestion_probability"] >= .35) & (snapshot["predicted_congestion_probability"] < .6)).sum()),
            "High": int(((snapshot["predicted_congestion_probability"] >= .6) & (snapshot["predicted_congestion_probability"] < .8)).sum()),
            "Very High": int((snapshot["predicted_congestion_probability"] >= .8).sum()),
        }).reset_index()
        risk_counts.columns = ["Risk level","Zones"]
        fig2 = px.pie(risk_counts, names="Risk level", values="Zones", hole=.62,
                      color="Risk level",
                      color_discrete_map={"Low":"#8be28b","Moderate":"#f4d35e","High":"#ffad5c","Very High":"#ff6b6b"})
        fig2.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", font_color="#dce9e6",
                           margin=dict(l=5,r=5,t=15,b=5), legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Zone operations")
    operator_zone = st.selectbox("Inspect a zone", all_zones, key="operator_zone")
    oz = snapshot[snapshot["zone_id"] == operator_zone].iloc[0]
    op1,op2,op3 = st.columns(3)
    op1.metric("Congestion probability", f"{float(oz['predicted_congestion_probability'])*100:.1f}%")
    op2.metric("Occupancy", f"{float(oz['occupancy']):,.0f}")
    op3.metric("Dynamic price", fmt_price(oz["dynamic_price"]))
    if not decision.empty:
        drow = decision[decision["zone_id"] == operator_zone]
        if not drow.empty:
            st.info(f"Suggested operational response: **{drow.iloc[0].get('operational_response','Review zone demand')}**  \n"
                    f"Recommended action: {drow.iloc[0].get('recommended_action','Review pricing')}")
    st.markdown("### Pricing sensitivity simulator")
    sens = df_for("sensitivity").copy()
    sens["Price Sensitivity"] = numeric(sens["Price Sensitivity"])
    available_sens = sorted(sens["Price Sensitivity"].dropna().unique())
    selected_sensitivity = st.slider("Pricing sensitivity", min_value=0.0, max_value=1.0,
                                     value=0.5, step=0.05,
                                     help="Scenario control interpolates the uploaded sensitivity-analysis results.")
    # Interpolate between the scenario rows provided in the CSV.
    sens_sorted = sens.sort_values("Price Sensitivity")
    x = sens_sorted["Price Sensitivity"].to_numpy(dtype=float)
    def interpolate(column):
        y = numeric(sens_sorted[column]).to_numpy(dtype=float)
        return float(np.interp(selected_sensitivity, x, y))
    before = interpolate("Congestion Before (%)")
    after = interpolate("Congestion After (%)")
    reduction = interpolate("Congestion Reduction (pp)")
    shift = interpolate("Demand Shift (%)")
    transferred = interpolate("Transferred Demand")
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Congestion before", f"{before:.2f}%")
    s2.metric("Simulated after", f"{after:.2f}%", delta=f"-{reduction:.3f} pp")
    s3.metric("Demand shifted", f"{shift:.2f}%")
    s4.metric("Transferred demand", f"{transferred:,.0f}")
    scen = pd.DataFrame({"Stage":["Before pricing response","After pricing response"],
                         "Congestion (%)":[before,after]})
    sf = px.bar(scen, x="Stage", y="Congestion (%)", color="Stage",
                color_discrete_sequence=["#22c7b8","#b7f36b"], text_auto=".2f")
    sf.update_layout(showlegend=False, height=280, paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)", font_color="#dce9e6",
                     margin=dict(l=5,r=5,t=15,b=5))
    st.plotly_chart(sf, use_container_width=True)
    st.caption("This is an interpolation of the three precomputed scenarios in sensitivity_analysis.csv (0.2, 0.5 and 0.8), not a live retraining or inference call to an ML model.")

    with st.expander("Model and decision-support diagnostics"):
        st.markdown("**Feature importance (SHAP)**")
        shap = df_for("shap").copy()
        shap["Mean_Absolute_SHAP"] = numeric(shap["Mean_Absolute_SHAP"])
        shap = shap.sort_values("Mean_Absolute_SHAP", ascending=True).tail(15)
        f = px.bar(shap, x="Mean_Absolute_SHAP", y="Feature", orientation="h",
                   color_discrete_sequence=["#22c7b8"])
        f.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#dce9e6", margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(f, use_container_width=True)
        st.markdown("**ST-GNN comparison file**")
        comp = df_for("model_comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("The uploaded comparison CSV reports evaluation metrics only; it is not a serialized ST-GNN model.")

# ----------------------------- Footer ----------------------------------------
st.divider()
st.markdown(
    "<div class='small-muted'>VoltIQ demo · Built with Streamlit and Plotly · "
    "Predictions and scenario outputs are read from the supplied CSV files. "
    "No real-time charger telemetry or geographic coordinates are included in this dataset.</div>",
    unsafe_allow_html=True
)
