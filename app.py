"""
Charge Network - EV charging dashboard.

Two experiences in one app:
  * EV Driver        - pick a zone, check predicted congestion, compare prices,
                       and get a recommended nearby alternative.
  * Network Operator - monitor all zones, spot hotspots, and try pricing
                       sensitivity scenarios.

The forecasting models (XGBoost / ST-GNN) are represented by the exported
prediction files. They run behind the scenes; the UI only shows outcomes.

Expected data files (in ./data/ or next to app.py):
  dynamic_pricing_simulation.csv
  nearest_alternative_zones__1_.csv
  zone_neighbours.csv
  final_pricing_comparison__1_.csv
  sensitivity_analysis__1_.csv
  residual_stgnn_comparison__1_.csv
  shap_feature_importance.csv
Optional:
  zone_coordinates.csv  (zone_id, latitude, longitude) for a true geographic map
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Charge Network", page_icon="⚡", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_SEARCH_DIRS = [APP_DIR / "data", APP_DIR]

FILES = {
    "pricing": "dynamic_pricing_simulation.csv",
    "alt": "nearest_alternative_zones__1_.csv",
    "nb": "zone_neighbours.csv",
    "final": "final_pricing_comparison__1_.csv",
    "sens": "sensitivity_analysis__1_.csv",
    "resid": "residual_stgnn_comparison__1_.csv",
    "shap": "shap_feature_importance.csv",
    "coords": "zone_coordinates.csv",
}

# Palette: dark charcoal base, teal and electric-green accents.
BG = "#1C2127"
PANEL = "#20262D"
CARD = "#262D35"
BORDER = "#323B45"
TEXT = "#E8EEF2"
MUTED = "#8B98A5"
TEAL = "#14B8A6"
GREEN = "#3DFF9A"
AMBER = "#E5C15A"
ORANGE = "#F59E42"
CORAL = "#FF5C6C"

RISK_COLOR = {"Low": TEAL, "Moderate": AMBER, "High": ORANGE, "Very High": CORAL}
CONG_SCALE = [[0.0, TEAL], [0.4, AMBER], [0.6, ORANGE], [1.0, CORAL]]

CROWDED = 0.40  # congestion probability at which a zone counts as busy
CAPACITY_QUANTILE = 0.95  # "typical capacity" = 95th percentile occupancy per zone
W_CONGESTION, W_DISTANCE, W_PRICE = 0.5, 0.3, 0.2  # alternative scoring weights

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: __BG__;}
section[data-testid="stSidebar"] {background: __PANEL__; border-right: 1px solid __BORDER__;}
h1, h2, h3, h4 {color: __TEXT__ !important; letter-spacing: -0.01em;}
.card {background: __CARD__; border: 1px solid __BORDER__; border-radius: 14px;
       padding: 14px 16px; margin-bottom: 12px;}
.card.rec {border-color: __GREEN__; box-shadow: 0 0 0 1px __GREEN__66, 0 0 26px __GREEN__1f;}
.card .label {color: __MUTED__; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;}
.card .value {color: __TEXT__; font-size: 1.7rem; font-weight: 650; line-height: 1.2; margin-top: 2px;}
.card .sub {color: __MUTED__; font-size: .84rem; margin-top: 3px;}
.badge {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .68rem;
        font-weight: 700; letter-spacing: .04em; background: __GREEN__; color: #0B1A12;
        margin-left: 8px; vertical-align: middle; text-transform: uppercase;}
.hero {color: __MUTED__; font-size: .95rem; margin-top: -8px; margin-bottom: 16px;}
.note {border-left: 3px solid __TEAL__; background: __CARD__; padding: 10px 14px;
       border-radius: 8px; color: __TEXT__; margin-bottom: 12px; font-size: .92rem;}
.note.warn {border-left-color: __CORAL__;}
div[data-testid="stMetric"] {background: __CARD__; border: 1px solid __BORDER__;
       border-radius: 14px; padding: 12px 16px;}
div[data-testid="stMetricLabel"] p {color: __MUTED__;}
</style>
"""
CSS = (
    CSS.replace("__BG__", BG)
    .replace("__PANEL__", PANEL)
    .replace("__CARD__", CARD)
    .replace("__BORDER__", BORDER)
    .replace("__TEXT__", TEXT)
    .replace("__MUTED__", MUTED)
    .replace("__TEAL__", TEAL)
    .replace("__GREEN__", GREEN)
    .replace("__CORAL__", CORAL)
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find(name: str, required: bool = True) -> Path | None:
    for base in DATA_SEARCH_DIRS:
        path = base / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(
            f"Could not find '{name}'. Place it in a 'data/' folder next to app.py."
        )
    return None


def risk_of(p: float) -> str:
    if p >= 0.6:
        return "Very High"
    if p >= 0.4:
        return "High"
    if p >= 0.2:
        return "Moderate"
    return "Low"


@st.cache_data(show_spinner="Loading zone data…")
def load_pricing() -> pd.DataFrame:
    df = pd.read_csv(
        find(FILES["pricing"]),
        usecols=[
            "time",
            "zone_id",
            "occupancy",
            "service_price",
            "dynamic_price",
            "price_increase_pct",
            "predicted_congestion_probability",
        ],
    )
    df["time"] = pd.to_datetime(df["time"])
    df["zone_id"] = df["zone_id"].astype(int)
    df = df.rename(columns={"predicted_congestion_probability": "congestion"})

    capacity = df.groupby("zone_id")["occupancy"].quantile(CAPACITY_QUANTILE).clip(lower=1)
    df["capacity"] = df["zone_id"].map(capacity)
    df["load"] = (df["occupancy"] / df["capacity"]).clip(0, 1)
    df["risk"] = np.select(
        [df["congestion"] >= 0.6, df["congestion"] >= 0.4, df["congestion"] >= 0.2],
        ["Very High", "High", "Moderate"],
        default="Low",
    )
    return df.sort_values(["time", "zone_id"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_alternatives() -> pd.DataFrame:
    alt = pd.read_csv(find(FILES["alt"]))
    return alt.astype({"zone_id": int, "alternative_zone": int})


@st.cache_data(show_spinner=False)
def load_neighbours() -> pd.DataFrame:
    nb = pd.read_csv(find(FILES["nb"]))
    return nb.astype({"zone_id": int, "neighbour_zone_id": int})


def _mds_layout(zone_ids: tuple, nb: pd.DataFrame) -> pd.DataFrame:
    """Classical multidimensional scaling on graph shortest-path distances.

    The dataset has no coordinates, so this places zones on a 2D map where
    nearby zones (by neighbour distance) appear close together.
    """
    idx = {z: i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)
    rows = nb["zone_id"].map(idx)
    cols = nb["neighbour_zone_id"].map(idx)
    ok = rows.notna() & cols.notna()
    r = rows[ok].astype(int).to_numpy()
    c = cols[ok].astype(int).to_numpy()
    w = nb.loc[ok, "distance_km"].to_numpy() + 1e-6

    graph = csr_matrix((w, (r, c)), shape=(n, n))
    dist = shortest_path(graph, directed=False)
    finite = dist[np.isfinite(dist)]
    dist[~np.isfinite(dist)] = finite.max() * 1.2 if finite.size else 1.0
    np.fill_diagonal(dist, 0.0)

    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.clip(vals[top], 1e-9, None))
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=list(zone_ids))


@st.cache_data(show_spinner=False)
def load_layout(zone_ids: tuple) -> tuple[pd.DataFrame, str]:
    coords_path = find(FILES["coords"], required=False)
    if coords_path is not None:
        coords = pd.read_csv(coords_path)
        coords["zone_id"] = coords["zone_id"].astype(int)
        coords = coords.set_index("zone_id")
        pos = pd.DataFrame(
            {"x": coords["longitude"], "y": coords["latitude"]}
        ).reindex(list(zone_ids))
        if pos.notna().all().all():
            return pos, "Map uses real zone coordinates."
    pos = _mds_layout(zone_ids, load_neighbours())
    return pos, "Map layout is approximate, derived from distances between neighbouring zones."


@st.cache_data(show_spinner=False)
def load_final_pricing() -> pd.DataFrame:
    return pd.read_csv(find(FILES["final"])).set_index("Pricing Strategy")


@st.cache_data(show_spinner=False)
def load_sensitivity() -> pd.DataFrame:
    return pd.read_csv(find(FILES["sens"])).sort_values("Price Sensitivity").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_model_info():
    resid = pd.read_csv(find(FILES["resid"]), index_col=0)
    shap = pd.read_csv(find(FILES["shap"]))
    return resid, shap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def snapshot(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df["time"] == t].set_index("zone_id").sort_index()


def zone_label(z: int) -> str:
    return f"Zone {z}"


def action_for(pct: float) -> str:
    return "Hold price" if pct <= 0 else f"Raise price {pct * 100:.0f}%"


def card(label, value, sub="", accent=None, rec=False, badge=""):
    style = f' style="border-left:3px solid {accent}"' if accent else ""
    cls = "card rec" if rec else "card"
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="{cls}"{style}><div class="label">{label}{badge_html}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str, warn: bool = False):
    cls = "note warn" if warn else "note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.14, x=0),
        font=dict(color=MUTED),
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def candidates(zone: int, snap: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    """Nearby alternatives scored on congestion (lower is better), distance and price."""
    c = alt.loc[alt["zone_id"] == zone].sort_values("distance_km").head(8).copy()
    c = c[c["alternative_zone"].isin(snap.index)].reset_index(drop=True)
    if c.empty:
        return c
    c["congestion"] = c["alternative_zone"].map(snap["congestion"])
    c["price"] = c["alternative_zone"].map(snap["dynamic_price"])
    c["vehicles"] = c["alternative_zone"].map(snap["occupancy"])
    c["risk"] = c["congestion"].map(risk_of)

    d_span = c["distance_km"].max() or 1.0
    p_min, p_max = c["price"].min(), c["price"].max()
    p_norm = (c["price"] - p_min) / (p_max - p_min) if p_max > p_min else 0.0
    c["score"] = (
        W_CONGESTION * c["congestion"]
        + W_DISTANCE * c["distance_km"] / d_span
        + W_PRICE * p_norm
    )
    return c.sort_values("score").reset_index(drop=True)


def on_map_click():
    """Callback: when a zone is clicked on the driver map, select it."""
    try:
        state = st.session_state.get("map_chart") or {}
        points = (state.get("selection") or {}).get("points") or []
        if points and points[0].get("customdata") is not None:
            cd = points[0]["customdata"]
            zid = cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd
            st.session_state["zone_pick"] = int(zid)
    except Exception:
        pass


def zone_map(pos, snap, selected, alt_ids, operator: bool) -> go.Figure:
    p = pos.loc[snap.index]
    hover = [
        f"<b>Zone {z}</b><br>Congestion {c * 100:.0f}% · {r}<br>Vehicles {int(o)}"
        for z, c, r, o in zip(snap.index, snap.congestion, snap.risk, snap.occupancy)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p["x"],
            y=p["y"],
            mode="markers",
            customdata=snap.index.values,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=10 + 14 * snap["load"].values,
                color=snap["congestion"].values,
                colorscale=CONG_SCALE,
                cmin=0,
                cmax=1,
                opacity=0.92,
                showscale=operator,
                colorbar=dict(
                    title="Congestion",
                    tickformat=".0%",
                    thickness=10,
                    len=0.6,
                    bgcolor="rgba(0,0,0,0)",
                ),
            ),
            showlegend=False,
        )
    )
    if alt_ids:
        a = p.loc[[z for z in alt_ids if z in p.index]]
        fig.add_trace(
            go.Scatter(
                x=a["x"],
                y=a["y"],
                mode="markers",
                customdata=a.index.values,
                hoverinfo="skip",
                marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color=GREEN, width=2.5)),
                showlegend=False,
            )
        )
    if selected is not None and selected in p.index:
        s = p.loc[[selected]]
        fig.add_trace(
            go.Scatter(
                x=s["x"],
                y=s["y"],
                mode="markers",
                customdata=s.index.values,
                hoverinfo="skip",
                marker=dict(size=32, color="rgba(0,0,0,0)", line=dict(color="#FFFFFF", width=2)),
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT),
        dragmode="pan",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
    )
    return fig


# ---------------------------------------------------------------------------
# EV Driver experience
# ---------------------------------------------------------------------------
def render_alternatives(c: pd.DataFrame):
    good = c[c["congestion"] < CROWDED]
    best = good.iloc[0] if len(good) else c.iloc[0]
    if len(good):
        lead = "Picked for the best balance of low congestion, short distance and fair price."
    else:
        lead = "All nearby zones are busy too. This is the least congested option."

    st.markdown("#### Recommended alternative")
    card(
        "Best option nearby",
        f"Zone {int(best['alternative_zone'])}",
        f"{best['distance_km']:.1f} km away · congestion {best['congestion'] * 100:.0f}% · "
        f"price {best['price']:.2f}<br>{lead}",
        accent=RISK_COLOR[best["risk"]],
        rec=True,
        badge="Recommended",
    )

    others = c[c["alternative_zone"] != best["alternative_zone"]].head(3)
    if not others.empty:
        st.markdown("#### Other nearby zones")
        cols = st.columns(len(others))
        for col, (_, r) in zip(cols, others.iterrows()):
            with col:
                card(
                    f"Zone {int(r['alternative_zone'])}",
                    f"{r['distance_km']:.1f} km",
                    f"Congestion {r['congestion'] * 100:.0f}%<br>Price {r['price']:.2f}",
                    accent=RISK_COLOR[r["risk"]],
                )


def render_driver(df: pd.DataFrame, t: pd.Timestamp, pos, alt: pd.DataFrame, layout_note: str):
    snap = snapshot(df, t)
    zones = snap.index.tolist()
    if st.session_state.get("zone_pick") not in zones:
        st.session_state["zone_pick"] = zones[0]
    zone = int(st.session_state["zone_pick"])
    row = snap.loc[zone]
    crowded = row["congestion"] >= CROWDED
    cands = candidates(zone, snap, alt)

    st.markdown("# Find a charge")
    st.markdown(
        f'<div class="hero">Predictions for {t:%A %d %b, %H:%M}. Pick a zone to check '
        f"availability, compare prices and find a better option nearby.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.caption("Click a zone on the map, or choose one below.")
        alt_ids = cands["alternative_zone"].tolist() if (crowded and not cands.empty) else None
        st.plotly_chart(
            zone_map(pos, snap, zone, alt_ids, operator=False),
            use_container_width=True,
            key="map_chart",
            on_select=on_map_click,
            selection_mode="points",
            config={"displayModeBar": False},
        )
        st.selectbox("Zone", zones, key="zone_pick", format_func=zone_label)
        if alt_ids:
            st.caption("Green rings mark nearby alternatives.")

    with right:
        st.subheader(zone_label(zone))
        if crowded:
            note("This zone is likely to be busy during this hour. Nearby options are shown below.", warn=True)
        else:
            note("Looks clear. A good time to charge here.")

        card(
            "Vehicles in zone",
            f"{int(row['occupancy'])}",
            f"About {row['load'] * 100:.0f}% of typical capacity",
            accent=RISK_COLOR[row["risk"]],
        )
        card(
            "Predicted congestion",
            f"{row['congestion'] * 100:.0f}%",
            f"{row['risk']} risk this hour",
            accent=RISK_COLOR[row["risk"]],
        )
        surge = (
            f"Fixed {row['service_price']:.2f} · +{row['price_increase_pct'] * 100:.0f}% surge"
            if row["price_increase_pct"] > 0
            else f"Fixed {row['service_price']:.2f} · no surge"
        )
        card("Price right now", f"{row['dynamic_price']:.2f}", surge)

    st.divider()

    st.markdown("### Nearby alternatives")
    if cands.empty:
        st.info("No alternative zones are on record for this zone.")
    elif crowded:
        render_alternatives(cands)
    else:
        st.caption("No need to move. This zone is comfortable right now.")
        with st.expander("Show nearby zones anyway"):
            render_alternatives(cands)

    st.divider()

    st.markdown("### Price and availability over the next 24 hours")
    horizon = df[(df["zone_id"] == zone) & (df["time"] >= t) & (df["time"] < t + pd.Timedelta(hours=24))]
    if horizon.empty:
        st.caption("No forecast beyond the end of the data.")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fig.add_scatter(
            x=horizon["time"], y=horizon["service_price"], name="Fixed price",
            line=dict(color=MUTED, dash="dot", width=2),
        )
        fig.add_scatter(
            x=horizon["time"], y=horizon["dynamic_price"], name="Dynamic price",
            line=dict(color=GREEN, width=3),
        )
        st.plotly_chart(style_fig(fig), use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        fig.add_scatter(
            x=horizon["time"], y=horizon["congestion"], name="Predicted congestion",
            line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(20,184,166,0.15)",
        )
        fig.add_hline(
            y=CROWDED, line_dash="dash", line_color=CORAL,
            annotation_text="Busy threshold", annotation_font_color=CORAL,
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(style_fig(fig), use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Dynamic prices rise during busy periods, by up to 15% at very high congestion. "
        + layout_note
    )


# ---------------------------------------------------------------------------
# Network Operator experience
# ---------------------------------------------------------------------------
def render_operator(df, t, pos, final, sens, resid, shap, layout_note):
    snap = snapshot(df, t)
    at_risk = snap[snap["congestion"] >= CROWDED]

    st.markdown("# Network operations")
    st.markdown(
        f'<div class="hero">Live view for {t:%A %d %b, %H:%M}. Monitor zones, spot hotspots '
        f"and test pricing strategies.</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("Zones monitored", f"{len(snap)}", "Across the network")
    with k2:
        card("Zones at risk", f"{len(at_risk)}", "High or very high congestion",
             accent=CORAL if len(at_risk) else TEAL)
    with k3:
        card("Average congestion", f"{snap['congestion'].mean() * 100:.0f}%", "Network-wide, this hour")
    with k4:
        card("Average surge", f"{snap['price_increase_pct'].mean() * 100:.1f}%",
             "Applied across all zones")

    map_col, hot_col = st.columns([1.4, 1], gap="large")
    with map_col:
        st.subheader("Congestion map")
        st.plotly_chart(
            zone_map(pos, snap, None, None, operator=True),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(layout_note)
    with hot_col:
        st.subheader("Hotspots")
        hs = snap.sort_values("congestion", ascending=False).head(10).reset_index()
        hs_df = pd.DataFrame(
            {
                "Zone": hs["zone_id"].map(zone_label),
                "Vehicles": hs["occupancy"].astype(int),
                "Congestion": hs["congestion"] * 100,
                "Price now": hs["dynamic_price"].round(2),
                "Action": hs["price_increase_pct"].map(action_for),
            }
        )
        st.dataframe(
            hs_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Congestion": st.column_config.ProgressColumn(
                    "Congestion", min_value=0, max_value=100, format="%.0f%%"
                ),
            },
        )

    st.divider()

    st.subheader("Pricing strategy")
    fixed = final.loc["Fixed Pricing"]
    dyn = final.loc["Dynamic Pricing"]
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Average price, fixed", f"{fixed['Mean Price']:.2f}")
    with p2:
        st.metric(
            "Average price, dynamic",
            f"{dyn['Mean Price']:.2f}",
            f"{(dyn['Mean Price'] / fixed['Mean Price'] - 1) * 100:+.1f}% vs fixed",
        )
    with p3:
        st.metric(
            "Peak price, dynamic",
            f"{dyn['Maximum Price']:.2f}",
            f"{(dyn['Maximum Price'] / fixed['Maximum Price'] - 1) * 100:+.1f}% vs fixed",
        )
    if "Advanced Dynamic Pricing (Model 2)" in final.index:
        adv = final.loc["Advanced Dynamic Pricing (Model 2)"]
        st.caption(
            f"Advanced dynamic pricing (Model 2) averages {adv['Mean Price']:.2f} "
            f"with a peak of {adv['Maximum Price']:.2f}, a small further gain over standard dynamic pricing."
        )

    st.divider()

    st.subheader("Try a pricing scenario")
    st.caption(
        "Price sensitivity is how strongly drivers respond to price changes. "
        "Higher sensitivity moves more demand from busy zones to quieter ones."
    )
    levels = sens["Price Sensitivity"].tolist()
    chosen = st.select_slider(
        "Price sensitivity",
        options=levels,
        value=levels[len(levels) // 2],
        key="sens_pick",
    )
    r = sens.loc[sens["Price Sensitivity"] == chosen].iloc[0]

    s1, s2, s3 = st.columns(3)
    with s1:
        card("Congestion before", f"{r['Congestion Before (%)']:.2f}%", "Network average")
    with s2:
        card("Congestion after", f"{r['Congestion After (%)']:.2f}%",
             f"{r['Congestion Reduction (pp)']:.2f} points lower", accent=GREEN)
    with s3:
        conserved = "Demand conserved" if bool(r["Demand Conserved"]) else "Demand changed"
        card("Demand moved", f"{r['Demand Shift (%)']:.1f}%", conserved)

    fig = go.Figure()
    labels = [str(v) for v in sens["Price Sensitivity"]]
    fig.add_bar(name="Before", x=labels, y=sens["Congestion Before (%)"], marker_color=MUTED)
    fig.add_bar(name="After", x=labels, y=sens["Congestion After (%)"], marker_color=GREEN)
    fig.update_layout(barmode="group", xaxis_title="Price sensitivity", yaxis_title="Congestion (%)")
    st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    with st.expander("How the forecasts work (for analysts)"):
        st.write(
            "Congestion probabilities come from the forecasting models and drive the surge rules: "
            "Moderate adds 5%, High adds 10%, and Very High adds 15%. The main screens show outcomes "
            "only, so model scores live here."
        )
        top = shap.head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["Mean_Absolute_SHAP"], y=top["Feature"], orientation="h",
                               marker_color=TEAL))
        fig.update_layout(xaxis_title="Mean absolute SHAP value", title="Top features driving predictions")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Model comparison on held-out data**")
        st.dataframe(resid.round(3), use_container_width=True)
        st.caption(
            "The residual ST-GNN gives the lowest RMSE and highest R². The persistence baseline "
            "has the lowest MAE. Both results are shown so the trade-off is visible."
        )


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    df = load_pricing()
    zone_ids = tuple(sorted(df["zone_id"].unique().tolist()))
    pos, layout_note = load_layout(zone_ids)
    alt = load_alternatives()
    final = load_final_pricing()
    sens = load_sensitivity()
    resid, shap = load_model_info()

    times = sorted(pd.to_datetime(df["time"].unique()))

    with st.sidebar:
        st.markdown("## ⚡ Charge Network")
        mode = st.radio("Experience", ["EV Driver", "Network Operator"], key="mode")
        st.divider()
        t = st.select_slider(
            "Time",
            options=times,
            value=times[max(0, len(times) - 24)],  # leaves a full 24h of forecast ahead
            format_func=lambda x: x.strftime("%a %d %b · %H:%M"),
            key="time",
        )
        st.caption(
            f"Data: {times[0]:%d %b} – {times[-1]:%d %b %Y} · "
            f"{df['zone_id'].nunique()} zones · hourly"
        )

    if mode == "EV Driver":
        render_driver(df, t, pos, alt, layout_note)
    else:
        render_operator(df, t, pos, final, sens, resid, shap, layout_note)


main()
