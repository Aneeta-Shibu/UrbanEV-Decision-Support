
section("📊 Network overview", "A quick summary of the currently selected records.")
avg_occupancy = pd.to_numeric(df[occupancy_col], errors="coerce").mean() if occupancy_col else np.nan
avg_risk = probability_pct.mean() if probability_pct is not None else np.nan
avg_price = pd.to_numeric(df[price_col], errors="coerce").mean() if price_col else np.nan
high_count = int(df["risk_category_display"].isin(["High", "Very High"]).sum()) if "risk_category_display" in df.columns else 0

m1, m2, m3, m4 = st.columns(4)
metric_card(m1, "📈 Average occupancy", f"{avg_occupancy:.2f}" if pd.notna(avg_occupancy) else "N/A",
            "Average occupancy in selected records", "card-blue")
metric_card(m2, "🚦 Average congestion risk", f"{avg_risk:.2f}%" if pd.notna(avg_risk) else "N/A",
            "Model-estimated risk, not a guarantee", "card-orange")
metric_card(m3, "⚠️ High-risk records", f"{high_count:,}",
            "Records labelled High or Very High", "card-purple")
metric_card(m4, "💰 Recommended price", f"{avg_price:.3f}" if pd.notna(avg_price) else "N/A",
            "Average price in dataset units", "card-green")
st.caption(f"Showing **{len(df):,} records** after applying your selected filters.")

overview_tab, alerts_tab, pricing_tab, zones_tab, technical_tab = st.tabs([
    "🏠 Overview", "🚦 Congestion alerts", "💰 Dynamic pricing", "📍 Charging zones", "🧠 Technical analysis"
])

with overview_tab:
    section("Congestion risk at a glance", "See how records are distributed across risk levels.")
    if "risk_category_display" in df.columns:
        order = ["Low", "Moderate", "High", "Very High", "Unknown"]
        risk_counts = df["risk_category_display"].value_counts().reindex(order, fill_value=0)
        risk_counts = risk_counts[risk_counts > 0]
        st.bar_chart(risk_counts, color="#635bdb")
        rcols = st.columns(4)
        for col, label, emoji in zip(rcols, ["Low", "Moderate", "High", "Very High"], ["🟢", "🟡", "🟠", "🔴"]):
            col.metric(f"{emoji} {label}", f"{int((df['risk_category_display'] == label).sum()):,}")
    else:
        st.info("Congestion probability was not found in the loaded data.")
    st.markdown("""
    **How to read the risk levels**
    - 🟢 **Low:** congestion is less likely according to the model.
    - 🟡 **Moderate:** monitor demand in this zone.
    - 🟠 **High:** prepare for possible congestion.
    - 🔴 **Very High:** prioritise this zone for congestion management.
    """)

with alerts_tab:
    section("🚦 Congestion alerts", "Review records flagged by the model for closer attention.")
    if "risk_category_display" not in df.columns:
        st.info("Alerts unavailable because the probability field was not found.")
    else:
        attention = df[df["risk_category_display"].isin(["High", "Very High"])].copy()
        if attention.empty:
            st.success("No High or Very High risk records match the current filters.")
        else:
            st.warning(f"{len(attention):,} records are flagged as High or Very High risk.")
            preferred = ["time", "zone_id", "occupancy", prob_col, "risk_percentage",
                         "risk_category", "risk_category_display", "dynamic_price",
                         "price_change", "price_increase_pct", "recommended_action", "operational_response"]
            visible = [c for c in preferred if c and c in attention.columns]
            attention = attention.sort_values("risk_percentage", ascending=False)
            st.dataframe(attention[visible], use_container_width=True, height=390)
            st.download_button("⬇️ Download congestion alerts", data=attention.to_csv(index=False).encode("utf-8"),
                file_name="urbanev_congestion_alerts.csv", mime="text/csv")

with pricing_tab:
    section("💰 Dynamic pricing", "Review recommended prices over time and the size of price adjustments.")
    if price_col:
        price_trend = df.dropna(subset=["time"]).copy() if "time" in df.columns else pd.DataFrame()
        if not price_trend.empty:
            price_trend["day"] = price_trend["time"].dt.floor("D")
            st.markdown("#### Recommended price over time")
            st.line_chart(price_trend.groupby("day")[price_col].mean(), color="#7c3aed")
        else:
            st.info("Time data is unavailable for a price trend.")
        if "price_increase_pct" in df.columns:
            adjustment = pd.to_numeric(df["price_increase_pct"], errors="coerce").dropna()
            if not adjustment.empty:
                st.markdown("#### Price adjustment distribution")
                st.bar_chart(adjustment.value_counts().sort_index(), color="#0d9488")
        if "service_price" in df.columns:
            current_avg = pd.to_numeric(df["service_price"], errors="coerce").mean()
            recommended_avg = pd.to_numeric(df[price_col], errors="coerce").mean()
            p1, p2, p3 = st.columns(3)
            p1.metric("Average current price", f"{current_avg:.3f}" if pd.notna(current_avg) else "N/A")
            p2.metric("Average recommended price", f"{recommended_avg:.3f}" if pd.notna(recommended_avg) else "N/A")
            if pd.notna(current_avg) and current_avg != 0 and pd.notna(recommended_avg):
                p3.metric("Average price change", f"{((recommended_avg-current_avg)/current_avg*100):+.1f}%")
            else:
                p3.metric("Average price change", "N/A")
        st.caption("Price units follow the source dataset. Recommendations are simulated, not live prices.")
    else:
        st.info("A recommended price column was not found.")

with zones_tab:
    section("📍 Charging zone information", "Explore zone locations and metadata.")
    if zones is not None:
        if {"longitude", "latitude"}.issubset(zones.columns):
            map_data = zones[["latitude", "longitude"]].copy()
            map_data["latitude"] = pd.to_numeric(map_data["latitude"], errors="coerce")
            map_data["longitude"] = pd.to_numeric(map_data["longitude"], errors="coerce")
            map_data = map_data.dropna()
            if not map_data.empty:
                st.markdown("#### Charging zone map")
                st.map(map_data)
        st.markdown("#### Zone details")
        st.dataframe(zones, use_container_width=True, height=300)
    else:
        st.info("Zone metadata file was not found.")

with technical_tab:
    section("🧠 Technical analysis", "Model transparency and feature importance for project evaluation.")
    st.markdown("#### Model explainability (SHAP)")
    if shap is not None and not shap.empty:
        numeric_cols = shap.select_dtypes(include=np.number).columns.tolist()
        feature_col = next((c for c in shap.columns if "feature" in c.lower()), shap.columns[0] if len(shap.columns) else None)
        importance_col = next((c for c in numeric_cols if "importance" in c.lower() or "shap" in c.lower()), numeric_cols[0] if numeric_cols else None)
        if feature_col and importance_col:
            plot_data = shap[[feature_col, importance_col]].dropna().sort_values(importance_col, ascending=False).head(15)
            st.markdown("**Most influential features**")
            st.bar_chart(plot_data.set_index(feature_col), color="#db2777")
        st.dataframe(shap, use_container_width=True)
    else:
        st.info("SHAP feature-importance data was not found.")
    st.markdown("#### Export selected records")
    st.download_button("⬇️ Download filtered decision-support data",
        data=df.to_csv(index=False).encode("utf-8"), file_name="urbanev_decision_support_filtered.csv",
        mime="text/csv")

st.markdown("---")
st.markdown("""<div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:14px;padding:1rem 1.2rem;color:#3730a3;">
<strong>ℹ️ Important note</strong><br>Congestion probabilities are model estimates. Pricing recommendations are simulation-based and should be validated before live deployment.
