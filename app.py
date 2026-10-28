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

st.title("UrbanEV Intelligent Decision-Support System")

st.caption(
    "Predictive Congestion Modeling and Dynamic Pricing "
    "Optimization for Electric Vehicle Charging Networks"
)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    files = {
        "decision": "intelligent_decision_support.csv",
        "pricing": "dynamic_pricing_simulation.csv",
        "zones": "zone-information.csv",
        "shap": "shap_feature_importance.csv"
    }

    loaded = {}

    for key, filename in files.items():

        path = BASE / filename

        if path.exists():
            loaded[key] = pd.read_csv(path)
        else:
            loaded[key] = None

    return loaded


try:
    data = load_data()

except Exception as e:
    st.error(f"Error loading project data: {e}")
    st.stop()


decision = data["decision"]
pricing = data["pricing"]
zones = data["zones"]
shap_data = data["shap"]


if decision is None and pricing is None:

    st.error(
        "Decision-support data was not found. "
        "Please check that the CSV files are in the same "
        "folder as app.py."
    )

    st.stop()


# =========================================================
# 4. PREPARE MAIN DATASET
# =========================================================

df = (
    decision.copy()
    if decision is not None
    else pricing.copy()
)

df.columns = df.columns.astype(str).str.strip()


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


# Convert numerical columns when available.

numeric_columns = [
    "occupancy",
    "predicted_congestion_probability",
    "congestion_probability",
    "service_price",
    "dynamic_price",
    "price_increase_pct",
    "risk_priority",
    "congestion_probability_pct"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# =========================================================
# 5. SIDEBAR FILTERS
# =========================================================

st.sidebar.title("Dashboard Filters")


# Date filter

if "time" in df.columns and df["time"].notna().any():

    valid_times = df["time"].dropna()

    min_date = valid_times.min().date()
    max_date = valid_times.max().date()

    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):

        start_date, end_date = selected_dates

        df = df[
            df["time"].dt.date.between(
                start_date,
                end_date
            )
        ].copy()


# Zone filter

selected_zones = []

if "zone_id" in df.columns:

    available_zones = sorted(
        df["zone_id"].dropna().unique().tolist()
    )

    selected_zones = st.sidebar.multiselect(
        "Select charging zones",
        options=available_zones,
        default=[]
    )

    if selected_zones:

        df = df[
            df["zone_id"].isin(selected_zones)
        ].copy()


if df.empty:

    st.warning(
        "No records match the selected filters. "
        "Please change the date range or zone selection."
    )

    st.stop()


# =========================================================
# 6. NETWORK OVERVIEW
# =========================================================

st.header("Network Overview")

col1, col2, col3, col4 = st.columns(4)


# Total records

col1.metric(
    "Total Records",
    f"{len(df):,}"
)


# Number of zones

if "zone_id" in df.columns:

    col2.metric(
        "Charging Zones",
        f"{df['zone_id'].nunique():,}"
    )

else:

    col2.metric(
        "Charging Zones",
        "N/A"
    )


# Average occupancy

if "occupancy" in df.columns:

    average_occupancy = df["occupancy"].mean()

    col3.metric(
        "Average Occupancy",
        (
            f"{average_occupancy:.2f}"
            if pd.notna(average_occupancy)
            else "N/A"
        )
    )

else:

    col3.metric(
        "Average Occupancy",
        "N/A"
    )


# Time coverage

if "time" in df.columns:

    timestamps = df["time"].nunique()

    col4.metric(
        "Time Points",
        f"{timestamps:,}"
    )

else:

    col4.metric(
        "Time Points",
        "N/A"
    )


st.divider()


# =========================================================
# 7. CONGESTION RISK ANALYSIS
# =========================================================

st.header("Congestion Risk Analysis")


# Determine the probability column.

prob_col = next(
    (
        col
        for col in [
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

    # The model output is normally a fraction between 0 and 1.
    # Convert it to a percentage for display.

    if probability.notna().any():

        if probability.max() <= 1:

            probability_pct = probability * 100

        else:

            probability_pct = probability

        df["risk_percentage"] = probability_pct

else:

    # Use the existing percentage column if available.

    if "congestion_probability_pct" in df.columns:

        df["risk_percentage"] = pd.to_numeric(
            df["congestion_probability_pct"],
            errors="coerce"
        )

        probability_pct = df["risk_percentage"]


# Risk classification

def classify_risk(value):

    if pd.isna(value):
        return "Unknown"

    if value < 20:
        return "Low"

    elif value < 40:
        return "Moderate"

    elif value < 60:
        return "High"

    else:
        return "Very High"


if probability_pct is not None:

    df["risk_category_display"] = (
        df["risk_percentage"].apply(classify_risk)
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
        .reindex(risk_order, fill_value=0)
    )


    # Risk metrics

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Average Congestion Probability",
        f"{df['risk_percentage'].mean():.2f}%"
    )

    high_risk_records = df[
        df["risk_category_display"].isin(
            ["High", "Very High"]
        )
    ]

    risk_col2.metric(
        "High-Risk Records",
        f"{len(high_risk_records):,}"
    )

    if "zone_id" in high_risk_records.columns:

        high_risk_zones = (
            high_risk_records["zone_id"].nunique()
        )

    else:

        high_risk_zones = len(high_risk_records)

    risk_col3.metric(
        "Zones Requiring Attention",
        f"{high_risk_zones:,}"
    )


    # Risk distribution chart

    chart_col, summary_col = st.columns(
        [2, 1]
    )

    with chart_col:

        st.subheader("Congestion Risk Distribution")

        st.bar_chart(risk_counts)


    with summary_col:

        st.subheader("Risk Summary")

        risk_summary = pd.DataFrame({
            "Risk Category": risk_order,
            "Number of Records": [
                int(risk_counts[category])
                for category in risk_order
            ]
        })

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )


    # Records requiring attention

    st.subheader("High-Risk Charging Records")

    if not high_risk_records.empty:

        display_columns = [
            "time",
            "zone_id",
            "occupancy",
            "predicted_congestion_probability",
            "congestion_risk",
            "risk_category_display",
            "service_price",
            "dynamic_price",
            "price_increase_pct",
            "recommended_action",
            "operational_response",
            "risk_priority"
        ]

        available_columns = [
            col
            for col in display_columns
            if col in high_risk_records.columns
        ]

        high_risk_display = high_risk_records[
            available_columns
        ].copy()

        high_risk_display = high_risk_display.sort_values(
            "risk_percentage",
            ascending=False
        )

        st.dataframe(
            high_risk_display,
            use_container_width=True,
            height=350,
            hide_index=True
        )

    else:

        st.info(
            "No records fall into the High or Very High "
            "categories under the selected filters."
        )


    st.caption(
        "Risk categories use probability thresholds: "
        "Low <20%, Moderate 20–<40%, "
        "High 40–<60%, and Very High ≥60%."
    )


else:

    st.warning(
        "Congestion probability was not found. "
        "Please check the probability columns in the dataset."
    )


st.divider()


# =========================================================
# 8. DYNAMIC PRICING
# =========================================================

st.header("Dynamic Pricing")


if (
    "service_price" in df.columns
    and "dynamic_price" in df.columns
):

    df["service_price"] = pd.to_numeric(
        df["service_price"],
        errors="coerce"
    )

    df["dynamic_price"] = pd.to_numeric(
        df["dynamic_price"],
        errors="coerce"
    )


    # Price summary metrics

    price_col1, price_col2, price_col3 = st.columns(3)

    current_price = df["service_price"].mean()
    recommended_price = df["dynamic_price"].mean()

    price_col1.metric(
        "Average Service Price",
        (
            f"{current_price:.3f}"
            if pd.notna(current_price)
            else "N/A"
        )
    )

    price_col2.metric(
        "Average Dynamic Price",
        (
            f"{recommended_price:.3f}"
            if pd.notna(recommended_price)
            else "N/A"
        )
    )


    if "price_increase_pct" in df.columns:

        df["price_increase_pct"] = pd.to_numeric(
            df["price_increase_pct"],
            errors="coerce"
        )

        average_adjustment = (
            df["price_increase_pct"].mean()
        )

        price_col3.metric(
            "Average Price Adjustment",
            (
                f"{average_adjustment:.2f}%"
                if pd.notna(average_adjustment)
                else "N/A"
            )
        )

    else:

        price_col3.metric(
            "Average Price Adjustment",
            "N/A"
        )


    # Compare current and dynamic prices

    st.subheader("Service Price vs Dynamic Price")

    price_comparison = pd.DataFrame({
        "Price Type": [
            "Service Price",
            "Dynamic Price"
        ],
        "Average Price": [
            current_price,
            recommended_price
        ]
    })

    st.bar_chart(
        price_comparison.set_index("Price Type")
    )


    # Price trend over time

    if "time" in df.columns:

        st.subheader("Dynamic Pricing Trend")

        price_trend = df[
            ["time", "service_price", "dynamic_price"]
        ].dropna(
            subset=["time"]
        ).copy()

        if not price_trend.empty:

            daily_prices = (
                price_trend
                .set_index("time")
                .resample("D")[
                    ["service_price", "dynamic_price"]
                ]
                .mean()
                .dropna(how="all")
            )

            if not daily_prices.empty:

                st.line_chart(daily_prices)

            else:

                st.info(
                    "No daily price trends are available."
                )


    # Price adjustment distribution

    if "price_increase_pct" in df.columns:

        st.subheader("Price Adjustment Distribution")

        adjustment_data = df[
            "price_increase_pct"
        ].dropna()

        if not adjustment_data.empty:

            adjustment_counts = (
                adjustment_data
                .round(2)
                .value_counts()
                .sort_index()
            )

            st.bar_chart(adjustment_counts)

        else:

            st.info(
                "No price adjustment data is available."
            )


    # Recommended actions

    if "recommended_action" in df.columns:

        st.subheader("Recommended Operational Actions")

        action_counts = (
            df["recommended_action"]
            .fillna("Not specified")
            .value_counts()
        )

        st.bar_chart(action_counts)

        st.dataframe(
            df[
                [
                    col
                    for col in [
                        "time",
                        "zone_id",
                        "congestion_risk",
                        "service_price",
                        "dynamic_price",
                        "price_increase_pct",
                        "recommended_action",
                        "operational_response"
                    ]
                    if col in df.columns
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )


else:

    st.warning(
        "Dynamic pricing requires service_price and dynamic_price "
        "columns. Check that the correct decision-support CSV "
        "has been loaded."
    )


st.divider()


# =========================================================
# 9. CHARGING ZONE INFORMATION AND MAP
# =========================================================

st.header("Charging Zone Information")

if zones is not None and not zones.empty:

    zone_view = zones.copy()

    zone_view.columns = (
        zone_view.columns.astype(str).str.strip()
    )


    # Check the required zone identifier.

    if "TAZID" not in zone_view.columns:

        st.error(
            "TAZID was not found in zone-information.csv."
        )

    else:

        zone_view["TAZID"] = pd.to_numeric(
            zone_view["TAZID"],
            errors="coerce"
        )

        zone_view = zone_view.dropna(
            subset=["TAZID"]
        ).copy()


        # Convert numeric fields.

        for col in [
            "longitude",
            "latitude",
            "charge_count",
            "area",
            "perimeter"
        ]:

            if col in zone_view.columns:

                zone_view[col] = pd.to_numeric(
                    zone_view[col],
                    errors="coerce"
                )


        # Filter metadata to selected zones if applicable.

        if selected_zones:

            selected_ids = pd.to_numeric(
                pd.Series(selected_zones),
                errors="coerce"
            ).dropna()

            zone_view = zone_view[
                zone_view["TAZID"].isin(selected_ids)
            ].copy()


        st.write(
            f"**Charging zones displayed:** "
            f"{len(zone_view):,}"
        )


        # Zone metadata table.

        st.dataframe(
            zone_view,
            use_container_width=True,
            height=300,
            hide_index=True
        )


        # Map.

        if (
            "latitude" in zone_view.columns
            and "longitude" in zone_view.columns
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

            map_data = map_data[
                map_data["latitude"].between(-90, 90)
                & map_data["longitude"].between(-180, 180)
            ]

            if not map_data.empty:

                st.subheader("Charging Zone Locations")

                st.map(map_data)

                st.caption(
                    "Map displays charging-zone locations "
                    "using the available latitude and longitude data."
                )

            else:

                st.warning(
                    "No valid coordinates were found for "
                    "the selected charging zones."
                )

        else:

            st.warning(
                "Latitude and longitude columns are missing."
            )

else:

    st.warning(
        "Zone information is unavailable. Check that "
        "zone-information.csv is in the same folder as app.py."
    )


st.divider()


# =========================================================
# 10. MODEL EXPLAINABILITY
# =========================================================

st.header("Model Explainability (SHAP)")


if shap_data is not None and not shap_data.empty:

    shap_view = shap_data.copy()

    shap_view.columns = (
        shap_view.columns.astype(str).str.strip()
    )

    st.subheader("Feature Importance Results")

    st.dataframe(
        shap_view,
        use_container_width=True,
        hide_index=True
    )


    # Identify feature and importance columns.

    feature_col = next(
        (
            col
            for col in shap_view.columns
            if "feature" in col.lower()
        ),
        None
    )

    numeric_shap_columns = (
        shap_view.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    importance_col = next(
        (
            col
            for col in numeric_shap_columns
            if (
                "importance" in col.lower()
                or "shap" in col.lower()
            )
        ),
        None
    )


    if (
        feature_col is not None
        and importance_col is not None
    ):

        shap_view[importance_col] = pd.to_numeric(
            shap_view[importance_col],
            errors="coerce"
        )

        shap_plot = shap_view[
            [feature_col, importance_col]
        ].dropna().copy()

        shap_plot = shap_plot.sort_values(
            importance_col,
            ascending=False
        ).head(15)

        if not shap_plot.empty:

            st.subheader("Top 15 Important Features")

            st.bar_chart(
                shap_plot.set_index(feature_col)
            )

        else:

            st.info(
                "No valid feature-importance values "
                "are available to plot."
            )

    else:

        st.info(
            "The SHAP file does not contain clearly identifiable "
            "feature and numerical importance columns."
        )

else:

    st.info(
        "SHAP feature-importance results are unavailable."
    )


st.divider()


# =========================================================
# 11. EXPORT DATA
# =========================================================

st.header("Export Results")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Decision-Support Data",
    data=csv_data,
    file_name="urbanev_decision_support_filtered.csv",
    mime="text/csv"
)


# =========================================================
# 12. FOOTER
# =========================================================

st.caption(
    "UrbanEV Decision Support | Predictive Congestion Modeling "
    "and Dynamic Pricing Optimization"
)

st.info(
    "Dynamic pricing results are simulation-based recommendations. "
    "They should be validated against operational constraints "
    "and real-world driver responses before deployment."
)
    
  
