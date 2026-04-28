"""
DemandCast — Operations Dashboard
=================================
Streamlit app for the NYC taxi demand forecasting project.

Audience: a head of operations at a taxi company. Every number on screen
is translated into something they can act on (drivers per zone, dispatch
waves, % above/below normal).

Run from the project root with the .venv active:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, time

import altair as alt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_URI = "models:/DemandCast/Production"

# Must match train.py exactly — same names, same order
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]

# Validation-set metrics from notebooks/04_evaluation.md (tuned RandomForest).
# Used to draw the confidence band and populate the Model Trust tab without
# re-running validation on every page load.
TUNED_METRICS = {
    "mae": 8.73,
    "rmse": 18.14,
    "r2": 0.948,
    "mbe": 0.0671,
    "mape_pct": 58.53,
}

# Roughly the size of one dispatch wave in the ops vocabulary used in the
# evaluation report. Used to translate the prediction into "waves of drivers."
DRIVERS_PER_WAVE = 9

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fmt_hour(h: int) -> str:
    """Render a 0–23 hour as a stakeholder-friendly time string."""
    if h == 0:
        return "12 AM"
    if h < 12:
        return f"{h} AM"
    if h == 12:
        return "12 PM"
    return f"{h - 12} PM"


# Vega expression that mirrors fmt_hour() for axis tick labels in Altair.
HOUR_LABEL_EXPR = (
    "datum.value == 0 ? '12 AM' : "
    "datum.value < 12 ? datum.value + ' AM' : "
    "datum.value == 12 ? '12 PM' : "
    "(datum.value - 12) + ' PM'"
)

st.set_page_config(
    page_title="DemandCast — NYC Taxi Demand",
    layout="wide",
    page_icon="🚖",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Production model from MLflow…")
def load_model():
    """Pull the Production RandomForest from the MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


@st.cache_data(show_spinner="Loading features.parquet…")
def load_features() -> pd.DataFrame:
    path = Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    df = pd.read_parquet(path)
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    return df


@st.cache_data
def zone_hour_average(df: pd.DataFrame) -> pd.DataFrame:
    """Average demand for every (zone, hour-of-day) — used for the
    'X% above/below normal' comparison and for lag fall-back."""
    return (
        df.groupby(["PULocationID", "hour"])["demand"]
        .mean()
        .rename("avg_demand")
        .reset_index()
    )


@st.cache_data
def validation_predictions(df: pd.DataFrame, _model) -> pd.DataFrame:
    """Score the validation window (Jan 22–28) so we can show a real
    predicted-vs-actual scatter on the Model Trust tab."""
    val_cutoff = pd.Timestamp("2025-01-22")
    test_cutoff = pd.Timestamp("2025-01-29")
    val = df[(df["pickup_hour"] >= val_cutoff) & (df["pickup_hour"] < test_cutoff)].copy()
    val = val.dropna(subset=FEATURE_COLS)
    val["pred"] = _model.predict(val[FEATURE_COLS])
    val["error"] = val["pred"] - val["demand"]
    return val[["pickup_hour", "PULocationID", "demand", "pred", "error"]]


# ---------------------------------------------------------------------------
# Lag look-up helpers
# ---------------------------------------------------------------------------
def lookup_lag(
    df: pd.DataFrame,
    avg_by_zone_hour: pd.DataFrame,
    zone: int,
    when: pd.Timestamp,
    hours_back: int,
) -> tuple[float, bool]:
    """Return (lag_value, is_real). If the historical row exists, use it.
    Otherwise fall back to that zone's average demand at that hour-of-day."""
    target = when - pd.Timedelta(hours=hours_back)
    hit = df[(df["PULocationID"] == zone) & (df["pickup_hour"] == target)]
    if not hit.empty:
        return float(hit["demand"].iloc[0]), True
    fallback = avg_by_zone_hour[
        (avg_by_zone_hour["PULocationID"] == zone)
        & (avg_by_zone_hour["hour"] == target.hour)
    ]
    if not fallback.empty:
        return float(fallback["avg_demand"].iloc[0]), False
    return 0.0, False


def build_feature_row(
    df: pd.DataFrame,
    avg_by_zone_hour: pd.DataFrame,
    zone: int,
    when: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Assemble a single-row DataFrame matching FEATURE_COLS exactly.
    Also return a flag dict telling the UI which lags were real vs fallback."""
    lag_1h, real_1 = lookup_lag(df, avg_by_zone_hour, zone, when, 1)
    lag_24h, real_24 = lookup_lag(df, avg_by_zone_hour, zone, when, 24)
    lag_168h, real_168 = lookup_lag(df, avg_by_zone_hour, zone, when, 168)

    dow = when.dayofweek
    row = {
        "hour": when.hour,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "month": when.month,
        "is_rush_hour": int(when.hour in (7, 8, 17, 18) and dow < 5),
        "demand_lag_1h": lag_1h,
        "demand_lag_24h": lag_24h,
        "demand_lag_168h": lag_168h,
    }
    flags = {"1h": real_1, "24h": real_24, "168h": real_168}
    return pd.DataFrame([row])[FEATURE_COLS], flags


# ---------------------------------------------------------------------------
# Load resources up front
# ---------------------------------------------------------------------------
try:
    model = load_model()
except Exception as exc:  # noqa: BLE001 — surface the real error to the user
    st.error(f"Could not load the Production model from MLflow: {exc}")
    st.info(
        "Start the MLflow server (`mlflow ui`) at http://localhost:5000 and "
        "make sure a model named **DemandCast** is registered in the "
        "**Production** stage."
    )
    st.stop()

df = load_features()
avg_by_zone_hour = zone_hour_average(df)
zones = sorted(df["PULocationID"].unique().tolist())

data_min = df["pickup_hour"].min()
data_max = df["pickup_hour"].max()
# features.parquet already drops rows with NaN lag_168h during the build,
# so every remaining row has a valid 168h history — no extra buffer needed.
earliest_date = data_min.date()
latest_date = data_max.date()


# ---------------------------------------------------------------------------
# Sidebar — operator inputs
# ---------------------------------------------------------------------------
st.sidebar.title("🚖 DemandCast")
st.sidebar.caption("NYC Yellow Taxi — hourly demand forecast")
st.sidebar.markdown("---")
st.sidebar.subheader("Forecast inputs")

# Default to a zone with a healthy amount of activity so the demo looks alive.
default_zone_idx = zones.index(132) if 132 in zones else 0
zone = st.sidebar.selectbox(
    "Pickup zone (TLC ID)",
    options=zones,
    index=default_zone_idx,
    help="NYC TLC pickup-zone ID (1–263). Try zone 132 for JFK or 138 for LaGuardia.",
)

selected_date = st.sidebar.date_input(
    "Date",
    value=date(2025, 1, 24),
    min_value=earliest_date,
    max_value=latest_date,
    help=f"Constrained to the dataset window ({earliest_date} – {latest_date}).",
)

hour = st.sidebar.selectbox(
    "Time of day",
    options=list(range(24)),
    index=17,
    format_func=fmt_hour,
)

selected_dt = pd.Timestamp(datetime.combine(selected_date, time(hour=hour)))

st.sidebar.markdown("---")
st.sidebar.markdown("**Auto-computed features**")
dow = selected_dt.dayofweek
is_weekend = dow >= 5
is_rush_hour = (hour in (7, 8, 17, 18)) and not is_weekend
st.sidebar.write(
    f"- Day: **{DAY_NAMES[dow]}**  ({'weekend' if is_weekend else 'weekday'})\n"
    f"- Month: **{selected_dt.month}**\n"
    f"- Rush hour: **{'Yes' if is_rush_hour else 'No'}**"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚖 DemandCast — Operations Dashboard")
st.caption(
    "Forecast hourly Yellow Taxi demand for any NYC pickup zone, then translate "
    "the number into a dispatch decision."
)

tab_forecast, tab_patterns, tab_trust = st.tabs(
    ["🎯 Live Forecast", "📊 Demand Patterns", "✅ Model Trust"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Live Forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    feature_row, lag_flags = build_feature_row(df, avg_by_zone_hour, zone, selected_dt)
    prediction = float(model.predict(feature_row)[0])
    mae = TUNED_METRICS["mae"]
    pred_low = max(0.0, prediction - mae)
    pred_high = prediction + mae

    # Historical baseline for this zone × hour-of-day
    hist_row = avg_by_zone_hour[
        (avg_by_zone_hour["PULocationID"] == zone) & (avg_by_zone_hour["hour"] == hour)
    ]
    hist_avg = float(hist_row["avg_demand"].iloc[0]) if not hist_row.empty else float("nan")
    if not np.isnan(hist_avg) and hist_avg > 0:
        delta_pct = (prediction - hist_avg) / hist_avg * 100
        delta_label = f"{delta_pct:+.0f}% vs typical {fmt_hour(hour)} in this zone"
    else:
        delta_pct = float("nan")
        delta_label = None

    left, right = st.columns([1.1, 1])

    with left:
        st.metric(
            label=f"Predicted demand — Zone {zone} on "
            f"{DAY_NAMES[dow]} {selected_date.isoformat()} at {fmt_hour(hour)}",
            value=f"{int(round(prediction))} rides",
            delta=delta_label,
            delta_color="off",
        )
        st.caption(
            f"Likely range: **{int(round(pred_low))} – {int(round(pred_high))} "
            f"rides** (± validation MAE of {mae:.1f})"
        )

        drivers = max(0, int(round(prediction)))
        waves = max(1, int(round(drivers / DRIVERS_PER_WAVE)))
        wave_word = "wave" if waves == 1 else "waves"
        st.success(
            f"**Dispatch recommendation:** position roughly **{drivers} drivers** "
            f"in zone {zone} for the **{fmt_hour(hour)}** hour — about **{waves} "
            f"dispatch {wave_word}** ({DRIVERS_PER_WAVE} drivers per wave)."
        )

        if not np.isnan(delta_pct):
            if abs(delta_pct) < 10:
                tone = st.info
                verdict = "in line with the typical pattern for this zone & hour."
            elif delta_pct >= 10:
                tone = st.warning
                verdict = (
                    f"running **{delta_pct:.0f}% above** the typical pattern — "
                    f"consider pulling drivers from quieter neighbouring zones."
                )
            else:
                tone = st.warning
                verdict = (
                    f"running **{abs(delta_pct):.0f}% below** the typical pattern — "
                    f"redirect drivers to busier zones rather than letting them idle."
                )
            tone(f"This forecast is {verdict}")

    with right:
        st.markdown("**Inputs the model is using right now**")
        display_row = feature_row.iloc[0].to_dict()
        display_row["demand_lag_1h"] = (
            f"{display_row['demand_lag_1h']:.0f}"
            f" {'✓ historical' if lag_flags['1h'] else '~ avg fallback'}"
        )
        display_row["demand_lag_24h"] = (
            f"{display_row['demand_lag_24h']:.0f}"
            f" {'✓ historical' if lag_flags['24h'] else '~ avg fallback'}"
        )
        display_row["demand_lag_168h"] = (
            f"{display_row['demand_lag_168h']:.0f}"
            f" {'✓ historical' if lag_flags['168h'] else '~ avg fallback'}"
        )
        st.dataframe(
            pd.DataFrame(
                {"feature": list(display_row.keys()), "value": list(display_row.values())}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Lag features are looked up automatically from the actual history — "
            "an operator should never have to type these by hand."
        )

    st.divider()
    st.subheader(f"24-hour forecast — Zone {zone}, {selected_date.isoformat()}")

    # Predict every hour of the selected date for the selected zone.
    forecast_rows = []
    for h in range(24):
        h_dt = pd.Timestamp(datetime.combine(selected_date, time(hour=h)))
        f_row, _ = build_feature_row(df, avg_by_zone_hour, zone, h_dt)
        h_pred = float(model.predict(f_row)[0])
        actual_row = df[(df["PULocationID"] == zone) & (df["pickup_hour"] == h_dt)]
        actual = float(actual_row["demand"].iloc[0]) if not actual_row.empty else None
        forecast_rows.append(
            {
                "hour": h,
                "predicted": h_pred,
                "actual": actual,
                "low": max(0.0, h_pred - mae),
                "high": h_pred + mae,
            }
        )
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["time_label"] = forecast_df["hour"].map(fmt_hour)

    hour_axis = alt.X(
        "hour:O",
        title="Time of day",
        axis=alt.Axis(labelExpr=HOUR_LABEL_EXPR, labelAngle=0),
    )

    band = (
        alt.Chart(forecast_df)
        .mark_area(opacity=0.2, color="#1f77b4")
        .encode(
            x=hour_axis,
            y=alt.Y("low:Q", title="Rides per hour"),
            y2="high:Q",
            tooltip=[
                alt.Tooltip("time_label:N", title="Time"),
                alt.Tooltip("low:Q", title="Low", format=".1f"),
                alt.Tooltip("high:Q", title="High", format=".1f"),
            ],
        )
    )
    pred_line = (
        alt.Chart(forecast_df)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=hour_axis,
            y="predicted:Q",
            tooltip=[
                alt.Tooltip("time_label:N", title="Time"),
                alt.Tooltip("predicted:Q", title="Predicted", format=".1f"),
                alt.Tooltip("actual:Q", title="Actual", format=".1f"),
            ],
        )
    )
    actual_line = (
        alt.Chart(forecast_df.dropna(subset=["actual"]))
        .mark_line(point=True, color="#d62728", strokeDash=[4, 4])
        .encode(x=hour_axis, y="actual:Q")
    )
    selected_marker = (
        alt.Chart(pd.DataFrame({"hour": [hour]}))
        .mark_rule(color="#888", strokeDash=[2, 2])
        .encode(x=hour_axis)
    )
    st.altair_chart(
        (band + pred_line + actual_line + selected_marker).properties(height=320),
        width="stretch",
    )
    st.caption(
        "**Blue** = model prediction with ± MAE band. **Red dashed** = actual demand "
        "from the historical data (when available). **Grey line** marks the hour you "
        "selected in the sidebar."
    )

# ---------------------------------------------------------------------------
# Tab 2 — Demand Patterns (across all zones)
# ---------------------------------------------------------------------------
with tab_patterns:
    st.subheader("When does NYC need taxis?")
    st.caption(
        "Patterns aggregated across all 136 zones in the January 2025 dataset. "
        "Use these as the operational backdrop for any single-zone forecast."
    )

    heatmap_data = (
        df.groupby(["day_of_week", "hour"])["demand"].mean().reset_index()
    )
    heatmap_data["day"] = heatmap_data["day_of_week"].map(lambda i: DAY_NAMES[i])
    heatmap_data["time_label"] = heatmap_data["hour"].map(fmt_hour)
    heatmap = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X(
                "hour:O",
                title="Time of day",
                axis=alt.Axis(labelExpr=HOUR_LABEL_EXPR, labelAngle=0),
            ),
            y=alt.Y("day:O", sort=DAY_NAMES, title="Day of week"),
            color=alt.Color(
                "demand:Q",
                title="Avg rides / hour",
                scale=alt.Scale(scheme="oranges"),
            ),
            tooltip=[
                alt.Tooltip("day:O", title="Day"),
                alt.Tooltip("time_label:N", title="Time"),
                alt.Tooltip("demand:Q", title="Avg rides", format=".1f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(heatmap, width="stretch")
    st.caption(
        "Darker cells = higher average demand. The bright vertical bands at "
        "8 AM and 5 PM on weekdays are the commute spikes; the bright corner "
        "around midnight on Fri/Sat is nightlife."
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 15 busiest pickup zones**")
        top_zones = (
            df.groupby("PULocationID")["demand"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        top_zones["zone"] = top_zones["PULocationID"].astype(str)
        bar = (
            alt.Chart(top_zones)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("demand:Q", title="Avg rides per hour"),
                y=alt.Y("zone:N", sort="-x", title="Zone ID"),
                tooltip=[
                    alt.Tooltip("zone:N", title="Zone"),
                    alt.Tooltip("demand:Q", title="Avg rides", format=".1f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(bar, width="stretch")

    with col_b:
        st.markdown("**Weekday vs weekend — hourly pattern**")
        wk = (
            df.assign(kind=lambda d: np.where(d["is_weekend"] == 1, "Weekend", "Weekday"))
            .groupby(["kind", "hour"])["demand"]
            .mean()
            .reset_index()
        )
        wk["time_label"] = wk["hour"].map(fmt_hour)
        wk_chart = (
            alt.Chart(wk)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "hour:O",
                    title="Time of day",
                    axis=alt.Axis(labelExpr=HOUR_LABEL_EXPR, labelAngle=0),
                ),
                y=alt.Y("demand:Q", title="Avg rides per hour"),
                color=alt.Color("kind:N", title=None),
                tooltip=[
                    alt.Tooltip("kind:N", title="Type"),
                    alt.Tooltip("time_label:N", title="Time"),
                    alt.Tooltip("demand:Q", title="Avg rides", format=".1f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(wk_chart, width="stretch")
        st.caption(
            "Weekdays show the classic twin-peak commute. Weekends shift the "
            "mass later — afternoons and late nights dominate."
        )

# ---------------------------------------------------------------------------
# Tab 3 — Model Trust
# ---------------------------------------------------------------------------
with tab_trust:
    st.subheader("How wrong does this forecast get?")
    st.caption(
        "All four metrics are computed on the validation set (Jan 22–28, 2025) — "
        "the week the model never saw during training."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{TUNED_METRICS['mae']:.2f} rides")
    m2.metric("RMSE", f"{TUNED_METRICS['rmse']:.2f} rides")
    m3.metric("R²", f"{TUNED_METRICS['r2']:.3f}")
    m4.metric("Bias (MBE)", f"{TUNED_METRICS['mbe']:+.3f} rides")

    with st.expander("What do these numbers mean for operations?", expanded=True):
        st.markdown(
            f"""
- **MAE = {TUNED_METRICS['mae']:.2f} rides.** On an average hour in an average
  zone, the forecast is off by about **9 rides** — roughly **one dispatch wave**.
  Plan for that as your baseline scheduling tolerance.
- **RMSE = {TUNED_METRICS['rmse']:.2f} rides.** RMSE punishes big misses harder
  than MAE. The gap between RMSE and MAE means the model is usually tight, but
  occasionally gets blindsided by a spike (think rainstorm or transit outage).
- **R² = {TUNED_METRICS['r2']:.3f}.** The model explains about **95%** of the
  natural fluctuation in demand. The remaining 5% is random noise and external
  factors we don't track yet (weather, events).
- **MBE = {TUNED_METRICS['mbe']:+.3f} rides.** The forecast leans *very*
  slightly high on average — a safe direction for ops, since one extra idle
  driver costs less than one stranded passenger.
- **MAPE = {TUNED_METRICS['mape_pct']:.1f}%.** Inflated by zero-demand zone-hours
  (we add a tiny epsilon to avoid divide-by-zero); read MAE first.
"""
        )

    st.markdown("---")
    col_imp, col_scatter = st.columns(2)

    with col_imp:
        st.markdown("**What is the model actually paying attention to?**")
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": FEATURE_COLS, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            imp_chart = (
                alt.Chart(imp)
                .mark_bar(color="#2ca02c")
                .encode(
                    x=alt.X("importance:Q", title="Relative importance"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("feature:N"),
                        alt.Tooltip("importance:Q", format=".3f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(imp_chart, width="stretch")
            st.caption(
                "Recent demand (1h ago) and same-hour-last-week (168h ago) "
                "carry most of the signal — confirming that taxi demand is "
                "dominated by short-term momentum and weekly habit."
            )
        else:
            st.info("This model exposes no `feature_importances_` attribute.")

    with col_scatter:
        st.markdown("**Predicted vs actual on the validation week**")
        val_pred = validation_predictions(df, model)
        sample = val_pred.sample(min(2000, len(val_pred)), random_state=0)
        scatter = (
            alt.Chart(sample)
            .mark_circle(size=18, opacity=0.35, color="#1f77b4")
            .encode(
                x=alt.X("demand:Q", title="Actual rides per hour"),
                y=alt.Y("pred:Q", title="Predicted rides per hour"),
                tooltip=[
                    alt.Tooltip("demand:Q", title="Actual"),
                    alt.Tooltip("pred:Q", title="Predicted", format=".1f"),
                    alt.Tooltip("PULocationID:N", title="Zone"),
                    alt.Tooltip("pickup_hour:T", title="When"),
                ],
            )
        )
        max_val = float(max(sample["demand"].max(), sample["pred"].max()))
        diag = (
            alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]}))
            .mark_line(color="#888", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart((scatter + diag).properties(height=320), width="stretch")
        st.caption(
            "Each dot is one zone-hour from the validation week. Points on "
            "the dashed line are perfect predictions — the tight cluster "
            "shows the model tracks reality well, with the occasional miss "
            "on the high-demand tail."
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Model: tuned RandomForestRegressor (Optuna, 15 trials) — "
    "**DemandCast / Production** in the local MLflow registry. "
    "Data: NYC TLC Yellow Taxi trip records, January 2025."
)
