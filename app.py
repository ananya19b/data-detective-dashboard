from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from detective import investigate


st.set_page_config(page_title="Data Detective", page_icon="🔎", layout="wide")
st.title("🔎 Data Detective")
st.caption("Upload a dataset. Get investigation leads—not unsupported accusations.")


def load_data(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded)
    if suffix == ".xlsx":
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or XLSX file.")


def deterministic_summary(result) -> str:
    p = result.profile
    flagged = int(result.ranked_rows["is_flagged"].sum())
    top = result.ranked_rows[result.ranked_rows["is_flagged"]].head(3)
    lines = [
        f"Analyzed {p['rows']:,} rows and {p['columns']} columns; {flagged:,} rows produced at least one investigation lead.",
        f"Data quality: {p['missing_cells']:,} missing cells and {p['duplicate_rows']:,} duplicate rows.",
    ]
    for _, row in top.iterrows():
        lines.append(f"- Row {int(row['_row_id'])}: {row['reasons']}")
    lines.append("These are statistical signals, not confirmed errors or fraud. Validate them against source systems and business context.")
    return "\n".join(lines)



uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    with st.sidebar:
        st.header("Detection settings")
        iqr_multiplier = st.slider(
            "IQR sensitivity",
            min_value=1.5,
            max_value=4.0,
            value=2.5,
            step=0.5,
            help="Lower values flag more univariate outliers; higher values are more conservative.",
        )
    try:
        data = load_data(uploaded)
        result = investigate(
            data,
            iqr_multiplier=iqr_multiplier,
        )
    except Exception as exc:
        st.error(f"Could not analyze this file: {exc}")
        st.stop()

    p = result.profile
    flagged = result.ranked_rows[result.ranked_rows["is_flagged"]]
    a, b, c, d, e = st.columns(5)
    a.metric("Rows", f"{p['rows']:,}")
    b.metric("Columns", p["columns"])
    c.metric("Flagged rows", f"{len(flagged):,}")
    d.metric("Missing cells", f"{p['missing_cells']:,}")
    e.metric("Data health", f"{p['data_health_score']:.1f}/100")

    st.subheader("Statistical findings")
    st.markdown(deterministic_summary(result))

    numeric_cols = [c for c in data.select_dtypes(include="number").columns]
    overview_tab, quality_tab, explore_tab, relationship_tab, anomaly_tab = st.tabs(
        ["Overview", "Missing values", "Explore", "Relationships", "Anomalies"]
    )

    with overview_tab:
        st.subheader("Column profile")
        st.dataframe(result.column_profile, use_container_width=True, hide_index=True)
        st.caption("Min, max, mean, and standard deviation are shown for numeric columns.")
        if numeric_cols:
            summary = data[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
            st.subheader("Percentiles")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    with quality_tab:
        missing = result.column_profile[result.column_profile["missing"] > 0].copy()
        if missing.empty:
            st.success("No missing values detected.")
        else:
            fig = px.bar(
                missing.sort_values("missing_pct"),
                x="missing_pct",
                y="column",
                orientation="h",
                text="missing",
                title="Missing values by column",
                labels={"missing_pct": "Missing (%)"},
                color="missing_pct",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
            sample = data.isna().astype(int).T
            fig = go.Figure(
                data=go.Heatmap(
                    z=sample.values,
                    x=list(range(1, len(data) + 1)),
                    y=sample.index,
                    colorscale=[[0, "#eef2f7"], [1, "#e45756"]],
                    showscale=False,
                )
            )
            fig.update_layout(title="Missingness map (red = missing)", xaxis_title="Row ID", height=max(300, 28 * len(data.columns)))
            st.plotly_chart(fig, use_container_width=True)
        if p["duplicate_rows"]:
            st.warning(f"{p['duplicate_rows']:,} duplicate row(s) detected.")

    with explore_tab:
        if numeric_cols:
            selected = st.selectbox("Select a numeric column", numeric_cols)
            left, right = st.columns(2)
            with left:
                fig = px.histogram(
                    result.ranked_rows,
                    x=selected,
                    color="is_flagged",
                    marginal="box",
                    barmode="overlay",
                    color_discrete_map={True: "#e45756", False: "#4c78a8"},
                    title=f"Distribution of {selected}",
                )
                st.plotly_chart(fig, use_container_width=True)
            with right:
                chart_data = result.ranked_rows[["_row_id", selected, "is_flagged"]]
                fig = px.scatter(
                    chart_data,
                    x="_row_id",
                    y=selected,
                    color="is_flagged",
                    hover_data=["_row_id"],
                    color_discrete_map={True: "#e45756", False: "#4c78a8"},
                    title=f"{selected} across rows",
                )
                st.plotly_chart(fig, use_container_width=True)

        date_candidates = [
            c for c in data.columns
            if "date" in c.lower() or "time" in c.lower()
        ]
        if date_candidates and numeric_cols:
            date_col = st.selectbox("Date column", date_candidates)
            measure = st.selectbox("Time-series measure", numeric_cols, key="time_measure")
            time_data = result.ranked_rows[[date_col, measure, "is_flagged", "_row_id"]].copy()
            time_data[date_col] = pd.to_datetime(time_data[date_col], errors="coerce")
            time_data = time_data.dropna(subset=[date_col]).sort_values(date_col)
            fig = px.line(time_data, x=date_col, y=measure, title=f"{measure} over time")
            points = time_data[time_data["is_flagged"]]
            fig.add_scatter(x=points[date_col], y=points[measure], mode="markers", marker={"color": "#e45756", "size": 9}, name="Flagged")
            st.plotly_chart(fig, use_container_width=True)

    with relationship_tab:
        if len(numeric_cols) >= 2:
            corr = data[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Strongest numeric relationships")
            st.dataframe(result.correlations.head(10).drop(columns="strength"), use_container_width=True, hide_index=True)
            x_col = st.selectbox("X axis", numeric_cols)
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y axis", y_options)
            color_options = ["None"] + [c for c in data.columns if data[c].nunique(dropna=True) <= 15]
            color_col = st.selectbox("Colour by", color_options)
            scatter_data = result.ranked_rows
            fig = px.scatter(
                scatter_data,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col,
                symbol="is_flagged",
                hover_data=["_row_id", "reasons"],
                title=f"{y_col} vs {x_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are required.")

    with anomaly_tab:
        st.subheader("Ranked investigation leads")
        st.dataframe(flagged.head(50), use_container_width=True, hide_index=True)
        st.download_button(
            "Download flagged rows",
            flagged.to_csv(index=False).encode("utf-8"),
            file_name="data_detective_flagged_rows.csv",
            mime="text/csv",
        )
        left, right = st.columns(2)
        with left:
            if not flagged.empty:
                fig = px.bar(
                    flagged.head(15).sort_values("anomaly_score"),
                    x="anomaly_score",
                    y=flagged.head(15).sort_values("anomaly_score")["_row_id"].astype(str),
                    orientation="h",
                    title="Top rule-trigger counts",
                    labels={"y": "Row ID"},
                    color="anomaly_score",
                    color_continuous_scale="OrRd",
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            cause_counts = (
                flagged["reasons"].str.split("; ").explode().str.extract(r"^([^=]+)")[0]
                .str.replace(" outside IQR bounds.*", "", regex=True)
                .value_counts().head(10).rename_axis("signal").reset_index(name="count")
            )
            if not cause_counts.empty:
                fig = px.bar(cause_counts, x="count", y="signal", orientation="h", title="Most common anomaly signals")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detection thresholds and rare categories"):
            st.write("IQR checks")
            st.dataframe(result.numeric_summary, use_container_width=True, hide_index=True)
            st.write("Rare categories")
            st.dataframe(result.rare_categories, use_container_width=True, hide_index=True)
else:
    st.info("Start with one of the two files in `sample_data/`.")
