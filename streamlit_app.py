from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_INPUT_PATH = "CleanData/*_tweets_cleaned.csv"
DEFAULT_JAVA_HOME = Path("C:/Program Files/Eclipse Adoptium/jdk-17.0.19.10-hotspot")

STATE_CODES = {
    "Arizona": "AZ",
    "Georgia": "GA",
    "Michigan": "MI",
    "Nevada": "NV",
    "North Carolina": "NC",
    "Pennsylvania": "PA",
    "Wisconsin": "WI",
}

QUERY_FILES = {
    "geography": "query_01_geographic_distribution/geographic_distribution.csv",
    "temporal": "query_02_temporal_evolution/temporal_evolution.csv",
    "hashtags": "query_03_top_hashtags/top_hashtags.csv",
    "words": "query_04_frequent_words/frequent_words.csv",
    "distinctive_hashtags": "query_05_distinctive_hashtags/distinctive_hashtags.csv",
    "keywords": "query_06_trump_harris_keyword_comparison/trump_harris_keyword_comparison.csv",
    "cooccurrence": "query_07_word_cooccurrence/word_cooccurrence.csv",
    "polarization": "query_08_territorial_polarization/territorial_polarization.csv",
    "clusters": "query_09_kmeans_clustering/cluster_summary.csv",
    "cluster_terms": "query_09_kmeans_clustering/top_terms.csv",
    "svm_metrics": "query_10_svm_classification/metrics.csv",
    "svm_confusion": "query_10_svm_classification/confusion_matrix.csv",
}

QUERY_OPTIONS = {
    1: "Query 1 - Distribuzione geografica",
    2: "Query 2 - Evoluzione temporale",
    3: "Query 3 - Top hashtag",
    4: "Query 4 - Parole frequenti",
    5: "Query 5 - Hashtag distintivi",
    6: "Query 6 - Keyword Trump/Harris",
    7: "Query 7 - Co-occorrenza parole",
    8: "Query 8 - Polarizzazione territoriale",
    9: "Query 9 - K-Means clustering",
    10: "Query 10 - SVM classificazione",
}

SVM_FEATURE_OPTIONS = {
    "text": "Testo tweet",
    "hashtags": "Hashtag",
    "description": "Descrizione utente",
}

NOISY_SPARK_LOG_PARTS = (
    " WARN ",
    " WARNING ",
    "Using Spark's default log4j profile",
    "Setting default log level to",
    "NativeCodeLoader",
    "Unable to load native-hadoop library",
    "SparkUI",
    "BlockManager",
    "MetricsSystem",
    "DAGScheduler",
    "TaskSchedulerImpl",
    "PythonRunner",
    "InstanceBuilder",
    "GarbageCollectionMetrics",
    "java.lang.ProcessHandle",
    "[Stage ",
)

COLOR_SEQUENCE = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]
ORIENTATION_COLORS = {
    "PRO_TRUMP": "#dc2626",
    "PRO_HARRIS": "#2563eb",
    "BALANCED": "#16a34a",
}
SVM_METRIC_COLORS = {
    "area_under_roc": "#93c5fd",
    "accuracy": "#60a5fa",
    "f1": "#38bdf8",
    "weighted_precision": "#22d3ee",
    "weighted_recall": "#2dd4bf",
}


def title_with_query(query_number: int, title: str) -> str:
    return f"{title} ({QUERY_OPTIONS[query_number]})"


st.set_page_config(
    page_title="Swing States Tweet Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --panel-border: #d6dbe5;
        --soft-bg: #f7f8fb;
        --ink: #172033;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: rgba(15, 23, 42, 0.42);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: inherit;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.7rem;
        border-radius: 6px 6px 0 0;
        padding: 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def csv_path(output_dir: Path, key: str) -> Path:
    return output_dir / QUERY_FILES[key]


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_data(output_dir: Path) -> dict[str, pd.DataFrame]:
    data = {key: read_csv(str(csv_path(output_dir, key))) for key in QUERY_FILES}
    if not data["temporal"].empty:
        data["temporal"]["date"] = pd.to_datetime(data["temporal"]["date"], errors="coerce")
    for key, frame in data.items():
        if "state" in frame.columns:
            frame["state_code"] = frame["state"].map(STATE_CODES)
    return data


def available_outputs(data: dict[str, pd.DataFrame]) -> list[str]:
    return [key for key, frame in data.items() if not frame.empty]


def is_noisy_spark_log(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    upper_line = stripped.upper()
    if upper_line.startswith(("WARN", "WARNING")):
        return True
    return any(part in stripped for part in NOISY_SPARK_LOG_PARTS)


def run_spark_analysis(
    input_path: str,
    output_dir: Path,
    queries: list[int],
    svm_features: list[str],
    log_placeholder,
) -> int:
    command = [
        sys.executable,
        str(BASE_DIR / "spark_analysis.py"),
        "--input",
        input_path,
        "--output",
        str(output_dir),
        "--queries",
        *[str(query) for query in queries],
    ]
    if 10 in queries:
        command.extend(["--svm-features", *svm_features])

    env = os.environ.copy()
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if "JAVA_HOME" not in env and DEFAULT_JAVA_HOME.exists():
        env["JAVA_HOME"] = str(DEFAULT_JAVA_HOME)
        env["PATH"] = f"{DEFAULT_JAVA_HOME / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONWARNINGS"] = "ignore"

    lines = ["> " + " ".join(command)]
    hidden_lines = 0
    log_placeholder.code("\n".join(lines), language="powershell")

    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.rstrip()
        if is_noisy_spark_log(clean_line):
            hidden_lines += 1
            continue
        lines.append(clean_line)
        log_placeholder.code("\n".join(lines[-120:]), language="text")

    return_code = process.wait()
    if hidden_lines:
        lines.append(f"[streamlit] Log tecnico Spark nascosto: {hidden_lines} righe")
    lines.append(f"[streamlit] Spark terminato con codice {return_code}")
    log_placeholder.code("\n".join(lines[-140:]), language="text")
    return return_code


def show_missing(name: str) -> None:
    st.info(f"Output non trovato per {name}. Esegui la query corrispondente in `spark_analysis.py`.")


def format_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def filter_states(df: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    if df.empty or "state" not in df.columns or not states:
        return df
    return df[df["state"].isin(states)].copy()


def state_choropleth(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    color_scale: str = "Blues",
    hover_cols: list[str] | None = None,
    range_color: tuple[float, float] | None = None,
    colorbar_title: str | None = None,
) -> go.Figure:
    state_df = df.copy()
    if "state_code" not in state_df.columns and "state" in state_df.columns:
        state_df["state_code"] = state_df["state"].map(STATE_CODES)
    state_df = state_df.dropna(subset=["state_code"]).copy()
    fig = px.choropleth(
        state_df,
        locations="state_code",
        locationmode="USA-states",
        color=value_col,
        scope="usa",
        hover_name="state",
        hover_data=hover_cols or [],
        color_continuous_scale=color_scale,
        range_color=range_color,
        labels={value_col: colorbar_title or value_col},
    )
    fig.update_geos(
        visible=True,
        scope="usa",
        showland=True,
        landcolor="#eef2f7",
        lakecolor="#ffffff",
        subunitcolor="#cbd5e1",
        countrycolor="#94a3b8",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=46, b=0),
        height=440,
        geo=dict(projection_scale=0.92),
        coloraxis_colorbar=dict(thickness=14, title=colorbar_title or value_col),
    )
    return fig


def horizontal_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    frame = df.sort_values(x, ascending=True)
    fig = px.bar(
        frame,
        x=x,
        y=y,
        orientation="h",
        color=color,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=title,
    )
    fig.update_layout(height=max(360, min(760, 28 * len(frame) + 90)), margin=dict(l=0, r=0, t=50, b=0))
    return fig


def overview_tab(data: dict[str, pd.DataFrame], states: list[str]) -> None:
    geography = filter_states(data["geography"], states)
    temporal = filter_states(data["temporal"], states)
    keywords = filter_states(data["keywords"], states)

    if geography.empty:
        show_missing("la distribuzione geografica")
        return

    total_tweets = geography["tweet_count"].sum()
    total_users = geography["unique_users"].sum()
    active_cities = geography[["state", "city"]].drop_duplicates().shape[0]
    top_state = geography.groupby("state", as_index=False)["tweet_count"].sum().sort_values("tweet_count", ascending=False).head(1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tweet analizzati", format_int(total_tweets))
    col2.metric("Utenti unici", format_int(total_users))
    col3.metric("Citta monitorate", format_int(active_cities))
    col4.metric("Stato piu attivo", top_state.iloc[0]["state"] if not top_state.empty else "-")

    left, right = st.columns([1.15, 0.85])
    state_volume = geography.groupby("state", as_index=False).agg(tweet_count=("tweet_count", "sum"), unique_users=("unique_users", "sum"))
    with left:
        st.plotly_chart(
            state_choropleth(state_volume, "tweet_count", "Volume tweet per stato", "YlOrRd", ["tweet_count", "unique_users"]),
            width="stretch",
        )
    with right:
        city_rank = geography.head(12).copy()
        city_rank["area"] = city_rank["city"] + ", " + city_rank["state"]
        st.plotly_chart(horizontal_bar(city_rank, "tweet_count", "area", "Citta piu attive"), width="stretch")

    if not temporal.empty:
        daily = temporal.groupby("date", as_index=False)["tweet_count"].sum()
        fig = px.area(daily, x="date", y="tweet_count", title="Andamento complessivo dei tweet", color_discrete_sequence=["#2563eb"])
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=330)
        st.plotly_chart(fig, width="stretch")

    if not keywords.empty:
        mention_cols = ["trump_mentions", "harris_mentions"]
        mentions = keywords.groupby("state", as_index=False)[mention_cols].sum()
        fig = px.bar(
            mentions,
            x="state",
            y=mention_cols,
            barmode="group",
            title="Menzioni keyword Trump/Harris per stato",
            color_discrete_map={"trump_mentions": "#dc2626", "harris_mentions": "#2563eb"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=360, legend_title_text="")
        st.plotly_chart(fig, width="stretch")


def geography_tab(data: dict[str, pd.DataFrame], states: list[str]) -> None:
    geography = filter_states(data["geography"], states)
    if geography.empty:
        show_missing("la distribuzione geografica")
        return

    state_volume = geography.groupby("state", as_index=False).agg(tweet_count=("tweet_count", "sum"), unique_users=("unique_users", "sum"))
    metric = st.radio("Metrica geografica", ["tweet_count", "unique_users"], horizontal=True, format_func=lambda x: "Tweet" if x == "tweet_count" else "Utenti unici")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(
            state_choropleth(
                state_volume,
                metric,
                title_with_query(1, "Distribuzione geografica"),
                "Blues",
                ["tweet_count", "unique_users"],
            ),
            width="stretch",
        )
    with right:
        fig = px.treemap(
            geography,
            path=["state", "city"],
            values=metric,
            color="state",
            color_discrete_sequence=COLOR_SEQUENCE,
            title=title_with_query(1, "Peso relativo di stati e citta"),
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=440)
        st.plotly_chart(fig, width="stretch")

    top_n = st.slider("Numero citta nel ranking", 5, 30, 15)
    rank = geography.sort_values(metric, ascending=False).head(top_n).copy()
    rank["area"] = rank["city"] + ", " + rank["state"]
    st.plotly_chart(horizontal_bar(rank, metric, "area", title_with_query(1, "Ranking territoriale"), "state"), width="stretch")
    st.dataframe(geography, width="stretch", hide_index=True)


def temporal_tab(data: dict[str, pd.DataFrame], states: list[str]) -> None:
    temporal = filter_states(data["temporal"], states)
    if temporal.empty:
        show_missing("l'evoluzione temporale")
        return

    daily_state = temporal.groupby(["date", "state"], as_index=False)["tweet_count"].sum()
    fig = px.line(daily_state, x="date", y="tweet_count", color="state", title=title_with_query(2, "Volume giornaliero per stato"), markers=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=430)
    st.plotly_chart(fig, width="stretch")

    sentiment_cols = ["pro_trump", "pro_harris", "neutral_or_unclear"]
    daily_sentiment = temporal.groupby("date", as_index=False)[sentiment_cols].sum()
    fig = px.area(
        daily_sentiment,
        x="date",
        y=sentiment_cols,
        title=title_with_query(2, "Composizione politica giornaliera"),
        color_discrete_map={"pro_trump": "#dc2626", "pro_harris": "#2563eb", "neutral_or_unclear": "#64748b"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=430, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(temporal.sort_values(["date", "state"]), width="stretch", hide_index=True)


def content_tab(data: dict[str, pd.DataFrame], states: list[str]) -> None:
    hashtags = data["hashtags"]
    words = data["words"]
    distinctive_hashtags = data["distinctive_hashtags"]
    cooccurrence = data["cooccurrence"]
    keywords = filter_states(data["keywords"], states)

    left, right = st.columns(2)
    with left:
        if hashtags.empty:
            show_missing("i top hashtag")
        else:
            top_hashtags = hashtags.head(st.slider("Hashtag da mostrare", 5, 40, 20))
            st.plotly_chart(horizontal_bar(top_hashtags, "count", "hashtag", title_with_query(3, "Hashtag piu frequenti")), width="stretch")
    with right:
        if words.empty:
            show_missing("le parole frequenti")
        else:
            top_words = words.head(st.slider("Parole da mostrare", 5, 40, 20))
            st.plotly_chart(horizontal_bar(top_words, "count", "word", title_with_query(4, "Parole piu frequenti")), width="stretch")

    if distinctive_hashtags.empty:
        show_missing("gli hashtag distintivi per orientamento")
    else:
        top_trump = distinctive_hashtags[distinctive_hashtags["orientation_score"] > 0].nlargest(15, "orientation_score")
        top_harris = distinctive_hashtags[distinctive_hashtags["orientation_score"] < 0].nsmallest(15, "orientation_score")
        distinctive_plot = pd.concat([top_harris, top_trump], ignore_index=True)
        fig = px.bar(
            distinctive_plot.sort_values("orientation_score"),
            x="orientation_score",
            y="hashtag",
            orientation="h",
            color="dominant_orientation",
            color_discrete_map={"PRO_TRUMP": "#dc2626", "PRO_HARRIS": "#2563eb", "BALANCED": "#16a34a"},
            title=title_with_query(5, "Hashtag distintivi per orientamento politico"),
            hover_data=["total_count", "pro_trump_count", "pro_harris_count", "pro_trump_share", "pro_harris_share"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=620, legend_title_text="")
        fig.update_xaxes(range=[-1, 1], title_text="Score orientamento (-1 Harris, +1 Trump)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(distinctive_hashtags, width="stretch", hide_index=True)

    if keywords.empty:
        show_missing("il confronto keyword Trump/Harris")
    else:
        mention_cols = ["trump_mentions", "harris_mentions"]
        mentions = keywords.groupby("state", as_index=False)[mention_cols].sum()
        fig = px.bar(
            mentions,
            x="state",
            y=mention_cols,
            barmode="group",
            title=title_with_query(6, "Menzioni keyword Trump/Harris per stato"),
            color_discrete_map={"trump_mentions": "#dc2626", "harris_mentions": "#2563eb"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=380, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(keywords, width="stretch", hide_index=True)

    if cooccurrence.empty:
        show_missing("le co-occorrenze")
        return

    top_pairs = cooccurrence.head(35).copy()
    fig = px.scatter(
        top_pairs,
        x="word_a",
        y="word_b",
        size="count",
        color="count",
        color_continuous_scale="Teal",
        title=title_with_query(7, "Coppie di parole che appaiono insieme"),
        hover_data=["count"],
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cooccurrence, width="stretch", hide_index=True)


def polarization_tab(data: dict[str, pd.DataFrame], states: list[str]) -> None:
    polarization = filter_states(data["polarization"], states)
    if polarization.empty:
        show_missing("la polarizzazione territoriale")
        return

    state_polarization = (
        polarization.groupby("state", as_index=False)
        .agg(
            classified_users=("classified_users", "sum"),
            pro_trump_users=("pro_trump_users", "sum"),
            pro_harris_users=("pro_harris_users", "sum"),
            balanced_users=("balanced_users", "sum"),
        )
    )
    state_polarization["user_polarization_score"] = (
        (state_polarization["pro_trump_users"] - state_polarization["pro_harris_users"]).abs()
        / state_polarization["classified_users"].replace(0, pd.NA)
    ).fillna(0)

    left, right = st.columns([1.05, 0.95])
    with left:
        st.plotly_chart(
            state_choropleth(
                state_polarization,
                "user_polarization_score",
                title_with_query(8, "Polarizzazione utenti per stato"),
                "Reds",
                ["classified_users", "pro_trump_users", "pro_harris_users"],
                colorbar_title="Indice",
            ),
            width="stretch",
        )
    with right:
        stacked = state_polarization.melt(
            id_vars="state",
            value_vars=["pro_trump_users", "pro_harris_users", "balanced_users"],
            var_name="orientamento",
            value_name="utenti",
        )
        fig = px.bar(
            stacked,
            x="state",
            y="utenti",
            color="orientamento",
            title=title_with_query(8, "Composizione utenti classificati"),
            color_discrete_map={"pro_trump_users": "#dc2626", "pro_harris_users": "#2563eb", "balanced_users": "#16a34a"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=440, legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    min_users = st.slider("Soglia minima utenti classificati", 1, int(polarization["classified_users"].max()), 1)
    cities = polarization[polarization["classified_users"] >= min_users].head(25).copy()
    cities["area"] = cities["city"] + ", " + cities["state"]
    fig = px.bar(
        cities.sort_values("user_polarization_score"),
        x="user_polarization_score",
        y="area",
        orientation="h",
        color="dominant_user_orientation",
        color_discrete_map=ORIENTATION_COLORS,
        title=title_with_query(8, "Citta piu polarizzate"),
        hover_data=["classified_users", "pro_trump_users", "pro_harris_users", "balanced_users"],
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=620, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(polarization, width="stretch", hide_index=True)


def models_tab(data: dict[str, pd.DataFrame]) -> None:
    clusters = data["clusters"]
    cluster_terms = data["cluster_terms"]
    svm_metrics = data["svm_metrics"]
    svm_confusion = data["svm_confusion"]

    left, right = st.columns([1, 1])
    with left:
        if clusters.empty:
            show_missing("il clustering K-Means")
        else:
            cluster_display = clusters.copy()
            cluster_display["pro_trump_share"] = cluster_display["pro_trump"] / cluster_display["tweet_count"]
            cluster_display["pro_harris_share"] = cluster_display["pro_harris"] / cluster_display["tweet_count"]
            cluster_display["neutral_or_unclear_share"] = cluster_display["neutral_or_unclear"] / cluster_display["tweet_count"]
            cluster_display = cluster_display.melt(
                id_vars=["cluster", "tweet_count"],
                value_vars=["pro_trump_share", "pro_harris_share", "neutral_or_unclear_share"],
                var_name="orientamento",
                value_name="quota",
            )
            fig = px.bar(
                cluster_display,
                x="cluster",
                y="quota",
                color="orientamento",
                title=title_with_query(9, "Composizione percentuale dei cluster"),
                hover_data={"tweet_count": True, "quota": ":.1%", "orientamento": False},
                color_discrete_map={
                    "pro_trump_share": "#dc2626",
                    "pro_harris_share": "#2563eb",
                    "neutral_or_unclear_share": "#64748b",
                },
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=420, legend_title_text="")
            fig.update_yaxes(tickformat=".0%", title_text="Quota nel cluster")
            st.plotly_chart(fig, width="stretch")

    with right:
        if svm_metrics.empty:
            show_missing("le metriche SVM")
        else:
            metric_display = svm_metrics.copy()
            score_metrics = metric_display[~metric_display["metric"].isin(["train_rows", "test_rows"])]
            fig = px.bar(
                score_metrics,
                x="metric",
                y="value",
                title=title_with_query(10, "Metriche SVM"),
                color="metric",
                color_discrete_map=SVM_METRIC_COLORS,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=420, showlegend=False)
            fig.update_yaxes(range=[0, max(1.0, score_metrics["value"].max())])
            st.plotly_chart(fig, width="stretch")

    if not cluster_terms.empty:
        selected_cluster = st.selectbox("Cluster", sorted(cluster_terms["cluster"].unique()))
        terms = cluster_terms[cluster_terms["cluster"] == selected_cluster].sort_values("rank")
        st.plotly_chart(horizontal_bar(terms, "count", "word", title_with_query(9, f"Termini principali del cluster {selected_cluster}")), width="stretch")

    if not svm_confusion.empty:
        matrix = svm_confusion.pivot_table(index="label", columns="prediction", values="count", fill_value=0)
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", title=title_with_query(10, "Matrice di confusione SVM"))
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=420)
        st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("Swing States Tweet Dashboard")
    st.caption("Dashboard interattiva sui risultati prodotti da `spark_analysis.py`.")

    with st.sidebar:
        st.header("Dati")
        output_dir = DEFAULT_OUTPUT_DIR

        with st.expander("Esegui analisi Spark", expanded=False):
            input_path = st.text_input("Input CSV", value=DEFAULT_INPUT_PATH)
            default_svm_features = list(SVM_FEATURE_OPTIONS)
            svm_features = st.session_state.get("svm_features", default_svm_features)
            query_to_run = None
            for query_number, query_label in QUERY_OPTIONS.items():
                if st.button(
                    query_label,
                    key=f"run_query_{query_number}",
                    disabled=query_number == 10 and not svm_features,
                    width="stretch",
                ):
                    query_to_run = query_number
                if query_number == 10:
                    svm_features = st.multiselect(
                        "Feature SVM",
                        options=default_svm_features,
                        default=svm_features,
                        format_func=lambda feature: SVM_FEATURE_OPTIONS[feature],
                        help="Valgono solo per Query 10. Lo stato non e' selezionabile per evitare bias geografico.",
                        key="svm_features",
                    )
            log_placeholder = st.empty()

            if query_to_run is not None:
                with st.spinner("Spark sta eseguendo le query selezionate..."):
                    return_code = run_spark_analysis(
                        input_path=input_path,
                        output_dir=output_dir,
                        queries=[query_to_run],
                        svm_features=svm_features,
                        log_placeholder=log_placeholder,
                    )
                if return_code == 0:
                    st.cache_data.clear()
                    st.success("Analisi completata. I risultati sotto sono stati ricaricati.")
                else:
                    st.error("Spark ha terminato con errore. Controlla il log qui sopra.")

        data = load_data(output_dir)
        found = available_outputs(data)
        st.write(f"Output caricati: {len(found)} / {len(QUERY_FILES)}")

        all_states = sorted(
            {
                state
                for frame in data.values()
                if "state" in frame.columns
                for state in frame["state"].dropna().unique().tolist()
            }
        )
        selected_states = st.multiselect("Swing states", all_states, default=all_states)

    if not found:
        st.warning("Non ho trovato CSV in `output`. Esegui prima una query Spark dalla sidebar.")
        return

    tabs = st.tabs(["Overview", "Geografia", "Tempo", "Contenuti", "Polarizzazione", "Modelli"])
    with tabs[0]:
        overview_tab(data, selected_states)
    with tabs[1]:
        geography_tab(data, selected_states)
    with tabs[2]:
        temporal_tab(data, selected_states)
    with tabs[3]:
        content_tab(data, selected_states)
    with tabs[4]:
        polarization_tab(data, selected_states)
    with tabs[5]:
        models_tab(data)


if __name__ == "__main__":
    main()
