import argparse
import csv
import glob
import os
import shutil
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Normalizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


DEFAULT_INPUT = "CleanData/*_tweets_cleaned.csv"
DEFAULT_OUTPUT = "output"
_CLEARED_OUTPUT_DIRS: set[str] = set()

TRUMP_KEYWORDS = [
    "trump",
    "donald trump",
    "maga",
    "trump2024",
    "make america great again",
    "vote trump",
    "vote red",
]

HARRIS_KEYWORDS = [
    "harris",
    "kamala",
    "kamala harris",
    "harris2024",
    "vote harris",
    "vote blue",
    "democrats",
]

EXTRA_STOPWORDS = [
    "amp",
    "rt",
    "https",
    "http",
    "co",
    "www",
    "com",
    "tweet",
    "election",
    "elections",
    "2024",
]

KMEANS_STOPWORDS = EXTRA_STOPWORDS + [
    "trump",
    "donald",
    "donaldtrump",
    "maga",
    "harris",
    "kamala",
    "kamalaharris",
    "biden",
    "joe",
    "joebiden",
    "gop",
    "democrat",
    "democrats",
    "republican",
    "republicans",
    "president",
    "election",
    "vote",
    "voting",
    "voted",
    "arizona",
    "georgia",
    "michigan",
    "nevada",
    "carolina",
    "north",
    "pennsylvania",
    "wisconsin",
    "like",
    "people",
    "one",
    "get",
    "know",
    "think",
    "said",
    "even",
    "going",
    "time",
    "need",
    "right",
    "want",
    "make",
    "new",
    "see",
    "say",
    "says",
    "way",
    "also",
    "really",
    "much",
    "probably",
    "thing",
    "things",
    "still",
    "last",
    "ever",
    "well",
    "got",
    "didn",
    "doesn",
    "won",
]


def write_csv(df, output_dir: str, query_dir: str, filename: str) -> None:
    query_output_dir = os.path.join(output_dir, query_dir)
    if query_output_dir not in _CLEARED_OUTPUT_DIRS:
        if os.path.exists(query_output_dir):
            shutil.rmtree(query_output_dir)
        os.makedirs(query_output_dir, exist_ok=True)
        _CLEARED_OUTPUT_DIRS.add(query_output_dir)

    path = os.path.join(query_output_dir, f"{filename}.csv")
    rows = df.collect()

    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(df.columns)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    print(f"[output] {path} ({len(rows)} righe)", flush=True)


def resolve_input_paths(input_path: str) -> str | list[str]:
    if any(char in input_path for char in "*?["):
        matches = sorted(glob.glob(input_path))
        if not matches:
            raise FileNotFoundError(f"Nessun CSV trovato per pattern: {input_path}")
        return matches
    return input_path


def keyword_regex(keywords: list[str]) -> str:
    import re

    return r"(?i)\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"


def load_tweets(spark: SparkSession, input_path: str, multiline_csv: bool = False):
    resolved_input = resolve_input_paths(input_path)
    if isinstance(resolved_input, list):
        print(f"[load] {len(resolved_input)} file CSV", flush=True)
    else:
        print(f"[load] {resolved_input}", flush=True)

    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("multiLine", multiline_csv)
        .option("quote", '"')
        .option("escape", '"')
        .csv(resolved_input)
        .select(
            F.col("user_id").cast("string"),
            F.col("username").cast("string"),
            F.coalesce(F.col("description").cast("string"), F.lit("")).alias("description"),
            F.coalesce(F.col("text").cast("string"), F.lit("")).alias("text"),
            F.coalesce(F.col("hashtags").cast("string"), F.lit("")).alias("hashtags"),
            F.to_date("date").alias("date"),
            F.coalesce(F.col("city").cast("string"), F.lit("Unknown")).alias("city"),
            F.coalesce(F.col("state").cast("string"), F.lit("Unknown")).alias("state"),
            F.col("binary_label").cast("int").alias("binary_label"),
        )
        .withColumn("city", F.when(F.length(F.trim("city")) == 0, "Unknown").otherwise(F.trim("city")))
        .withColumn("state", F.when(F.length(F.trim("state")) == 0, "Unknown").otherwise(F.trim("state")))
        .withColumn(
            "full_text",
            F.concat_ws(" ", F.col("text"), F.col("hashtags"), F.col("description"), F.col("state")),
        )
    )


def run_query(number: int, title: str, query_func) -> None:
    print(f"[query {number}] {title}...", flush=True)
    query_func()
    print(f"[query {number}] completata", flush=True)


def add_tokens(df):
    tokenizer = RegexTokenizer(inputCol="text", outputCol="raw_words", pattern="[^A-Za-z]+", minTokenLength=3)
    remover = StopWordsRemover(
        inputCol="raw_words",
        outputCol="words",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + EXTRA_STOPWORDS,
    )
    return remover.transform(tokenizer.transform(df))


# Query 1 - Distribuzione geografica dei tweet.
# Conta i tweet e gli utenti unici per ogni coppia stato/citta', ordinando le aree
# con maggiore volume di discussione politica.
def geographic_distribution(df, output_dir: str) -> None:
    result = (
        df.groupBy("state", "city")
        .agg(
            F.count("*").alias("tweet_count"),
            F.countDistinct("user_id").alias("unique_users"),
        )
        .orderBy(F.desc("tweet_count"), "state", "city")
    )
    write_csv(result, output_dir, "query_01_geographic_distribution", "geographic_distribution")


# Query 2 - Evoluzione temporale.
# Aggrega i tweet per data e stato, mostrando il volume giornaliero e la
# composizione delle label politiche nel tempo.
def temporal_evolution(df, output_dir: str) -> None:
    result = (
        df.groupBy("date", "state")
        .agg(
            F.count("*").alias("tweet_count"),
            F.sum(F.when(F.col("binary_label") == 1, 1).otherwise(0)).alias("pro_trump"),
            F.sum(F.when(F.col("binary_label") == 0, 1).otherwise(0)).alias("pro_harris"),
            F.sum(F.when(F.col("binary_label").isNull(), 1).otherwise(0)).alias("neutral_or_unclear"),
        )
        .orderBy("date", "state")
    )
    write_csv(result, output_dir, "query_02_temporal_evolution", "temporal_evolution")


# Query 3 - Top hashtag.
# Estrae gli hashtag dalla colonna dedicata, li normalizza e restituisce quelli
# piu' frequenti nella conversazione.
def top_hashtags(df, output_dir: str, top_n: int) -> None:
    cleaned = F.regexp_replace(F.lower(F.col("hashtags")), r"[\[\]'\"{}]", " ")
    tags = (
        df.withColumn("hashtag", F.explode(F.split(cleaned, r"[,;\s]+")))
        .withColumn("hashtag", F.trim("hashtag"))
        .filter(F.length("hashtag") > 1)
        .withColumn("hashtag", F.when(F.col("hashtag").startswith("#"), F.col("hashtag")).otherwise(F.concat(F.lit("#"), F.col("hashtag"))))
    )

    result = tags.groupBy("hashtag").count().orderBy(F.desc("count"), "hashtag").limit(top_n)
    write_csv(result, output_dir, "query_03_top_hashtags", "top_hashtags")


# Query 4 - Parole piu' frequenti.
# Usa tokenizzazione e rimozione delle stopwords per individuare le parole piu'
# ricorrenti nel testo dei tweet.
def frequent_words(tokenized_df, output_dir: str, top_n: int) -> None:
    result = (
        tokenized_df.select(F.explode("words").alias("word"))
        .filter(F.length("word") > 2)
        .groupBy("word")
        .count()
        .orderBy(F.desc("count"), "word")
        .limit(top_n)
    )
    write_csv(result, output_dir, "query_04_frequent_words", "frequent_words")


# Query 5 - Confronto Trump/Harris tramite keyword.
# Misura, per ogni stato, quante volte compaiono keyword associate a Trump e
# Harris, distinguendo menzioni singole e menzioni di entrambi i candidati.
def trump_harris_keyword_comparison(df, output_dir: str) -> None:
    trump_re = keyword_regex(TRUMP_KEYWORDS)
    harris_re = keyword_regex(HARRIS_KEYWORDS)

    flagged = (
        df.withColumn("text_lc", F.lower(F.concat_ws(" ", "text", "hashtags")))
        .withColumn("mentions_trump", F.col("text_lc").rlike(trump_re).cast("int"))
        .withColumn("mentions_harris", F.col("text_lc").rlike(harris_re).cast("int"))
    )

    result = (
        flagged.groupBy("state")
        .agg(
            F.count("*").alias("tweet_count"),
            F.sum("mentions_trump").alias("trump_mentions"),
            F.sum("mentions_harris").alias("harris_mentions"),
            F.sum(
                F.when((F.col("mentions_trump") == 1) & (F.col("mentions_harris") == 0), 1).otherwise(0)
            ).alias("only_trump"),
            F.sum(
                F.when((F.col("mentions_trump") == 0) & (F.col("mentions_harris") == 1), 1).otherwise(0)
            ).alias("only_harris"),
            F.sum(
                F.when((F.col("mentions_trump") == 1) & (F.col("mentions_harris") == 1), 1).otherwise(0)
            ).alias("both_candidates"),
        )
        .withColumn("trump_share", F.round(F.col("trump_mentions") / F.col("tweet_count"), 4))
        .withColumn("harris_share", F.round(F.col("harris_mentions") / F.col("tweet_count"), 4))
        .orderBy("state")
    )
    write_csv(result, output_dir, "query_05_trump_harris_keyword_comparison", "trump_harris_keyword_comparison")


# Query 6 - Co-occorrenza delle parole.
# Costruisce coppie di parole frequenti che appaiono nello stesso tweet, utile
# per capire quali concetti vengono associati piu' spesso nella conversazione.
def word_cooccurrence(tokenized_df, output_dir: str, top_n_words: int, top_n_pairs: int) -> None:
    top_words = (
        tokenized_df.select(F.explode("words").alias("word"))
        .filter(F.length("word") > 2)
        .groupBy("word")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n_words)
    )
    top_word_list = [row["word"] for row in top_words.collect()]

    filtered = tokenized_df.withColumn(
        "top_words_in_tweet",
        F.array_sort(F.array_distinct(F.array_intersect(F.col("words"), F.array(*[F.lit(w) for w in top_word_list])))),
    )

    pairs = (
        filtered.select("top_words_in_tweet")
        .withColumn("word_a", F.explode("top_words_in_tweet"))
        .withColumn("word_b", F.explode("top_words_in_tweet"))
        .filter(F.col("word_a") < F.col("word_b"))
        .groupBy("word_a", "word_b")
        .count()
        .orderBy(F.desc("count"), "word_a", "word_b")
        .limit(top_n_pairs)
    )
    write_csv(pairs, output_dir, "query_06_word_cooccurrence", "word_cooccurrence")


# Query 7 - Polarizzazione territoriale degli utenti.
# Prima assegna a ogni utente un orientamento prevalente nella citta' in cui ha
# twittato, poi misura lo sbilanciamento tra utenti pro-Trump e pro-Harris.
def territorial_polarization(df, output_dir: str, min_city_users: int) -> None:
    classified = df.filter(F.col("binary_label").isin(0, 1) & F.col("user_id").isNotNull())

    user_city_orientation = (
        classified.groupBy("state", "city", "user_id")
        .agg(
            F.count("*").alias("classified_tweets"),
            F.sum(F.when(F.col("binary_label") == 1, 1).otherwise(0)).alias("pro_trump_tweets"),
            F.sum(F.when(F.col("binary_label") == 0, 1).otherwise(0)).alias("pro_harris_tweets"),
        )
        .withColumn(
            "user_orientation",
            F.when(F.col("pro_trump_tweets") > F.col("pro_harris_tweets"), "PRO_TRUMP")
            .when(F.col("pro_harris_tweets") > F.col("pro_trump_tweets"), "PRO_HARRIS")
            .otherwise("BALANCED"),
        )
    )

    result = (
        user_city_orientation.groupBy("state", "city")
        .agg(
            F.count("*").alias("classified_users"),
            F.sum(F.when(F.col("user_orientation") == "PRO_TRUMP", 1).otherwise(0)).alias("pro_trump_users"),
            F.sum(F.when(F.col("user_orientation") == "PRO_HARRIS", 1).otherwise(0)).alias("pro_harris_users"),
            F.sum(F.when(F.col("user_orientation") == "BALANCED", 1).otherwise(0)).alias("balanced_users"),
            F.sum("classified_tweets").alias("classified_tweets"),
            F.sum("pro_trump_tweets").alias("pro_trump_tweets"),
            F.sum("pro_harris_tweets").alias("pro_harris_tweets"),
        )
        .filter(F.col("classified_users") >= min_city_users)
        .withColumn("trump_user_share", F.round(F.col("pro_trump_users") / F.col("classified_users"), 4))
        .withColumn("harris_user_share", F.round(F.col("pro_harris_users") / F.col("classified_users"), 4))
        .withColumn("balanced_user_share", F.round(F.col("balanced_users") / F.col("classified_users"), 4))
        .withColumn(
            "user_polarization_score",
            F.round(F.abs(F.col("pro_trump_users") - F.col("pro_harris_users")) / F.col("classified_users"), 4),
        )
        .withColumn(
            "dominant_user_orientation",
            F.when(F.col("pro_trump_users") > F.col("pro_harris_users"), "PRO_TRUMP")
            .when(F.col("pro_harris_users") > F.col("pro_trump_users"), "PRO_HARRIS")
            .otherwise("BALANCED"),
        )
        .orderBy(F.desc("user_polarization_score"), F.desc("classified_users"))
    )
    write_csv(result, output_dir, "query_07_territorial_polarization", "territorial_polarization")


# Query 8 - K-Means clustering automatico dei tweet.
# Trasforma testo e hashtag in feature TF-IDF normalizzate e raggruppa i tweet
# in cluster tematici non supervisionati. Il CountVectorizer ignora termini
# troppo rari per evitare cluster quasi vuoti formati da outlier. Le descrizioni
# utente sono escluse per evitare cluster dominati da profili molto attivi.
# Le parole candidate/state sono rimosse solo qui per far emergere topic meno ovvi.
def kmeans_clustering(df, output_dir: str, k: int) -> None:
    ml_df = df.select("text", "hashtags", "state", "binary_label").withColumn(
        "features_text", F.concat_ws(" ", "text", "hashtags")
    )

    pipeline = Pipeline(
        stages=[
            RegexTokenizer(inputCol="features_text", outputCol="raw_ml_words", pattern="[^A-Za-z]+", minTokenLength=3),
            StopWordsRemover(
                inputCol="raw_ml_words",
                outputCol="ml_words",
                stopWords=StopWordsRemover.loadDefaultStopWords("english") + KMEANS_STOPWORDS,
            ),
            CountVectorizer(inputCol="ml_words", outputCol="tf", vocabSize=20000, minDF=20.0, maxDF=0.6),
            IDF(inputCol="tf", outputCol="features"),
            Normalizer(inputCol="features", outputCol="normalized_features", p=2.0),
            KMeans(k=k, seed=42, featuresCol="normalized_features", predictionCol="cluster"),
        ]
    )

    model = pipeline.fit(ml_df)
    clustered = model.transform(ml_df)

    cluster_summary = (
        clustered.groupBy("cluster")
        .agg(
            F.count("*").alias("tweet_count"),
            F.sum(F.when(F.col("binary_label") == 1, 1).otherwise(0)).alias("pro_trump"),
            F.sum(F.when(F.col("binary_label") == 0, 1).otherwise(0)).alias("pro_harris"),
            F.sum(F.when(F.col("binary_label").isNull(), 1).otherwise(0)).alias("neutral_or_unclear"),
        )
        .withColumn("pro_trump_share", F.round(F.col("pro_trump") / F.col("tweet_count"), 4))
        .withColumn("pro_harris_share", F.round(F.col("pro_harris") / F.col("tweet_count"), 4))
        .orderBy("cluster")
    )
    write_csv(cluster_summary, output_dir, "query_08_kmeans_clustering", "cluster_summary")

    cluster_terms = (
        clustered.select("cluster", F.explode("ml_words").alias("word"))
        .filter(F.length("word") > 2)
        .groupBy("cluster", "word")
        .count()
        .withColumn("rank", F.row_number().over(Window.partitionBy("cluster").orderBy(F.desc("count"), "word")))
        .filter(F.col("rank") <= 15)
        .orderBy("cluster", "rank")
    )
    write_csv(cluster_terms, output_dir, "query_08_kmeans_clustering", "top_terms")


# Query 9 - SVM per classificazione dell'orientamento politico.
# Addestra una Linear SVM binaria sui tweet etichettati dall'LLM, escludendo i
# neutrali/ambigui, e salva metriche e matrice di confusione.
def svm_classification(df, output_dir: str, feature_columns: list[str]) -> None:
    if not feature_columns:
        raise ValueError("La SVM richiede almeno una feature testuale.")

    labeled = (
        df.filter(F.col("binary_label").isin(0, 1))
        .select(F.col("binary_label").cast("double").alias("label"), *feature_columns)
        .withColumn("features_text", F.concat_ws(" ", *feature_columns))
    )

    train_df, test_df = labeled.randomSplit([0.8, 0.2], seed=42)

    pipeline = Pipeline(
        stages=[
            RegexTokenizer(inputCol="features_text", outputCol="raw_svm_words", pattern="[^A-Za-z]+", minTokenLength=3),
            StopWordsRemover(
                inputCol="raw_svm_words",
                outputCol="svm_words",
                stopWords=StopWordsRemover.loadDefaultStopWords("english") + EXTRA_STOPWORDS,
            ),
            HashingTF(inputCol="svm_words", outputCol="tf", numFeatures=1 << 18),
            IDF(inputCol="tf", outputCol="features"),
            LinearSVC(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=50, regParam=0.1),
        ]
    )

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    metrics = [
        ("area_under_roc", binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})),
        ("accuracy", multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})),
        ("f1", multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})),
        ("weighted_precision", multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})),
        ("weighted_recall", multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})),
        ("train_rows", float(train_df.count())),
        ("test_rows", float(test_df.count())),
    ]

    metrics_df = predictions.sparkSession.createDataFrame(metrics, ["metric", "value"])
    confusion = (
        predictions.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )

    write_csv(metrics_df, output_dir, "query_09_svm_classification", "metrics")
    write_csv(confusion, output_dir, "query_09_svm_classification", "confusion_matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisi Spark sui tweet preprocessati degli swing states.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Glob/path dei CSV puliti.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Cartella output per i risultati.")
    parser.add_argument(
        "--multiline-csv",
        action="store_true",
        help="Abilita la lettura di record CSV multilinea. Piu' robusta ma molto piu' lenta.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        type=int,
        choices=range(1, 10),
        metavar="N",
        default=list(range(1, 10)),
        help="Query da eseguire, da 1 a 9. Esempio: --queries 1 oppure --queries 1 2 7.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Numero di righe per ranking hashtag/parole.")
    parser.add_argument("--cooccurrence-words", type=int, default=50, help="Numero di parole usate per le co-occorrenze.")
    parser.add_argument("--cooccurrence-pairs", type=int, default=100, help="Numero di coppie di co-occorrenza salvate.")
    parser.add_argument(
        "--min-city-users",
        type=int,
        default=1,
        help="Soglia minima di utenti classificati per la polarizzazione cittadina.",
    )
    parser.add_argument("--k", type=int, default=4, help="Numero di cluster K-Means.")
    parser.add_argument(
        "--svm-features",
        nargs="+",
        choices=["text", "hashtags", "description"],
        default=["text", "hashtags", "description"],
        help="Feature testuali per la SVM. Lo stato e' escluso per evitare bias geografico.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    print("[spark] Avvio sessione Spark...", flush=True)
    spark = (
        SparkSession.builder.appName("SwingStatesTweetsAnalysis")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("[spark] Sessione Spark pronta", flush=True)

    selected_queries = set(args.queries)
    tweets = load_tweets(spark, args.input, args.multiline_csv)
    if len(selected_queries) > 1 or not selected_queries.issubset({1, 2, 3, 5, 7}):
        print("[cache] Cache dataset tweet attivata", flush=True)
        tweets = tweets.cache()
    tokenized = add_tokens(tweets).cache() if selected_queries.intersection({4, 6}) else None

    if 1 in selected_queries:
        run_query(1, "Distribuzione geografica", lambda: geographic_distribution(tweets, args.output))
    if 2 in selected_queries:
        run_query(2, "Evoluzione temporale", lambda: temporal_evolution(tweets, args.output))
    if 3 in selected_queries:
        run_query(3, "Top hashtag", lambda: top_hashtags(tweets, args.output, args.top_n))
    if 4 in selected_queries:
        run_query(4, "Parole frequenti", lambda: frequent_words(tokenized, args.output, args.top_n))
    if 5 in selected_queries:
        run_query(5, "Confronto keyword Trump/Harris", lambda: trump_harris_keyword_comparison(tweets, args.output))
    if 6 in selected_queries:
        run_query(6, "Co-occorrenza parole", lambda: word_cooccurrence(tokenized, args.output, args.cooccurrence_words, args.cooccurrence_pairs))
    if 7 in selected_queries:
        run_query(7, "Polarizzazione territoriale", lambda: territorial_polarization(tweets, args.output, args.min_city_users))
    if 8 in selected_queries:
        run_query(8, "K-Means", lambda: kmeans_clustering(tweets, args.output, args.k))
    if 9 in selected_queries:
        run_query(9, "SVM", lambda: svm_classification(tweets, args.output, args.svm_features))

    print("[spark] Stop sessione Spark", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
