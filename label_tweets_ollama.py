import glob
import json
import os
import re
import time
import argparse

import pandas as pd
import requests


# ==========================
# CONFIGURAZIONE
# ==========================

DATA_FOLDER = "CleanData"

TEXT_COLUMN = "text"
HASHTAGS_COLUMN = "hashtags"
LABEL_COLUMN = "binary_label"
CHECKPOINT_ROWS = 100

OLLAMA_MODEL: str | None = None
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

PROMPT = """You are labeling tweet stance for the 2024 US presidential election.

Return exactly one character:
1 = Trump-aligned: pro Donald Trump, pro MAGA, Republican campaign support, or anti-Harris / anti-Democrat.
0 = Harris-aligned: pro Kamala Harris, pro Democratic campaign support, vote blue, or anti-Trump / anti-MAGA.
N = neutral, unclear, factual news, joke without clear stance, spam, or impossible to infer.

Important rules:
- Do NOT label a tweet as 1 just because it mentions "Trump", "MAGA", or has a Trump hashtag.
- Do NOT label a tweet as 0 just because it mentions "Harris", "Kamala", or has a Harris hashtag.
- Criticism of Trump/MAGA is 0, even if the tweet contains #Trump, #Trump2024, or MAGA.
- Criticism of Harris/Democrats/Biden is 1, even if the tweet contains Harris hashtags.
- Endorsements, "vote for", praise, slogans, and attacks on the opponent are stance signals.
- "Vote blue", "VOTE BLUE", "stop Trump", "never Trump", "fuck Trump", and anti-MAGA criticism are 0.
- "Trump 2024", "MAGA", "vote Trump", "vote red", "TDS", and criticism of Kamala/Biden/Democrats are 1,
  unless the surrounding text is clearly mocking or attacking Trump/MAGA.
- If the tweet is only reporting news without approval or criticism, return N.
- Use hashtags as context, but the tweet text can override misleading hashtags.

Examples:
Tweet: Fuck Donald Trump.
Hashtag: none
Answer: 0

Tweet: Never Trump! Vote blue.
Hashtag: #Trump2024
Answer: 0

Tweet: Marc Anthony Ain't Playing With MAGA!!! VOTE BLUE
Hashtag: none
Answer: 0

Tweet: MAGA men seem to think women don't have rights.
Hashtag: none
Answer: 0

Tweet: Trump 2024. Make America Great Again!
Hashtag: #MAGA
Answer: 1

Tweet: Wearing a MAGA hat. Make America Great Again! Let's go!!
Hashtag: none
Answer: 1

Tweet: Kamala Harris will destroy America.
Hashtag: #KamalaHarris2024
Answer: 1

Tweet: Kamala has the worst case of TDS. Try posting your own policies.
Hashtag: #Trump2024
Answer: 1

Tweet: NBA legend LeBron James has endorsed Kamala Harris.
Hashtag: #KamalaHarris2024
Answer: 0

Tweet: CNN reports Trump will hold a rally in Arizona.
Hashtag: #Election2024
Answer: N

Now label this tweet.
Tweet: {text}
Hashtag: {hashtags}
Answer:"""


def get_rule_based_label(text: str, hashtags: str) -> int | None:
    """Gestisce segnali molto espliciti prima di interrogare il modello."""
    content = f"{text} {hashtags}".lower()

    harris_aligned_patterns = [
        r"\bvote\s+blue\b",
        r"#voteblue\b",
        r"\bnever\s+trump\b",
        r"\bstop\s+trump\b",
        r"\bdefeat\s+trump\b",
        r"\bdump\s+trump\b",
        r"\bfuck\s+(donald\s+)?trump\b",
        r"\bfuck\s+maga\b",
        r"\banti[-\s]?trump\b",
        r"\banti[-\s]?maga\b",
        r"\btrump\b.{0,80}\b(rapist|convicted felon|sexual abuser|pedophile|treason)\b",
        r"\b(chickenshit|coward|afraid)\b.{0,80}\btrump\b",
        r"\btrump\b.{0,80}\b(chickenshit|coward|afraid)\b",
    ]

    trump_aligned_patterns = [
        r"\bvote\s+trump\b",
        r"\bvote\s+red\b",
        r"\btrump\s*2024\b",
        r"#trump2024\b",
        r"#trumptosaveamerica\b",
        r"\bmake\s+america\s+great\s+again\b",
        r"\bkamala\b.{0,100}\b(tds|destroy|catastrophe|worst|weak|failed|liar|corrupt)\b",
        r"\bharris\b.{0,100}\b(tds|destroy|catastrophe|worst|weak|failed|liar|corrupt)\b",
        r"\bbiden\b.{0,100}\b(corrupt|weak|failed|destroyed|garbage)\b",
        r"\bdemocrats?\b.{0,100}\b(corrupt|weak|failed|destroyed|garbage)\b",
        r"\bwhy are you so obsessed with trump\b",
        r"\bworse case of tds\b",
        r"\bworst case of tds\b",
    ]

    # Gli attacchi a Trump/MAGA hanno priorita' sugli hashtag pro-Trump ingannevoli.
    if any(re.search(pattern, content) for pattern in harris_aligned_patterns):
        return 0

    if any(re.search(pattern, content) for pattern in trump_aligned_patterns):
        return 1

    return None


def get_model_label(text: str, hashtags: str) -> int | None:
    """Restituisce 1 (Trump), 0 (Harris) o None (neutro/ambiguo)."""
    if OLLAMA_MODEL is None:
        raise RuntimeError("Specifica il modello Ollama con --model.")

    rule_label = get_rule_based_label(text, hashtags)
    if rule_label is not None:
        return rule_label

    prompt = PROMPT.format(text=text, hashtags=hashtags or "nessuno")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip().upper()

            # Accetta anche risposte accidentali come "Risposta: 1".
            match = re.search(r"(?<!\d)([01N])(?!\d)", answer)
            if not match:
                return None

            value = match.group(1)
            return int(value) if value in {"0", "1"} else None
        except (requests.RequestException, json.JSONDecodeError) as error:
            if attempt == MAX_RETRIES:
                print(f"  Errore Ollama dopo {MAX_RETRIES} tentativi: {error}")
                return None
            time.sleep(attempt * 2)

    return None


def save_csv(df: pd.DataFrame, input_path: str) -> None:
    df.to_csv(input_path, index=False, encoding="utf-8")


def process_csv(
    input_path: str,
    relabel_existing: bool = False,
    checkpoint_rows: int = CHECKPOINT_ROWS,
) -> None:
    df = pd.read_csv(input_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"La colonna '{TEXT_COLUMN}' non esiste in {input_path}.")

    if LABEL_COLUMN not in df.columns:
        df[LABEL_COLUMN] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype("Int64")

    hashtags = (
        df[HASHTAGS_COLUMN].fillna("").astype(str)
        if HASHTAGS_COLUMN in df.columns
        else pd.Series("", index=df.index)
    )

    total = len(df)
    already_labeled = df[LABEL_COLUMN].notna().sum()
    print(f"Etichetto {total} tweet: {os.path.basename(input_path)}")
    if already_labeled and not relabel_existing:
        print(f"  Resume attivo: salto {already_labeled} tweet gia' etichettati")
    elif already_labeled and relabel_existing:
        print(f"  Relabel attivo: ricalcolo anche {already_labeled} tweet gia' etichettati")

    changed_since_save = 0
    processed = 0

    for position, (text, tweet_hashtags) in enumerate(zip(df[TEXT_COLUMN], hashtags), start=1):
        if not relabel_existing and pd.notna(df.at[position - 1, LABEL_COLUMN]):
            continue

        label = get_model_label(str(text), tweet_hashtags)
        df.at[position - 1, LABEL_COLUMN] = label if label is not None else pd.NA
        changed_since_save += 1
        processed += 1

        if processed % 25 == 0:
            print(f"  Processati {processed} nuovi tweet ({position}/{total})")

        if checkpoint_rows > 0 and changed_since_save >= checkpoint_rows:
            save_csv(df, input_path)
            changed_since_save = 0
            print(f"  Checkpoint salvato ({position}/{total})")

    # Sovrascrive il CSV pulito aggiungendo (o aggiornando) binary_label.
    save_csv(df, input_path)
    labeled = df[LABEL_COLUMN].notna().sum()
    print(f"Aggiornato: {input_path} ({labeled}/{total} tweet con etichetta binaria)")


def main() -> None:
    global OLLAMA_MODEL

    parser = argparse.ArgumentParser(
        description="Etichetta tweet cleaned con Ollama e salva binary_label nei CSV."
    )
    parser.add_argument(
        "--data-folder",
        default=DATA_FOLDER,
        help="Cartella contenente i CSV cleaned.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Uno o piu' nomi/path CSV da processare. Se omesso, processa tutti i CSV della cartella.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Modello Ollama da usare, come visualizzato da: ollama list.",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Ricalcola anche le righe che hanno gia' binary_label.",
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=CHECKPOINT_ROWS,
        help="Salva il CSV ogni N tweet processati. Usa 0 per salvare solo alla fine.",
    )
    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    if args.files:
        csv_files = [
            path if os.path.isabs(path) else os.path.join(args.data_folder, path)
            for path in args.files
        ]
    else:
        pattern = os.path.join(args.data_folder, "*.csv")
        csv_files = glob.glob(pattern)

    if not csv_files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {args.data_folder}."
        )

    for csv_file in csv_files:
        process_csv(
            csv_file,
            relabel_existing=args.relabel,
            checkpoint_rows=args.checkpoint_rows,
        )


if __name__ == "__main__":
    main()
