import glob
import json
import os
import re
import time

import pandas as pd
import requests


# ==========================
# CONFIGURAZIONE
# ==========================

DATA_FOLDER = "CleanData"

TEXT_COLUMN = "text"
HASHTAGS_COLUMN = "hashtags"
LABEL_COLUMN = "binary_label"

# Modifica questo valore con il nome visualizzato da: ollama list
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

PROMPT = """Classifica l'orientamento politico del tweet rispetto alle elezioni USA 2024.

Regole:
- Rispondi 1 solo se il tweet e' chiaramente a favore di Donald Trump / MAGA.
- Rispondi 0 solo se il tweet e' chiaramente a favore di Kamala Harris / Democratici.
- Rispondi N se il tweet e' neutro, ambiguo, ironico, contiene solo notizie o non esprime
  una preferenza chiara tra Trump e Harris.
- Non dedurre l'orientamento solo dal fatto che sia citato un candidato.
- Usa anche gli hashtag come contesto.

Tweet: {text}
Hashtag: {hashtags}

Rispondi esclusivamente con un singolo carattere: 1, 0 oppure N."""


def get_model_label(text: str, hashtags: str) -> int | None:
    """Restituisce 1 (Trump), 0 (Harris) o None (neutro/ambiguo)."""
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


def process_csv(input_path: str) -> None:
    df = pd.read_csv(input_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"La colonna '{TEXT_COLUMN}' non esiste in {input_path}.")

    hashtags = (
        df[HASHTAGS_COLUMN].fillna("").astype(str)
        if HASHTAGS_COLUMN in df.columns
        else pd.Series("", index=df.index)
    )

    labels = []
    total = len(df)
    print(f"Etichetto {total} tweet: {os.path.basename(input_path)}")

    for position, (text, tweet_hashtags) in enumerate(zip(df[TEXT_COLUMN], hashtags), start=1):
        label = get_model_label(str(text), tweet_hashtags)
        labels.append(label)

        if position % 25 == 0 or position == total:
            print(f"  Completati {position}/{total}")

    # Int64 consente valori 0/1 e celle vuote per i tweet neutrali o non classificabili.
    df[LABEL_COLUMN] = pd.Series(labels, index=df.index, dtype="Int64")

    # Sovrascrive il CSV pulito aggiungendo (o aggiornando) binary_label.
    df.to_csv(input_path, index=False, encoding="utf-8")
    labeled = df[LABEL_COLUMN].notna().sum()
    print(f"Aggiornato: {input_path} ({labeled}/{total} tweet con etichetta binaria)")


def main() -> None:
    pattern = os.path.join(DATA_FOLDER, "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {DATA_FOLDER}."
        )

    for csv_file in csv_files:
        process_csv(csv_file)


if __name__ == "__main__":
    main()
