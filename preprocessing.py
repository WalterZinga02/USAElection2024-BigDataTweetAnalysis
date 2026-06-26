import os
import re
import glob
import pandas as pd
from ftfy import fix_text


# ==========================
# CONFIGURAZIONE
# ==========================

INPUT_OUTPUT_FOLDER = "RawData"

TEXT_COLUMN = "text"
USERNAME_COLUMN = "username"
USER_DESCRIPTION_COLUMN = "description"

OUTPUT_SUFFIX = "_cleaned"
REMOVE_EMPTY_TEXT = True

STATE_MAP = {
    "arizona": "Arizona",
    "georgia": "Georgia",
    "michigan": "Michigan",
    "nevada": "Nevada",
    "north_carolina": "North Carolina",
    "pennsylvania": "Pennsylvania",
    "wisconsin": "Wisconsin",
}


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|https\S+|www\S+", "", text)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#\w+", text)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = fix_text(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = normalize_spaces(text)

    return text

def clean_description(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[^A-Za-z0-9À-ÖØ-öø-ÿ\s]", "", text)  # Rimuove caratteri speciali indesiderati
    return text

def clean_tweet(text: str) -> tuple[str, str]:
    if pd.isna(text):
        return "", ""

    text = fix_text(str(text))

    hashtags = extract_hashtags(text)

    # Rimuove hashtag dal testo del tweet
    text = re.sub(r"#\w+", "", text)

    text = clean_text(text)

    return text, ";".join(hashtags)


def clean_username(username: str) -> str:
    username = clean_text(username)

    username = re.sub(r"[^A-Za-z0-9À-ÖØ-öø-ÿ\s]", "", username)
    username = normalize_spaces(username)

    return username


def process_csv(input_path: str):
    df = pd.read_csv(input_path)

    # Rinomina la colonna della città
    if "Extracted_Location" in df.columns:
        df.rename(columns={"Extracted_Location": "city"}, inplace=True)

    # Elimina la location grezza
    if "location_raw" in df.columns:
        df.drop(columns=["location_raw"], inplace=True)

    # Ricava lo stato dal nome del file
    filename_lower = os.path.basename(input_path).lower()

    state = None
    for key, value in STATE_MAP.items():
        if key in filename_lower:
            state = value
            break

    df["state"] = state

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"La colonna '{TEXT_COLUMN}' non esiste in {input_path}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    cleaned_tweets = df[TEXT_COLUMN].apply(
        lambda x: pd.Series(clean_tweet(x))
    )

    df[TEXT_COLUMN] = cleaned_tweets[0]
    df["hashtags"] = cleaned_tweets[1]

    if USERNAME_COLUMN in df.columns:
        df[USERNAME_COLUMN] = df[USERNAME_COLUMN].apply(clean_username)

    if USER_DESCRIPTION_COLUMN in df.columns:
        df[USER_DESCRIPTION_COLUMN] = df[USER_DESCRIPTION_COLUMN].apply(clean_description)

    if REMOVE_EMPTY_TEXT:
        df = df[df[TEXT_COLUMN].str.strip() != ""]
    
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    output_path = os.path.join(
        INPUT_OUTPUT_FOLDER,
        f"{name}{OUTPUT_SUFFIX}{ext}"
    )

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Creato: {output_path}")


def main():
    csv_files = glob.glob(os.path.join(INPUT_OUTPUT_FOLDER, "*.csv"))

    csv_files = [
        f for f in csv_files
        if not os.path.basename(f).replace(".csv", "").endswith(OUTPUT_SUFFIX)
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"Nessun CSV da processare trovato in {INPUT_OUTPUT_FOLDER}"
        )

    for csv_file in csv_files:
        process_csv(csv_file)


if __name__ == "__main__":
    main()