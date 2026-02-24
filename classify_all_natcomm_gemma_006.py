import pandas as pd
import subprocess
import json
import os
import time
import sys
import re
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================
# 設定
# ==============================

BASE_DIR = "data_availability_project"
INPUT_DIR = os.path.join(BASE_DIR, "data_availability_all_1_1a")
OUTPUT_DIR = os.path.join(BASE_DIR, "classified_natcomm_2023_by_gemma")
FINAL_OUTPUT = os.path.join(BASE_DIR, "classified_natcomm_2023_by_gemma_all.csv")

LLAMA_MODEL = "gemma:2b"
MAX_RETRY = 3
DATA_COLUMN = "data"

# 🔥 8コア最適化
MAX_WORKERS = 6  # 8コア - 余裕2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 分類定義
# ==============================

CATEGORY_PROMPT = """
You must classify the Data Availability statement into EXACTLY ONE category.

Categories ordered from highest to lowest public accessibility:

1. Fully Public Repository Deposition
2. Public Within Article / Supplement Only
3. Reuse of Public Third-Party Data Only
4. Mixed Public Deposit + Author Request
5. Controlled-Access Repository Data
6. Author Upon Request Only
7. No Data Generated / Not Applicable

Strict rules:
- Choose EXACTLY ONE category.
- Follow the accessibility hierarchy.
- Output ONLY valid JSON.
- Do not use markdown.
- Do not wrap in ```json.
- Do not add explanations outside JSON.

Output format:
{
  "category": "EXACT CATEGORY NAME",
  "reason": "short explanation"
}
"""

# ==============================
# LLM呼び出し
# ==============================

def call_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", LLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


def extract_json(text):
    text = text.replace("```json", "")
    text = text.replace("```", "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def classify_text(text):

    prompt = f"""
{CATEGORY_PROMPT}

Data Availability Statement:
\"\"\"{text}\"\"\"

Classify now.
"""

    for _ in range(MAX_RETRY):

        output = call_llama(prompt)
        cleaned_json = extract_json(output)

        if cleaned_json:
            try:
                parsed = json.loads(cleaned_json)
                return parsed["category"], parsed["reason"]
            except Exception:
                pass

        time.sleep(0.5)

    return "ERROR", output


# ==============================
# 並列処理対象
# ==============================

def process_file(input_path):

    filename = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        df = pd.read_csv(input_path)
        df.columns = df.columns.str.strip()

        if DATA_COLUMN not in df.columns:
            return f"Column missing: {filename}"

        text = str(df.iloc[0][DATA_COLUMN])

        category, reason = classify_text(text)

        df["Category"] = category
        df["Reason"] = reason

        df.to_csv(output_path, index=False)

        return f"Done: {filename}"

    except Exception as e:
        return f"Error in {filename}: {str(e)}"


# ==============================
# 統合処理
# ==============================

def merge_outputs():

    csv_files = glob(os.path.join(OUTPUT_DIR, "*.csv"))

    if not csv_files:
        print("No classified files found.")
        return

    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df.to_csv(FINAL_OUTPUT, index=False)

    print(f"\nFinal merged file saved to: {FINAL_OUTPUT}")


# ==============================
# メイン（並列）
# ==============================

def main():

    if not os.path.exists(INPUT_DIR):
        print("Input directory not found:", INPUT_DIR)
        sys.exit(1)

    input_files = glob(os.path.join(INPUT_DIR, "*.csv"))

    if not input_files:
        print("No CSV files found.")
        sys.exit(1)

    processed_files = set(
        os.path.basename(f)
        for f in glob(os.path.join(OUTPUT_DIR, "*.csv"))
    )

    remaining_files = [
        f for f in input_files
        if os.path.basename(f) not in processed_files
    ]

    print(f"Total files: {len(input_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining: {len(remaining_files)}")
    print(f"Using {MAX_WORKERS} workers\n")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_file, file_path): file_path
            for file_path in remaining_files
        }

        for future in as_completed(futures):
            print(future.result())

    merge_outputs()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()