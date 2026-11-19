import argparse
import json
import os
import re
import time
import requests
import yaml
from openapi_schema_validator import OAS30Validator

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def extract_yaml(raw_text: str) -> str:
    """Extracts clean YAML from the model response (after the FINAL_YAML: marker or inside ```yaml code fences)."""
    text = raw_text.strip()

    # 1. If markdown fences exist, take the LAST one (CoT / CoVe)
    fences = re.findall(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()

    # 2. Fallback if no markdown fences are present
    if "FINAL_YAML:" in text:
        text = text.split("FINAL_YAML:")[-1].strip()

    # Match 'type: object' at column 0 only (schema root, never a nested object)
    matches = list(re.finditer(r"^type:\s*object\b", text, re.MULTILINE))
    if matches:
        text = text[matches[-1].start():]

    return text.strip()
    


# --- TECHNIQUE 1: Few-Shot Prompting (Taxonomy: New Tasks Without Extensive Training) ---
def run_few_shot(input_json: str) -> str:
    prompt = f"""You are an expert backend API architect. Generate a valid OpenAPI 3.0 YAML schema for the provided JSON.
Rules:
- Return ONLY raw YAML. No explanations, no markdown fences.
- Map null values to: type: string with nullable: true.
- Map empty arrays to: type: array with items: {{}}.
- Use integer for whole numbers and number for floats.

EXAMPLE 1 INPUT:
{{"status": "ok", "count": 5, "error": null, "tags": []}}

EXAMPLE 1 OUTPUT:
type: object
properties:
  status:
    type: string
  count:
    type: integer
  error:
    type: string
    nullable: true
  tags:
    type: array
    items: {{}}

EXAMPLE 2 INPUT:
{{"service": "auth", "roles": ["admin"], "meta": {{"active": true, "latency": 12.5}}, "items": [{{"id": 1}}]}}

EXAMPLE 2 OUTPUT:
type: object
properties:
  service:
    type: string
  roles:
    type: array
    items:
      type: string
  meta:
    type: object
    properties:
      active:
        type: boolean
      latency:
        type: number
  items:
    type: array
    items:
      type: object
      properties:
        id:
          type: integer

ACTUAL INPUT:
{input_json}

ACTUAL OUTPUT:"""
    return extract_yaml(call_ollama(prompt))


# --- TECHNIQUE 2: Chain-of-Thought (Taxonomy: Reasoning and Logic) ---
def run_cot(input_json: str) -> str:
    prompt = f"""You are an expert backend API architect. Convert the following JSON object into a valid OpenAPI 3.0 YAML Schema Object.

Strict formatting rules:
- Output ONLY the Schema Object starting directly with 'type: object' and 'properties:'.
- Do NOT generate a full OpenAPI document (do NOT include 'openapi:', 'info:', 'paths:', or 'components:').
- Do NOT add extra fields like 'required:', 'additionalProperties:', 'description:', or 'example:'.
- Map null values to: 'type: string' with 'nullable: true'.
- Map empty arrays [] to:
  type: array
  items: {{}}
- Map non-empty string arrays ["a", "b"] to:
  type: array
  items:
    type: string
  (NEVER write 'items: string' or 'items: type: string' on the same line!)
- Map non-empty object arrays [{{...}}] to 'type: array' with 'items:' containing 'type: object' and 'properties:'.
- Use 'integer' for whole numbers and 'number' for decimals.

Think step by step:
1. List every top-level key and nested key in the JSON with its exact OpenAPI 3.0 type.
2. Check special cases (null values, empty arrays [], non-empty arrays).
3. Keep your reasoning concise (max 5 bullet points), then output the final YAML Schema Object inside a single ```yaml code block starting with 'type: object'.

JSON INPUT:
{input_json}

Let's think step by step:"""
    return extract_yaml(call_ollama(prompt))


# --- TECHNIQUE 3: Chain-of-Verification / CoVe (Taxonomy: Reduce Hallucination) ---
def run_cove(input_json: str) -> str:
    # Step 1: Generate initial draft
    draft_prompt = f"""You are an expert backend API architect. Generate an OpenAPI 3.0 YAML Schema Object for this JSON.
Rules:
- Start directly with 'type: object' and 'properties:'.
- Do NOT include 'openapi:', 'info:', 'paths:', 'components:', 'required:', or 'example:'.
- Map null values to: 'type: string' with 'nullable: true'.
- Map empty arrays [] to: 'type: array' with 'items: {{}}'.
- Map non-empty string arrays ["a"] to:
  type: array
  items:
    type: string
- Use 'integer' for whole numbers and 'number' for decimals.
- Return ONLY raw YAML inside a ```yaml code block.

JSON:
{input_json}

YAML:"""
    draft_yaml = extract_yaml(call_ollama(draft_prompt))

    # Step 2: Verify draft and produce corrected output
    verify_prompt = f"""You are an OpenAPI 3.0 schema auditor. Verify the DRAFT YAML against the ORIGINAL JSON and fix ONLY actual errors.

ORIGINAL JSON:
{input_json}

DRAFT YAML:
{draft_yaml}

Verification checklist:
1. Exact keys: Does DRAFT YAML contain the exact same keys as ORIGINAL JSON at the same nesting levels? Do NOT invent or add any extra keys (such as 'nullable' or 'required') that are not in ORIGINAL JSON!
2. Numbers: Are whole numbers typed as 'integer' and decimals typed as 'number'?
3. Null values (ONLY if ORIGINAL JSON contains a null value): Is that specific key typed as 'type: string' with 'nullable: true'?
4. Empty arrays (ONLY if ORIGINAL JSON contains []): Does that key use 'type: array' and 'items: {{}}'? Do NOT change 'items: {{}}' to 'type: string' or 'type: object' if the JSON array is empty `[]`!
5. Non-empty arrays (ONLY if ORIGINAL JSON contains elements like ["a", "b"] or [{{...}}]): Does 'items:' have 'type: string' (for string arrays) or 'type: object' (for object arrays) indented on the next line? (Never write 'items: string' or 'items: type: string' on one line).

Briefly answer the 5 checks (max 5 short lines). If DRAFT YAML is already correct, keep it unchanged. Output the final OpenAPI 3.0 YAML Schema Object inside a single ```yaml code block starting with 'type: object'."""
    return extract_yaml(call_ollama(verify_prompt))


def evaluate_prediction(input_json_str: str, target_yaml_str: str, pred_yaml_str: str) -> dict:
    input_obj = json.loads(input_json_str)
    target_schema = yaml.safe_load(target_yaml_str)

    yaml_valid = False
    oas_valid = False
    json_valid = False
    exact_match = False
    error_msg = None

    try:
        pred_schema = yaml.safe_load(pred_yaml_str)
        if not isinstance(pred_schema, dict):
            raise ValueError("Parsed YAML is not a dictionary")
        yaml_valid = True

        if pred_schema.get("type") != "object" or "properties" not in pred_schema:
            raise ValueError("Root schema must be an OpenAPI Schema Object with 'type: object' and 'properties'")

        OAS30Validator.check_schema(pred_schema)
        oas_valid = True

        OAS30Validator(pred_schema).validate(input_obj)
        json_valid = True

        exact_match = pred_schema == target_schema
    except Exception as e:
        error_msg = str(e)[:200]

    return {
        "yaml_valid": yaml_valid,
        "oas_valid": oas_valid,
        "json_valid": json_valid,
        "exact_match": exact_match,
        "error": error_msg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of test samples to evaluate")
    args = parser.parse_args()

    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f][: args.limit]

    techniques = {
        "few_shot": run_few_shot,
        "cot": run_cot,
        "cove": run_cove,
    }

    os.makedirs("results", exist_ok=True)
    summary = {}

    for name, fn in techniques.items():
        print(f"\n=== Evaluating technique: {name} ({len(test_data)} samples) ===")
        records = []
        t0_all = time.time()

        for i, row in enumerate(test_data, 1):
            t0 = time.time()
            pred_yaml = fn(row["input_json"])
            elapsed = round(time.time() - t0, 2)

            metrics = evaluate_prediction(row["input_json"], row["target_yaml"], pred_yaml)
            records.append({
                "index": i,
                "latency_s": elapsed,
                "pred_yaml": pred_yaml,
                **metrics,
            })

            if i % 5 == 0 or i == len(test_data):
                print(
                    f"  [{i}/{len(test_data)}] "
                    f"OAS_valid={sum(r['oas_valid'] for r in records)}/{i}, "
                    f"JSON_valid={sum(r['json_valid'] for r in records)}/{i}, "
                    f"Exact={sum(r['exact_match'] for r in records)}/{i}"
                )

        n = len(records)
        stats = {
            "count": n,
            "yaml_parse_rate": round(100 * sum(r["yaml_valid"] for r in records) / n, 2),
            "oas_schema_valid_rate": round(100 * sum(r["oas_valid"] for r in records) / n, 2),
            "json_validation_accuracy": round(100 * sum(r["json_valid"] for r in records) / n, 2),
            "exact_match_rate": round(100 * sum(r["exact_match"] for r in records) / n, 2),
            "avg_latency_s": round(sum(r["latency_s"] for r in records) / n, 2),
            "total_time_s": round(time.time() - t0_all, 2),
        }
        summary[name] = stats

        with open(f"results/ollama_{name}.json", "w", encoding="utf-8") as out:
            json.dump({"stats": stats, "records": records}, out, indent=2, ensure_ascii=False)

    with open("results/ollama_summary.json", "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)

    print("\n=== FINAL SUMMARY (OLLAMA) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()