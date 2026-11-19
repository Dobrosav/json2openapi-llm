import json
import os
import random
import yaml
from datasets import load_dataset
from openapi_schema_validator import OAS30Validator
from sklearn.model_selection import train_test_split


def short_str(val, max_len=40):
    s = str(val) if val is not None else ""
    return s[:max_len] + "..." if len(s) > max_len else s


def infer_openapi_schema(value):
    """Deterministically generates a valid OpenAPI 3.0 schema for the provided JSON."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, str):
        return {"type": "string"}
    elif value is None:
        return {"type": "string", "nullable": True}
    elif isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        return {"type": "array", "items": infer_openapi_schema(value[0])}
    elif isinstance(value, dict):
        return {
            "type": "object",
            "properties": {k: infer_openapi_schema(v) for k, v in value.items()},
        }
    return {"type": "string"}


def build_varied_payload(row, rng):
    """Builds a varied REST JSON payload from a real APIBench record."""
    api_data = row.get("api_data") if isinstance(row.get("api_data"), dict) else {}

    api_name = short_str(api_data.get("api_name", "unknown_api"), 30)
    provider = short_str(row.get("provider", "generic"), 25)
    domain = short_str(api_data.get("domain", "general"), 25)
    framework = short_str(api_data.get("framework", "python"), 25)
    api_call = short_str(row.get("api_call", ""), 40)

    raw_args = api_data.get("api_arguments")
    args = [short_str(a, 20) for a in raw_args[:3]] if isinstance(raw_args, list) else []

    raw_reqs = api_data.get("python_environment_requirements")
    reqs = [short_str(r, 20) for r in raw_reqs[:3]] if isinstance(raw_reqs, list) else []

    # Pool of candidate fields with diverse OpenAPI types
    candidates = [
        ("id", rng.randint(1, 99999)),
        ("api_name", api_name),
        ("provider", provider),
        ("domain", domain),
        ("framework", framework),
        ("endpoint", api_call),
        ("active", rng.choice([True, False])),
        ("deprecated", rng.choice([True, False])),
        ("latency_ms", round(rng.uniform(10.5, 450.9), 2)),
        ("score", round(rng.uniform(0.5, 1.0), 3)),
        ("arg_count", len(args)),
        ("status_code", rng.choice([200, 201, 202, 400, 404])),
        ("error", None),
        ("arguments", args if rng.random() > 0.25 else []),
        ("requirements", reqs if rng.random() > 0.25 else []),
        ("tags", [domain, framework] if rng.random() > 0.3 else []),
    ]

    # Randomly select 3 to 6 fields
    picked = dict(rng.sample(candidates, k=rng.randint(3, 6)))

    # Randomly choose nesting structure (flat, nested object, or array of objects)
    style = rng.choice(["flat", "nested_meta", "nested_config", "with_items"])

    if style == "flat":
        return picked

    elif style == "nested_meta":
        return {
            "status": "ok",
            "data": picked,
            "meta": {
                "version": rng.randint(1, 3),
                "cached": rng.choice([True, False]),
            },
        }

    elif style == "nested_config":
        return {
            "service": api_name,
            "config": picked,
            "warnings": [],
        }

    else:  # with_items (array of objects)
        return {
            "provider": provider,
            "total": len(args) + 1,
            "entries": [
                {"name": api_name, "enabled": True, "weight": round(rng.uniform(0.1, 1.0), 2)}
            ],
            "details": picked,
        }


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(42)
    print("Downloading gorilla-llm/APIBench...")
    ds = load_dataset("gorilla-llm/APIBench", split="train")

    pairs = []
    for row in ds:
        payload = build_varied_payload(dict(row), rng)
        schema = infer_openapi_schema(payload)

        try:
            OAS30Validator.check_schema(schema)
            OAS30Validator(schema).validate(payload)
        except Exception:
            continue

        yaml_str = yaml.safe_dump(schema, sort_keys=False).strip()
        pairs.append({
            "input_json": json.dumps(payload, ensure_ascii=False),
            "target_yaml": yaml_str,
        })

    print(f"Total valid pairs: {len(pairs)}")

    train_data, temp_data = train_test_split(pairs, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    test_data = test_data[:100]

    os.makedirs("data", exist_ok=True)
    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(val_data, "data/val.jsonl")
    save_jsonl(test_data, "data/test.jsonl")

    unique_test_schemas = len({r["target_yaml"] for r in test_data})
    print(f"Saved -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    print(f"Unique YAML schemas in test set: {unique_test_schemas}/{len(test_data)}")


if __name__ == "__main__":
    main()