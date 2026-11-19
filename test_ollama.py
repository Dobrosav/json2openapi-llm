import json
import requests
import yaml
from openapi_schema_validator import OAS30Validator

OLLAMA_API_URL = "http://localhost:11434/api/generate"
sample_json = {
    "patient": {"id": "p-456", "name": "Petar", "age": 34},
    "active": True,
    "history": []
}

prompt = f"""You are an expert backend API architect. Generate a valid OpenAPI 3.0 YAML schema definition for the provided JSON.
Return ONLY raw YAML. Do NOT use markdown code blocks or explanations.
In OpenAPI 3.0, every 'array' type MUST have 'items' as a valid schema object (use 'items: {{}}' for empty arrays, NEVER 'items: null').

EXAMPLE INPUT:
{{"status": "success", "count": 5, "items_list": []}}

EXAMPLE OUTPUT:
type: object
properties:
  status:
    type: string
  count:
    type: integer
  items_list:
    type: array
    items: {{}}

ACTUAL INPUT:
{json.dumps(sample_json)}

ACTUAL OUTPUT:"""

response = requests.post(OLLAMA_API_URL, json={
    "model": "llama3",
    "prompt": prompt,
    "stream": False
})

raw_yaml = response.json()["response"].strip()
print("--- GENERATED YAML ---")
print(raw_yaml)

schema = yaml.safe_load(raw_yaml)


OAS30Validator.check_schema(schema)

OAS30Validator(schema).validate(sample_json)

print("\n SUCCESS SCHEME VALIDATION!")