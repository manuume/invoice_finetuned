import os
import base64
import httpx
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pymongo import MongoClient
from datetime import datetime

app = FastAPI(title="Receipt Extraction API - Final")
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "manohar3181/invoice_model"
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MongoDB connection string not found. Please set the MONGO_URI environment variable.")

client = MongoClient(MONGO_URI)
db = client.receipt_db
collection = db.extractions
print("✅ Connected to MongoDB.")

schema_dict = {
    "menu": [{"nm": "string", "cnt": "number", "price": "number"}],
    "sub_total": {"subtotal_price": "number", "tax_price": "number"},
    "total": {"total_price": "number"}
}
PROMPT = f"""You are an expert receipt data extractor. Your job is to extract all information from the provided receipt image into the following JSON format. If a value is not present, use an empty string or null.
JSON Schema:
{json.dumps(schema_dict, indent=2)}
"""
client_http = httpx.AsyncClient(timeout=120.0)

def postprocess(text: str):
    try:
        if "```json" in text:
            cleaned_text = text.split("```json")[1].split("```")[0].strip()
        else:
            cleaned_text = text[text.find('{'):text.rfind('}')+1]
        return json.loads(cleaned_text)
    except:
        return {"error": "Failed to parse JSON", "raw_text": text}

@app.post("/extract/")
async def extract_data(file: UploadFile = File(...)):
    image_bytes = await file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.0
    }

    try:
        response = await client_http.post(VLLM_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"]
        parsed_json = postprocess(raw_content)

        if "error" not in parsed_json:
            db_entry = {
                "filename": file.filename,
                "extracted_data": parsed_json,
                "processed_at": datetime.utcnow()
            }
            collection.insert_one(db_entry)

        return parsed_json

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with vLLM server: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")