"""
tests/test_model.py

Unit tests for model utilities.
No actual model weights loaded — tests JSON parsing, field extraction, etc.
"""
import pytest
from src.data.dataset import extract_target_fields, format_target_as_json
from src.model.qwen_model import _parse_json_response


# ── JSON parsing ──────────────────────────────────────────────────────────────

def test_parse_clean_json():
    raw = '{"store_name":"CU","total_price":"12500","tax_price":null,"items":[]}'
    result = _parse_json_response(raw)
    assert result["store_name"] == "CU"
    assert result["total_price"] == "12500"


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"store_name":"CU","total_price":"12500"}\n```'
    result = _parse_json_response(raw)
    assert result["store_name"] == "CU"


def test_parse_invalid_json_returns_raw():
    raw = "Sorry, I cannot extract from this image."
    result = _parse_json_response(raw)
    assert result.get("parse_error") is True
    assert "raw_output" in result


# ── CORD field extraction ─────────────────────────────────────────────────────

SAMPLE_GT = {
    "gt_parse": {
        "store_info": {"store_name": "GS25 Gangnam"},
        "total": {"total_price": "15000", "tax_price": "1500"},
        "menu": [
            {"nm": "Americano", "price": "3000"},
            {"nm": "Croissant", "price": "2500"},
        ],
    }
}


def test_extract_store_name():
    target = extract_target_fields(SAMPLE_GT)
    assert target["store_name"] == "GS25 Gangnam"


def test_extract_total_price():
    target = extract_target_fields(SAMPLE_GT)
    assert target["total_price"] == "15000"


def test_extract_tax_price():
    target = extract_target_fields(SAMPLE_GT)
    assert target["tax_price"] == "1500"


def test_extract_items():
    target = extract_target_fields(SAMPLE_GT)
    assert len(target["items"]) == 2
    assert target["items"][0]["name"] == "Americano"
    assert target["items"][1]["price"] == "2500"


def test_extract_missing_fields_returns_defaults():
    target = extract_target_fields({"gt_parse": {}})
    assert target["store_name"] == ""
    assert target["total_price"] == ""
    assert target["tax_price"] is None
    assert target["items"] == []


def test_format_target_as_json_is_valid():
    import json
    target = extract_target_fields(SAMPLE_GT)
    json_str = format_target_as_json(target)
    parsed = json.loads(json_str)
    assert parsed["store_name"] == "GS25 Gangnam"
