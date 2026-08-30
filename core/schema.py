"""Frozen event schema and validation helpers."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "event_id": "string",
    "timestamp": "datetime64[ns]",
    "attack_id": "string",
    "label": "int8",
    "round": "int8",
    "customer_id": "string",
    "customer_account_age_days": "int32",
    "device_id": "string",
    "ip_asn": "string",
    "session_id": "string",
    "session_duration_s": "int32",
    "channel": "string",
    "card_id": "string",
    "card_bin": "string",
    "vpa": "string",
    "token_id": "string",
    "counterparty_id": "string",
    "counterparty_type": "string",
    "counterparty_mcc": "string",
    "counterparty_account_age_days": "int32",
    "amount": "float64",
    "is_first_time_payee": "int8",
    "auth_result": "string",
    "decline_code": "string",
    "meta": "object",
}

ALLOWED_ATTACK_IDS = {"LEGIT", "A1", "A2", "A3", "A4"}
ALLOWED_CHANNELS = {"CNP", "UPI_P2P", "UPI_P2M", "AGENTIC", "CARD_PRESENT"}
ALLOWED_AUTH_RESULTS = {"APPROVED", "DECLINED"}


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every required column cast to its contract dtype."""
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = df.copy()
    for column, dtype in REQUIRED_COLUMNS.items():
        if dtype == "datetime64[ns]":
            result[column] = pd.to_datetime(result[column]).astype(dtype)
        elif dtype == "object":
            result[column] = result[column].astype(object)
        else:
            result[column] = result[column].astype(dtype)
    return result


def validate(df: pd.DataFrame) -> None:
    """Raise when required columns, dtypes, or shared categorical values are invalid."""
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    wrong = {
        col: (str(df[col].dtype), expected)
        for col, expected in REQUIRED_COLUMNS.items()
        if str(df[col].dtype) != expected
    }
    if wrong:
        raise TypeError(f"Incorrect dtypes: {wrong}")
    if not set(df["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 or 1")
    if not set(df["attack_id"].dropna()).issubset(ALLOWED_ATTACK_IDS):
        raise ValueError("attack_id contains an unsupported value")
    if not set(df["channel"].dropna()).issubset(ALLOWED_CHANNELS):
        raise ValueError("channel contains an unsupported value")
    if not set(df["auth_result"].dropna()).issubset(ALLOWED_AUTH_RESULTS):
        raise ValueError("auth_result contains an unsupported value")
    if (df["amount"] < 0).any():
        raise ValueError("amount cannot be negative")
    invalid_meta = ~df["meta"].map(lambda value: isinstance(value, dict))
    if invalid_meta.any():
        raise TypeError("every meta value must be a dict")

