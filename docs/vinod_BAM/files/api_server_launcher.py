"""Thin launch wrapper for the BAM API server.

This keeps `api_server.py` unchanged while exposing a no-argument factory
that works with `uvicorn --factory`.
"""

from __future__ import annotations

from api_server import create_app as _create_app
from scoring_engine import BAMScoringEngine


def create_app():
    return _create_app(BAMScoringEngine())
