"""exporter.py — Export pipeline annotations to multiple formats.

Supports JSON (default), flat CSV, and COCO-style JSON.
"""
import csv
import json
from pathlib import Path
from typing import Any
