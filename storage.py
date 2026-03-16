"""
storage.py - Persistence layer for Reinforcement Learning Playground
===================================================================

Provides:
    - JSON serialization for RL training runs
    - File-based storage for reproducibility
    - Session artifact management
"""

from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = DATA_DIR / "runs"


def ensure_data_dir() -> None:
    """Create local data directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filename: str):
    """Load JSON data from the local data directory.

    Parameters:
        filename (str): Name of the JSON file to load.

    Returns:
        JSON data or empty list if file does not exist.
    """
    ensure_data_dir()
    path = DATA_DIR / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(filename: str, data) -> None:
    """Save JSON data to the local data directory.

    Parameters:
        filename (str): Name of the JSON file to save.
        data: Data to serialize as JSON.
    """
    ensure_data_dir()
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_latest_session(session_data: dict) -> None:
    """Save the latest RL session to a JSON file.

    Parameters:
        session_data (dict): Complete session summary to persist.
    """
    ensure_data_dir()
    filename = "latest_rl_playground.json"
    path = RUNS_DIR / filename
    path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
