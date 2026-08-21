from pathlib import Path
import logging
import json

_config_cache = None
CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration from a JSON file"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
            return _config_cache
    except FileNotFoundError:
        logging.error("Config file %s not found.", config_path)
        return {}
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in %s: %s", config_path, e)
        return {}
    except Exception as e:
        logging.error("Error loading config file %s: %s", config_path, e)
        return {}

def get_search_config(config: dict = None) -> dict:
    """Retrieve search terms from the configuration"""
    if config is None:
        config = load_config()
    return config.get("search", {})

def get_vector_search_config(config: dict = None) -> dict:
    """Retrieve vector search settings from the configuration"""
    if config is None:
        config = load_config()
    return config.get("vector_search", {})

def get_all_stopwords(config: dict = None) -> set[str]:
    """Retrieve all stopwords from all languages as one combined set."""
    if config is None:
        config = load_config()
    stopwords_dict = config.get("stopwords", {})
    combined = set()
    for words in stopwords_dict.values():
        combined.update(words)
    return combined