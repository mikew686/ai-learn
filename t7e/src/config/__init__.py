"""Shared config from environment. Defaults are for local (localhost, postgres)."""

from config.settings import load_config

__all__ = ["load_config"]
