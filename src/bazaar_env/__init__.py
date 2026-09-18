"""Bazaar dealer-market environment."""

from __future__ import annotations


def load_environment(*args, **kwargs):
    from .env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)

__all__ = ["load_environment"]
