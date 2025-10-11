"""Semantic and fragment cache implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import CacheConfig


@dataclass
class CacheItem:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._items: Dict[str, CacheItem] = {}

    def get(self, key: str) -> Optional[Any]:
        if not self._config.enabled:
            return None
        item = self._items.get(key)
        if not item:
            return None
        if item.expires_at < time.time():
            del self._items[key]
            return None
        return item.value

    def set(self, key: str, value: Any) -> None:
        if not self._config.enabled:
            return
        if len(self._items) >= self._config.max_entries:
            self._evict_one()
        self._items[key] = CacheItem(value=value, expires_at=time.time() + self._config.ttl_seconds)

    def _evict_one(self) -> None:
        if not self._items:
            return
        oldest_key = min(self._items.items(), key=lambda item: item[1].expires_at)[0]
        del self._items[oldest_key]
