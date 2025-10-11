import time
import unittest

from ptis.cache import TTLCache
from ptis.config import CacheConfig
from ptis.reasoning.types import BranchState
from ptis.state_store import StateStore
from ptis.adapters.qwen import ChatMessage


class StateAndCacheTests(unittest.TestCase):
    def test_state_store_records_trace(self) -> None:
        store = StateStore()
        trace_id = store.new_trace()
        branch = BranchState(branch_id=0, temperature=0.5, messages=[ChatMessage("user", "hello")])
        store.record_branches(trace_id, [branch])
        record = store.get_trace(trace_id)
        self.assertIsNotNone(record)
        self.assertEqual(len(record.branches), 1)

    def test_ttl_cache_expiry(self) -> None:
        cache = TTLCache(CacheConfig(ttl_seconds=1, max_entries=2))
        cache.set("key", "value")
        self.assertEqual(cache.get("key"), "value")
        time.sleep(1.1)
        self.assertIsNone(cache.get("key"))


if __name__ == "__main__":
    unittest.main()
