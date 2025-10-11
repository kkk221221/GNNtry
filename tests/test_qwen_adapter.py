import json
import unittest
import urllib.error
from unittest import mock

from ptis.adapters.qwen import ChatCompletionRequest, ChatMessage, QwenAdapter
from ptis.config import QwenProviderConfig


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class QwenAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QwenProviderConfig(api_key="key", max_retries=2, timeout_s=1)
        self.adapter = QwenAdapter(self.config)

    @mock.patch("urllib.request.urlopen")
    def test_successful_request(self, mock_urlopen):
        mock_urlopen.return_value = DummyResponse({"id": "123", "choices": []})
        request = ChatCompletionRequest(
            model="qwen-plus",
            messages=[ChatMessage(role="user", content="hi")],
        )
        result = self.adapter.chat(request)
        self.assertEqual(result["id"], "123")
        mock_urlopen.assert_called_once()

    @mock.patch("time.sleep", return_value=None)
    @mock.patch("urllib.request.urlopen")
    def test_retry_on_url_error(self, mock_urlopen, _sleep):
        error = urllib.error.URLError("network")
        mock_urlopen.side_effect = [error, DummyResponse({"id": "ok", "choices": []})]
        request = ChatCompletionRequest(
            model="qwen-plus",
            messages=[ChatMessage(role="user", content="hi")],
        )
        result = self.adapter.chat(request)
        self.assertEqual(result["id"], "ok")
        self.assertEqual(mock_urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
