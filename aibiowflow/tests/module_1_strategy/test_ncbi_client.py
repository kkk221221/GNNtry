# -*- coding: utf-8 -*-
"""
Unit tests for the NCBIClient in Module 1.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import requests # To access requests.exceptions
import json

from aibiowflow.module_1_strategy.ncbi_client import NCBIClient
from aibiowflow.module_1_strategy.exceptions import NCBIAPIError

class TestNCBIClient(unittest.TestCase):
    """
    Test suite for the NCBIClient.
    """

    def setUp(self):
        """Set up common resources for tests."""
        self.api_key = "test_api_key"
        self.default_config = {
            "ncbi_esearch_url": "https://test.ncbi.com/esearch",
            "ncbi_esummary_url": "https://test.ncbi.com/esummary",
            "ncbi_retry_attempts": 2,
            "ncbi_backoff_factor": 0.1 # seconds for faster tests
        }
        self.client = NCBIClient(api_key=self.api_key, client_config=self.default_config)

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_search_geo_success(self, mock_requests_get):
        """Test successful GEO search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "idlist": ["123", "456"]
            }
        }
        mock_response.url = self.default_config["ncbi_esearch_url"] # Mock the final URL
        mock_requests_get.return_value = mock_response

        gse_ids = self.client.search_geo("test query", max_results=10)

        self.assertEqual(gse_ids, ["123", "456"])
        expected_params = {
            "db": "gds",
            "term": "test query",
            "retmode": "json",
            "retmax": "10",
            "usehistory": "y",
            "api_key": self.api_key
        }
        mock_requests_get.assert_called_once_with(
            self.default_config["ncbi_esearch_url"],
            params=expected_params,
            timeout=30
        )
        mock_response.raise_for_status.assert_called_once()

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_search_geo_no_results(self, mock_requests_get):
        """Test GEO search returning no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "idlist": []
            }
        }
        mock_response.url = self.default_config["ncbi_esearch_url"]
        mock_requests_get.return_value = mock_response

        gse_ids = self.client.search_geo("empty query")
        self.assertEqual(gse_ids, [])

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_search_geo_malformed_response_missing_keys(self, mock_requests_get):
        """Test GEO search with malformed JSON response (missing keys)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"wrongkey": "data"} # Missing esearchresult
        mock_response.url = self.default_config["ncbi_esearch_url"]
        mock_requests_get.return_value = mock_response

        with self.assertRaisesRegex(NCBIAPIError, "ESearch response missing 'esearchresult' or 'idlist'"):
            self.client.search_geo("test query")

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_search_geo_invalid_json_response(self, mock_requests_get):
        """Test GEO search with response that is not valid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error decoding", "doc", 0)
        mock_response.text = "This is not JSON"
        mock_response.url = self.default_config["ncbi_esearch_url"]
        mock_requests_get.return_value = mock_response

        with self.assertRaisesRegex(NCBIAPIError, "Invalid JSON response from ESearch"):
            self.client.search_geo("test query")

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_success(self, mock_requests_get):
        """Test successful retrieval of GEO summaries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "uids": ["123", "456"],
                "123": {
                    "accession": "GSE123",
                    "title": "Title 1",
                    "summary": "Summary 1",
                    "taxon": "Homo sapiens",
                    "n_samples": "10",
                    "pdat": "2023/01/01",
                    "extrelations": [{
                        "relationtype": "SRA",
                        "relations": [{"targetobject": "SRP111"}]
                    }]
                },
                "456": {
                    "accession": "GSE456",
                    "title": "Title 2",
                    "summary": "Summary 2",
                    "taxon": "Mus musculus; Rattus norvegicus", # Multiple species
                    "n_samples": "5",
                    "pdat": "2023/01/02",
                    # No SRA relation
                }
            }
        }
        mock_response.url = self.default_config["ncbi_esummary_url"]
        mock_requests_get.return_value = mock_response

        summaries = self.client.get_geo_summaries(["123", "456"])

        expected_summaries = [
            {
                "gse_id": "GSE123", "title": "Title 1", "summary": "Summary 1",
                "species": ["Homo sapiens"], "sample_count": 10,
                "submission_date": "2023/01/01", "srp_id": "SRP111"
            },
            {
                "gse_id": "GSE456", "title": "Title 2", "summary": "Summary 2",
                "species": ["Mus musculus", "Rattus norvegicus"], "sample_count": 5,
                "submission_date": "2023/01/02", "srp_id": None
            }
        ]
        self.assertEqual(summaries, expected_summaries)
        expected_params = {
            "db": "gds",
            "id": "123,456",
            "retmode": "json",
            "api_key": self.api_key
        }
        mock_requests_get.assert_called_once_with(
            self.default_config["ncbi_esummary_url"],
            params=expected_params,
            timeout=30
        )
        mock_response.raise_for_status.assert_called_once()

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_empty_input(self, mock_requests_get):
        """Test get_geo_summaries with an empty list of GSE IDs."""
        summaries = self.client.get_geo_summaries([])
        self.assertEqual(summaries, [])
        mock_requests_get.assert_not_called()

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_malformed_response_no_result_key(self, mock_requests_get):
        """Test ESummary with response missing 'result' key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "some error"} # Missing 'result'
        mock_response.url = self.default_config["ncbi_esummary_url"]
        mock_requests_get.return_value = mock_response

        with self.assertRaisesRegex(NCBIAPIError, "ESummary response missing 'result' dictionary"):
            self.client.get_geo_summaries(["123"])

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_entry_details_parsing(self, mock_requests_get):
        """Test parsing of various fields in ESummary entries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "uids": ["789"],
                "789": {
                    "accession": "GSE789", "title": "Test Title", "summary": "Test Summary",
                    "taxon": "Pan troglodytes", # Single species
                    "n_samples": "invalid_count", # Invalid sample count
                    "pdat": "N/A",
                    "extrelations": [ # SRA ID in a different part of relations
                        {"relationtype": "Other"},
                        {"relationtype": "SRA", "relations": [
                            {"targetobject": "OTHER001"},
                            {"targetobject": "SRP333"}
                        ]}
                    ]
                }
            }
        }
        mock_response.url = self.default_config["ncbi_esummary_url"]
        mock_requests_get.return_value = mock_response

        summaries = self.client.get_geo_summaries(["789"])
        expected_summary = {
            "gse_id": "GSE789", "title": "Test Title", "summary": "Test Summary",
            "species": ["Pan troglodytes"], "sample_count": 0, # Default on parse error
            "submission_date": "N/A", "srp_id": "SRP333"
        }
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0], expected_summary)

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_request_with_retry_http_error_non_retryable(self, mock_requests_get):
        """Test non-retryable HTTP error (e.g., 400 Bad Request)."""
        mock_response = MagicMock(spec=requests.Response) # Use spec for requests.Response
        mock_response.status_code = 400
        mock_response.text = "Bad Request Details"
        mock_response.url = "https://test.ncbi.com/some_service" # Mock the final URL
        # Configure raise_for_status to raise an HTTPError
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Client Error", response=mock_response
        )
        mock_requests_get.return_value = mock_response

        with self.assertRaisesRegex(NCBIAPIError, "HTTP error 400 calling NCBI API") as cm:
            self.client._request_with_retry("https://test.ncbi.com/some_service", {"param": "value"})
        self.assertEqual(cm.exception.status_code, 400)
        # Ensure it only tried once for a 400 error
        self.assertEqual(mock_requests_get.call_count, 1)


    @patch('aibiowflow.module_1_strategy.ncbi_client.time.sleep') # Mock time.sleep
    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_request_with_retry_http_error_retryable_then_success(self, mock_requests_get, mock_sleep):
        """Test retryable HTTP error (e.g., 503) then success."""
        mock_response_fail1 = MagicMock(spec=requests.Response)
        mock_response_fail1.status_code = 503
        mock_response_fail1.text = "Service Unavailable"
        mock_response_fail1.url = "https://test.ncbi.com/retry_service"
        mock_response_fail1.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Server Error", response=mock_response_fail1
        )

        mock_response_success = MagicMock(spec=requests.Response)
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "ok"}
        mock_response_success.url = "https://test.ncbi.com/retry_service"
        # mock_response_success.raise_for_status = MagicMock() # Does not raise

        mock_requests_get.side_effect = [mock_response_fail1, mock_response_success]

        response = self.client._request_with_retry("https://test.ncbi.com/retry_service", {"param": "value"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_requests_get.call_count, 2) # Original + 1 retry
        mock_sleep.assert_called_once_with(self.default_config["ncbi_backoff_factor"] * (2**0)) # First backoff

    @patch('aibiowflow.module_1_strategy.ncbi_client.time.sleep')
    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_request_with_retry_all_attempts_fail_http(self, mock_requests_get, mock_sleep):
        """Test when all retry attempts fail with HTTP errors."""
        mock_response_fail = MagicMock(spec=requests.Response)
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Internal Server Error"
        mock_response_fail.url = "https://test.ncbi.com/failing_service"
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_response_fail
        )

        # All attempts will return this failing response
        mock_requests_get.return_value = mock_response_fail

        with self.assertRaisesRegex(NCBIAPIError, "HTTP error 500 calling NCBI API") as cm:
            self.client._request_with_retry("https://test.ncbi.com/failing_service", {"param": "value"})

        self.assertEqual(cm.exception.status_code, 500)
        # Total calls = 1 (original) + configured retry_attempts
        self.assertEqual(mock_requests_get.call_count, self.default_config["ncbi_retry_attempts"] + 1)
        # Check sleep calls
        expected_sleep_calls = [
            call(self.default_config["ncbi_backoff_factor"] * (2**i))
            for i in range(self.default_config["ncbi_retry_attempts"])
        ]
        mock_sleep.assert_has_calls(expected_sleep_calls)


    @patch('aibiowflow.module_1_strategy.ncbi_client.time.sleep')
    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_request_with_retry_network_error_then_success(self, mock_requests_get, mock_sleep):
        """Test network error (e.g., Timeout) then success."""
        mock_requests_get.side_effect = [
            requests.exceptions.Timeout("Connection timed out"),
            MagicMock(status_code=200, json=lambda: {"status": "ok"}, url="https://service_url")
        ]

        response = self.client._request_with_retry("https://service_url", {"p": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_requests_get.call_count, 2)
        mock_sleep.assert_called_once_with(self.default_config["ncbi_backoff_factor"] * (2**0))

    @patch('aibiowflow.module_1_strategy.ncbi_client.time.sleep')
    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_request_with_retry_all_attempts_fail_network(self, mock_requests_get, mock_sleep):
        """Test when all retry attempts fail with network errors."""
        mock_requests_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")

        with self.assertRaisesRegex(NCBIAPIError, "Network error calling NCBI API") as cm:
            self.client._request_with_retry("https://failing_network_service", {"p": "1"})

        self.assertIsInstance(cm.exception.original_exception, requests.exceptions.ConnectionError)
        self.assertEqual(mock_requests_get.call_count, self.default_config["ncbi_retry_attempts"] + 1)
        expected_sleep_calls = [
            call(self.default_config["ncbi_backoff_factor"] * (2**i))
            for i in range(self.default_config["ncbi_retry_attempts"])
        ]
        mock_sleep.assert_has_calls(expected_sleep_calls)

    def test_client_initialization_no_api_key(self):
        """Test client initialization without an API key."""
        with self.assertLogs('aibiowflow.module_1_strategy.ncbi_client', level='WARNING') as log_cm:
            client_no_key = NCBIClient(api_key=None, client_config=self.default_config)
        self.assertIsNotNone(client_no_key)
        self.assertIn("NCBI API key not provided", log_cm.output[0]) # Check first relevant log

    def test_client_initialization_default_config(self):
        """Test client initialization uses default URLs and settings if not provided."""
        client_default = NCBIClient(api_key=self.api_key) # No client_config
        self.assertEqual(client_default.esearch_url, NCBIClient.DEFAULT_ESEARCH_URL)
        self.assertEqual(client_default.esummary_url, NCBIClient.DEFAULT_ESUMMARY_URL)
        self.assertEqual(client_default.retry_attempts, NCBIClient.DEFAULT_RETRY_ATTEMPTS)
        self.assertEqual(client_default.backoff_factor, NCBIClient.DEFAULT_BACKOFF_FACTOR)

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_srp_id_parsing_variations(self, mock_requests_get):
        """Test various ways SRP ID might be (or not be) present."""
        base_entry = {
            "accession": "GSE001", "title": "T", "summary": "S", "taxon": "H.sapiens",
            "n_samples": "1", "pdat": "2023"
        }

        # Case 1: Standard SRA relation
        entry1 = {**base_entry, "accession": "GSE001", "extrelations": [{"relationtype": "SRA", "relations": [{"targetobject": "SRP001"}]}]}
        # Case 2: No SRA relation type
        entry2 = {**base_entry, "accession": "GSE002", "extrelations": [{"relationtype": "Other", "relations": [{"targetobject": "XXX001"}]}]}
        # Case 3: SRA relation type, but no targetobject or wrong type
        entry3 = {**base_entry, "accession": "GSE003", "extrelations": [{"relationtype": "SRA", "relations": [{"target_typo": "SRP003"}]}]}
        # Case 4: No extrelations at all
        entry4 = {**base_entry, "accession": "GSE004"}
        # Case 5: extrelations is not a list
        entry5 = {**base_entry, "accession": "GSE005", "extrelations": "not_a_list"}
        # Case 6: relations is not a list
        entry6 = {**base_entry, "accession": "GSE006", "extrelations": [{"relationtype": "SRA", "relations": "not_a_list_either"}]}
         # Case 7: relation item in relations is not a dict
        entry7 = {**base_entry, "accession": "GSE007", "extrelations": [{"relationtype": "SRA", "relations": ["not_a_dict"]}]}


        mock_response_json = {
            "result": {
                "uids": ["1", "2", "3", "4", "5", "6", "7"],
                "1": entry1, "2": entry2, "3": entry3, "4": entry4, "5": entry5, "6": entry6, "7": entry7,
            }
        }
        mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: mock_response_json, url="url")

        summaries = self.client.get_geo_summaries(["1", "2", "3", "4", "5", "6", "7"])

        self.assertEqual(summaries[0]["srp_id"], "SRP001")
        self.assertIsNone(summaries[1]["srp_id"])
        self.assertIsNone(summaries[2]["srp_id"])
        self.assertIsNone(summaries[3]["srp_id"])
        self.assertIsNone(summaries[4]["srp_id"])
        self.assertIsNone(summaries[5]["srp_id"])
        self.assertIsNone(summaries[6]["srp_id"])

    @patch('aibiowflow.module_1_strategy.ncbi_client.requests.get')
    def test_get_geo_summaries_species_parsing_variations(self, mock_requests_get):
        """Test parsing of species field."""
        base_entry = {"accession": "GSE001", "title": "T", "summary": "S", "n_samples": "1", "pdat": "2023"}

        entry1 = {**base_entry, "accession": "GSE001", "taxon": "Homo sapiens"}
        entry2 = {**base_entry, "accession": "GSE002", "taxon": "Mus musculus; Rattus norvegicus"}
        entry3 = {**base_entry, "accession": "GSE003", "taxon": ["Gallus gallus", "Anas platyrhynchos"]} # List of strings
        entry4 = {**base_entry, "accession": "GSE004", "taxon": "N/A"}
        entry5 = {**base_entry, "accession": "GSE005"} # Missing taxon field
        entry6 = {**base_entry, "accession": "GSE006", "taxon": ""} # Empty string
        entry7 = {**base_entry, "accession": "GSE007", "taxon": [123, "Bos taurus"]} # Mixed list

        mock_response_json = {"result": {
            "uids": ["1","2","3","4","5","6","7"],
            "1": entry1, "2": entry2, "3": entry3, "4": entry4, "5": entry5, "6": entry6, "7": entry7,
        }}
        mock_requests_get.return_value = MagicMock(status_code=200, json=lambda: mock_response_json, url="url")
        summaries = self.client.get_geo_summaries(["1","2","3","4","5","6","7"])

        self.assertEqual(summaries[0]["species"], ["Homo sapiens"])
        self.assertEqual(summaries[1]["species"], ["Mus musculus", "Rattus norvegicus"])
        self.assertEqual(summaries[2]["species"], ["Gallus gallus", "Anas platyrhynchos"])
        self.assertEqual(summaries[3]["species"], []) # N/A results in empty list
        self.assertEqual(summaries[4]["species"], []) # Missing results in empty
        self.assertEqual(summaries[5]["species"], []) # Empty string results in empty
        self.assertEqual(summaries[6]["species"], ["123", "Bos taurus"]) # Converts non-strings to strings

if __name__ == '__main__':
    unittest.main()
