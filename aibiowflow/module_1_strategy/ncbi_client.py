# -*- coding: utf-8 -*-
"""
NCBI E-utilities Client.

This module provides a client to interact with NCBI E-utilities APIs,
specifically for searching GEO datasets (ESearch) and retrieving their
summaries (ESummary).
"""
import requests
import time
import logging
from typing import List, Dict, Any, Optional

from .exceptions import NCBIAPIError

logger = logging.getLogger(__name__)

class NCBIClient:
    """
    NCBI E-utilities API 客户端。

    封装了对 NCBI API (如 ESearch, ESummary) 的调用，用于检索 GEO 数据集信息。
    包含重试逻辑和错误处理。
    """

    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_BACKOFF_FACTOR = 1.0  # Seconds
    DEFAULT_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    DEFAULT_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, api_key: Optional[str] = None, client_config: Optional[Dict[str, Any]] = None):
        """
        初始化 NCBIClient。

        :param api_key: NCBI API 密钥 (可选，但推荐用于更高速率限制)。
        :param client_config: 包含客户端配置的字典，例如:
            {
                "ncbi_esearch_url": "custom_esearch_url",
                "ncbi_esummary_url": "custom_esummary_url",
                "ncbi_retry_attempts": 3,
                "ncbi_backoff_factor": 1.5
            }
        """
        self.api_key = api_key
        config = client_config or {}

        self.esearch_url: str = config.get("ncbi_esearch_url", self.DEFAULT_ESEARCH_URL)
        self.esummary_url: str = config.get("ncbi_esummary_url", self.DEFAULT_ESUMMARY_URL)
        self.retry_attempts: int = int(config.get("ncbi_retry_attempts", self.DEFAULT_RETRY_ATTEMPTS))
        self.backoff_factor: float = float(config.get("ncbi_backoff_factor", self.DEFAULT_BACKOFF_FACTOR))

        logger.info(
            f"NCBIClient initialized. ESearch URL: {self.esearch_url}, ESummary URL: {self.esummary_url}, "
            f"Retries: {self.retry_attempts}, Backoff: {self.backoff_factor}s"
        )
        if not self.api_key:
            logger.warning("NCBI API key not provided. Requests may be subject to lower rate limits.")

    def _request_with_retry(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """
        执行带有重试逻辑的GET请求。

        :param url: 请求的URL。
        :param params: 请求的参数字典。
        :return: requests.Response 对象。
        :raises NCBIAPIError: 如果所有重试尝试均失败。
        """
        if self.api_key:
            params["api_key"] = self.api_key

        last_exception: Optional[Exception] = None
        for attempt in range(self.retry_attempts + 1):
            try:
                # Log URL and params before making the request for better traceability
                # Be cautious about logging API keys if they are ever part of params directly (they are not in this implementation)
                log_params = {k: v for k, v in params.items() if k != "api_key"}
                logger.debug(f"NCBI API request attempt {attempt + 1}: URL='{url}', Params='{log_params}'")

                response = requests.get(url, params=params, timeout=30) # 30秒超时
                # Log URL from response object as it's the final one after redirects, if any
                logger.debug(f"NCBI API response received from URL: {response.url}")
                response.raise_for_status()  # 如果状态码是 4xx 或 5xx，则抛出 HTTPError
                logger.info(f"NCBI API request to {response.url} successful on attempt {attempt + 1}.")
                return response
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code
                error_text_snippet = e.response.text[:200] if e.response and e.response.text else "N/A"
                logger.warning(
                    f"NCBI API request failed with HTTP status {status_code} "
                    f"(URL: {e.response.url}, Attempt: {attempt + 1}/{self.retry_attempts + 1}). "
                    f"Error: {e}. Response snippet: {error_text_snippet}"
                )
                if status_code in [429, 500, 502, 503, 504] and attempt < self.retry_attempts: # Retry on these
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying NCBI API call in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else: # Non-retryable HTTP error or max retries reached
                    full_error_message = f"HTTP error {status_code} calling NCBI API (URL: {e.response.url}). Response: {e.response.text}"
                    logger.error(full_error_message)
                    raise NCBIAPIError(
                        message=full_error_message,
                        status_code=status_code,
                        original_exception=e
                    ) from e
            except requests.exceptions.RequestException as e: # Other network issues (Timeout, ConnectionError)
                last_exception = e
                logger.warning(
                    f"NCBI API request failed due to network issue "
                    f"(URL: {url}, Params: {log_params}, Attempt: {attempt + 1}/{self.retry_attempts + 1}). Error: {e}"
                )
                if attempt < self.retry_attempts:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying NCBI API call in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else: # Max retries for network issues reached
                    network_error_message = f"Network error calling NCBI API (URL: {url}). Error: {e}"
                    logger.error(network_error_message)
                    raise NCBIAPIError(
                        message=network_error_message,
                        original_exception=e
                    ) from e

        final_error_message = "All retry attempts failed for NCBI API."
        if last_exception:
            final_error_message += f" Last error: {type(last_exception).__name__} - {str(last_exception)}"
        # Log params carefully, avoiding sensitive data if any were present
        log_params_final = {k: v for k, v in params.items() if k != "api_key"}
        logger.critical(final_error_message + f" (URL: {url}, Params: {log_params_final})")
        raise NCBIAPIError(final_error_message, original_exception=last_exception if last_exception else None)


    def search_geo(self, search_query: str, max_results: int = 200) -> List[str]:
        """
        使用 ESearch API 搜索 NCBI GEO 数据库。

        :param search_query: 用于搜索的查询字符串 (例如 "Alzheimer's disease AND rna-seq AND homo sapiens")。
        :param max_results: 返回的最大结果数量 (NCBI 默认可能较低)。
        :return: 匹配的 GEO Series (GSE) ID 列表。
        :raises NCBIAPIError: 如果 API 调用失败或未能解析响应。
        """
        logger.info(f"Searching GEO with query: '{search_query}', max_results: {max_results}")
        params = {
            "db": "gds",
            "term": search_query,
            "retmode": "json",
            "retmax": str(max_results), # ESearch retmax
            "usehistory": "y" # 虽然未使用 history, 但通常是好习惯
        }
        try:
            response = self._request_with_retry(self.esearch_url, params)
            data = response.json()

            if "esearchresult" not in data or "idlist" not in data["esearchresult"]:
                logger.error(f"Unexpected ESearch response format for query '{search_query}'. Keys: {list(data.keys())}. Response snippet: {str(data)[:500]}")
                raise NCBIAPIError("ESearch response missing 'esearchresult' or 'idlist'.")

            gse_ids = data["esearchresult"]["idlist"]
            logger.info(f"ESearch for query '{search_query}' found {len(gse_ids)} GSE IDs.")
            logger.debug(f"Found GSE IDs (first 20): {gse_ids[:20]}{'...' if len(gse_ids) > 20 else ''}")
            return gse_ids
        except ValueError as e_json: # JSONDecodeError
            # response might not be defined if _request_with_retry failed before returning one
            resp_text_snippet = "N/A (response object not available)"
            if 'response' in locals() and hasattr(response, 'text'):
                resp_text_snippet = response.text[:500]
            logger.error(f"Failed to decode JSON from ESearch response for query '{search_query}'. Response text snippet: {resp_text_snippet}...", exc_info=True)
            raise NCBIAPIError(message=f"Invalid JSON response from ESearch: {e_json}", original_exception=e_json) from e_json
        except NCBIAPIError: # Re-raise if it's already our specific error
            raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during GEO search for query '{search_query}': {e}", exc_info=True)
            raise NCBIAPIError(message=f"Unexpected error in search_geo: {e}", original_exception=e) from e

    def get_geo_summaries(self, gse_ids: List[str]) -> List[Dict[str, Any]]:
        """
        使用 ESummary API 获取指定 GEO Series ID 列表的摘要信息。

        :param gse_ids: GEO Series ID 的列表。
        :return: 一个字典列表，每个字典包含一个 GSE ID 的摘要信息。
                 每个字典至少包含: 'gse_id', 'title', 'summary', 'species', 'sample_count', 'srp_id' (如果可用), 'submission_date'.
        :raises NCBIAPIError: 如果 API 调用失败或未能解析响应。
        """
        if not gse_ids:
            logger.warning("get_geo_summaries called with an empty list of GSE IDs.")
            return []

        # Log only a few IDs if the list is very long
        log_gse_ids_display_str = ", ".join(gse_ids[:5])
        if len(gse_ids) > 5:
            log_gse_ids_display_str += ", ..."
        logger.info(f"Fetching summaries for {len(gse_ids)} GSE IDs. IDs: [{log_gse_ids_display_str}]")

        ids_string = ",".join(gse_ids)
        params = {
            "db": "gds",
            "id": ids_string,
            "retmode": "json"
        }
        try:
            response = self._request_with_retry(self.esummary_url, params)
            data = response.json()

            if "result" not in data or not isinstance(data["result"], dict):
                logger.error(f"Unexpected ESummary response format for IDs [{log_gse_ids_display_str}] (missing 'result' dict). Keys: {list(data.keys())}. Response snippet: {str(data)[:500]}")
                raise NCBIAPIError("ESummary response missing 'result' dictionary.")

            summaries = []
            id_list_from_response = data["result"].get("uids", []) # These are string keys for the result dict
            if not id_list_from_response:
                logger.warning(f"ESummary response 'result' does not contain 'uids' or 'uids' is empty for input IDs: [{log_gse_ids_display_str}]")

            processed_gse_ids = set()
            for gse_id_key_in_result in id_list_from_response:
                if gse_id_key_in_result not in data["result"]:
                    logger.warning(f"GSE ID key '{gse_id_key_in_result}' was in 'uids' but no summary data found in ESummary response 'result'.")
                    continue

                entry = data["result"][gse_id_key_in_result]
                actual_gse_id = entry.get("accession", gse_id_key_in_result)

                if not isinstance(actual_gse_id, str) or not actual_gse_id.startswith("GSE"):
                    logger.warning(f"Skipping entry with non-GSE accession '{actual_gse_id}' (from key '{gse_id_key_in_result}'). Entry snippet: {str(entry)[:200]}")
                    continue

                if actual_gse_id in processed_gse_ids:
                    logger.warning(f"Duplicate GSE ID '{actual_gse_id}' encountered in ESummary response. Skipping subsequent entry.")
                    continue
                processed_gse_ids.add(actual_gse_id)

                title = entry.get("title", "N/A")
                summary_text = entry.get("summary", "N/A")

                species_info = entry.get("taxon", "N/A")
                species_list: List[str] = []
                if isinstance(species_info, str) and species_info != "N/A":
                    species_list = [s.strip() for s in species_info.split(';') if s.strip()]
                elif isinstance(species_info, list):
                    species_list = [str(s) for s in species_info if isinstance(s, str) and s.strip()]

                sample_count_str = entry.get("n_samples", "0")
                sample_count = 0
                try:
                    sample_count = int(sample_count_str)
                except ValueError:
                    logger.warning(f"Could not parse sample_count '{sample_count_str}' for {actual_gse_id}. Defaulting to 0.")

                srp_id: Optional[str] = None
                ext_relations = entry.get("extrelations", [])
                if isinstance(ext_relations, list):
                    for relation_group in ext_relations:
                        if isinstance(relation_group, dict) and relation_group.get("relationtype") == "SRA":
                            relations = relation_group.get("relations", [])
                            if isinstance(relations, list):
                                for rel in relations:
                                    if isinstance(rel, dict) and isinstance(rel.get("targetobject"), str) and rel.get("targetobject", "").startswith("SRP"):
                                        srp_id = rel.get("targetobject")
                                        break
                            if srp_id:
                                break

                summaries.append({
                    "gse_id": actual_gse_id,
                    "title": title,
                    "summary": summary_text,
                    "species": species_list,
                    "sample_count": sample_count,
                    "submission_date": entry.get("pdat", "N/A"),
                    "srp_id": srp_id
                })
            logger.info(f"Successfully fetched and parsed summaries for {len(summaries)} unique GSE IDs from {len(gse_ids)} input IDs.")
            return summaries

        except ValueError as e_json: # JSONDecodeError
            resp_text_snippet = "N/A (response object not available)"
            if 'response' in locals() and hasattr(response, 'text'):
                resp_text_snippet = response.text[:500]
            logger.error(f"Failed to decode JSON from ESummary response for IDs [{log_gse_ids_display_str}]. Response text snippet: {resp_text_snippet}...", exc_info=True)
            raise NCBIAPIError(message=f"Invalid JSON response from ESummary: {e_json}", original_exception=e_json) from e_json
        except NCBIAPIError: # Re-raise
            raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during GEO summary retrieval for IDs [{log_gse_ids_display_str}]: {e}", exc_info=True)
            raise NCBIAPIError(message=f"Unexpected error in get_geo_summaries: {e}", original_exception=e) from e

# Example Usage (for standalone testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # NCBI_API_KEY = os.getenv("NCBI_API_KEY") # Load your NCBI API key from env for testing
    NCBI_API_KEY = None # Or test without one for lower rate limits

    client_config_example = {
        "ncbi_retry_attempts": 2,
        "ncbi_backoff_factor": 0.5
    }
    ncbi_client = NCBIClient(api_key=NCBI_API_KEY, client_config=client_config_example)

    test_query = "rna-seq AND alzheimer disease AND homo sapiens"
    print(f"\n--- Testing search_geo with query: '{test_query}' ---")
    try:
        gse_ids_found = ncbi_client.search_geo(test_query, max_results=5)
        if gse_ids_found:
            print(f"Found {len(gse_ids_found)} GSE IDs: {gse_ids_found}")

            print(f"\n--- Testing get_geo_summaries for IDs: {gse_ids_found} ---")
            summaries_data = ncbi_client.get_geo_summaries(gse_ids_found)
            if summaries_data:
                print(f"Retrieved {len(summaries_data)} summaries.")
                for i, summary_item in enumerate(summaries_data):
                    print(f"\nSummary for {summary_item.get('gse_id', 'N/A')}:")
                    print(f"  Title: {summary_item.get('title', 'N/A')[:80]}...")
                    print(f"  Species: {summary_item.get('species', [])}")
                    print(f"  Samples: {summary_item.get('sample_count', 0)}")
                    print(f"  SRP ID: {summary_item.get('srp_id', 'N/A')}")
                    # print(f"  Abstract: {summary_item.get('summary', 'N/A')[:100]}...")
                    if i >= 2 and len(summaries_data) > 3: # Print first 3
                        print("...")
                        break
            else:
                print("No summaries retrieved.")
        else:
            print("No GSE IDs found for the test query.")

    except NCBIAPIError as e:
        print(f"NCBI Client Error: {e}")
    except Exception as e_main:
        print(f"An unexpected error occurred in example usage: {e_main}")

```
