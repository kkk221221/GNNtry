# -*- coding: utf-8 -*-
"""
Service layer for Module 1: Literature & Strategy Service.

Contains the StrategyService class responsible for orchestrating the workflow
of interpreting user queries, fetching and filtering data, and generating
analysis proposals.
"""
import logging
import uuid
import json # Added for serializing candidate_list_json
import time # Added for caching TTL
import hashlib # Added for generating cache key if needed, though starting simple
from typing import Dict, List, Optional, Any, Tuple

# Attempt to import LLMGateway, handle if not available at this stage of generation
try:
    from aibiowflow.llm_gateway import LLMGateway
    from aibiowflow.llm_gateway.exceptions import LLMAPIError, LLMOutputValidationError, PromptTemplateError, ConfigurationError as LLMGatewayConfigurationError
except ImportError:
    LLMGateway = None # type: ignore
    LLMAPIError = Exception # type: ignore
    LLMOutputValidationError = Exception # type: ignore
    PromptTemplateError = Exception # type: ignore
    LLMGatewayConfigurationError = Exception # type: ignore
    logging.warning("LLMGateway or its exceptions could not be imported. StrategyService may not fully function.")


from .data_models import AnalysisProposal, QueryIntent, DatasetCandidate
from .exceptions import StrategyCreationError, NCBIAPIError, NoDataFoundError
from .ncbi_client import NCBIClient


class StrategyService:
    """
    文献与数据策略服务 (Literature & Strategy Service)。

    该服务负责将用户的自然语言科研问题，通过一系列处理步骤
    （意图解析、文献/数据检索、启发式过滤、LLM辅助分析），
    转化为一个结构化的、包含推荐数据集的分析方案 (`AnalysisProposal`)。
    """

    def __init__(self, module_config: Dict[str, Any], llm_gateway: Optional[LLMGateway], ncbi_api_key: Optional[str] = None):
        """
        初始化文献与数据策略服务。

        :param module_config: 包含此模块特定配置的字典 (通常来自 module_1_config.yaml)。
                              例如: heuristic_filtering规则, ncbi_client_settings, cache_ttl_seconds。
        :param llm_gateway: LLM网关的已初始化实例。如果为None，涉及LLM调用的功能将受限。
        :param ncbi_api_key: 用于NCBI E-utilities的API密钥 (可选)。
        """
        self.logger = logging.getLogger(__name__)
        self.module_config = module_config

        if llm_gateway is None and LLMGateway is not None : # Check if LLMGateway was expected but not provided
             self.logger.warning("LLMGateway instance not provided to StrategyService. LLM-dependent features will fail.")
        self.llm_gateway = llm_gateway

        # Extract NCBI client specific config from module_config
        ncbi_client_settings: Dict[str, Any] = self.module_config.get('ncbi_client_settings', {})
        # Fallback for older config structure if ncbi_client_settings is not present
        if not ncbi_client_settings:
             ncbi_client_settings = {
                "ncbi_esearch_url": self.module_config.get("ncbi_esearch_url"),
                "ncbi_esummary_url": self.module_config.get("ncbi_esummary_url"),
                "ncbi_retry_attempts": self.module_config.get("ncbi_retry_attempts"),
                "ncbi_backoff_factor": self.module_config.get("ncbi_backoff_factor"),
             }
             # Filter out None values to allow NCBIClient defaults to take over
             ncbi_client_settings = {k: v for k, v in ncbi_client_settings.items() if v is not None}


        self.ncbi_client = NCBIClient(api_key=ncbi_api_key, client_config=ncbi_client_settings)

        # Load heuristic filtering rules
        self.heuristic_rules: Dict[str, Any] = self.module_config.get('heuristic_filtering', {})
        self.allowed_species: List[str] = self.heuristic_rules.get('allowed_species', ["Homo sapiens", "Mus musculus"])
        self.min_sample_count: int = self.heuristic_rules.get('min_sample_count', 6)
        self.require_srp_id: bool = self.heuristic_rules.get('require_srp_id', True)
        self.match_experiment_type_keywords: List[str] = self.heuristic_rules.get('match_experiment_type_keywords', [])

        self.logger.info(f"Heuristic filtering rules loaded: Species={self.allowed_species}, MinSamples={self.min_sample_count}, RequireSRP={self.require_srp_id}, Keywords={self.match_experiment_type_keywords}")

        # Load cache TTL
        self.cache_ttl_seconds: int = int(self.module_config.get('cache_ttl_seconds', 86400)) # Default 24 hours, ensure int
        # Initialize cache: Key: user_query (or its hash), Value: (expiry_timestamp, AnalysisProposal object)
        self.cache: Dict[str, Tuple[float, AnalysisProposal]] = {}

        self.logger.info("StrategyService initialized successfully.")
        if LLMGateway is None :
            self.logger.error("LLMGateway class was not imported. This instance of StrategyService will be non-functional for LLM tasks.")


    def _interpret_user_query(self, user_query: str) -> QueryIntent:
        """
        使用LLM解析用户查询，提取结构化意图。 (对应任务6.1)
        """
        self.logger.debug(f"Interpreting user query: {user_query[:100]}...")
        if not self.llm_gateway:
            self.logger.error("LLMGateway not available for query interpretation.")
            raise StrategyCreationError("LLMGateway not configured, cannot interpret user query.")

        # Placeholder for actual LLM call - to be implemented in Step 7 (detail) / Step 8 (plan)
        # For skeleton, return a dummy QueryIntent or raise NotImplementedError
        try:
            query_intent_obj = self.llm_gateway.get_structured_response(
                prompt_name="interpret_query", # from prompts_m1.toml
                context={"user_query": user_query},
                output_schema=QueryIntent
            )
            if not isinstance(query_intent_obj, QueryIntent):
                # This case should ideally be caught by Pydantic validation in LLMGateway
                # or the type checker if LLMGateway.get_structured_response is well-typed.
                self.logger.error(f"LLM call for 'interpret_query' returned unexpected type: {type(query_intent_obj)}")
                raise StrategyCreationError(f"LLM did not return a valid QueryIntent object. Type was: {type(query_intent_obj)}")
            return query_intent_obj
        except (LLMAPIError, LLMOutputValidationError, PromptTemplateError, LLMGatewayConfigurationError) as e:
            if isinstance(e, LLMOutputValidationError):
                # Spec Section 7.0 mentions a potential correction loop for LLMOutputValidationError.
                # Logging this specifically, though a correction loop is not implemented in this version.
                validation_details = getattr(e, 'validation_errors', str(e))
                self.logger.error(
                    f"LLM output validation error during query interpretation. "
                    f"Details: {validation_details}. (Correction loop not implemented).",
                    exc_info=True
                )
            else:
                self.logger.error(f"LLM Gateway error during query interpretation: {e}", exc_info=True)
            raise StrategyCreationError(f"Failed to interpret user query via LLM: {str(e)}", original_exception=e) from e
        except Exception as e: # Catch-all for other unexpected errors from the gateway call
            self.logger.error(f"Unexpected error during LLM call for query interpretation: {e}", exc_info=True)
            raise StrategyCreationError(f"Unexpected error interpreting user query: {str(e)}", original_exception=e) from e


    def _translate_intent_to_search_query(self, query_intent: QueryIntent) -> str:
        """
        将结构化的查询意图转换为针对NCBI GEO优化的搜索字符串。 (对应任务6.1)
        """
        self.logger.debug(f"Translating query intent to search query: {query_intent.model_dump_json(indent=2)}")
        if not self.llm_gateway:
            self.logger.error("LLMGateway not available for search query translation.")
            raise StrategyCreationError("LLMGateway not configured, cannot translate intent to search query.")

        try:
            geo_search_string = self.llm_gateway.get_text_response(
                prompt_name="translate_to_search", # from prompts_m1.toml
                context={"intent_json": query_intent.model_dump_json()}
            )
            if not geo_search_string or not isinstance(geo_search_string, str):
                self.logger.error(f"LLM call for 'translate_to_search' returned invalid/empty string: '{geo_search_string}'")
                raise StrategyCreationError("LLM did not return a valid search string.")
            return geo_search_string.strip() # Ensure no leading/trailing whitespace
        except (LLMAPIError, PromptTemplateError, LLMGatewayConfigurationError) as e:
            self.logger.error(f"LLM Gateway error during search query translation: {e}", exc_info=True)
            raise StrategyCreationError(f"Failed to translate intent to search query via LLM: {str(e)}", original_exception=e) from e
        except Exception as e: # Catch-all for other unexpected errors
            self.logger.error(f"Unexpected error during LLM call for search query translation: {e}", exc_info=True)
            raise StrategyCreationError(f"Unexpected error translating intent to search query: {str(e)}", original_exception=e) from e


    def _filter_candidates(self, gse_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对从NCBI获取的原始GSE元数据列表应用启发式过滤规则。 (对应任务6.2 Part 2)
        """
        self.logger.debug(f"Applying heuristic filters to {len(gse_metadata_list)} candidates.")
        filtered_list: List[Dict[str, Any]] = []

        for item in gse_metadata_list:
            gse_id = item.get("gse_id", "UnknownGSE")
            passes_filter = True
            filter_reasons: List[str] = []

            # Rule 1: Allowed Species
            item_species = item.get("species", [])
            if not isinstance(item_species, list): # Ensure it's a list
                item_species = [str(item_species)] if item_species else []

            # Convert both lists to lowercase sets for case-insensitive intersection check
            allowed_species_set = {s.lower() for s in self.allowed_species}
            item_species_set = {s.lower() for s in item_species}

            if not allowed_species_set.intersection(item_species_set):
                passes_filter = False
                filter_reasons.append(f"Species not in allowed list ({self.allowed_species}). Found: {item_species}")

            # Rule 2: Minimum Sample Count
            sample_count = item.get("sample_count", 0)
            if not isinstance(sample_count, int): # Ensure it's an int
                try:
                    sample_count = int(sample_count)
                except ValueError:
                    sample_count = 0 # Default if parsing fails

            if sample_count < self.min_sample_count:
                passes_filter = False
                filter_reasons.append(f"Sample count {sample_count} is less than minimum {self.min_sample_count}")

            # Rule 3: Require SRP ID
            srp_id = item.get("srp_id")
            if self.require_srp_id and not srp_id:
                passes_filter = False
                filter_reasons.append("Required SRP ID is missing")

            # Rule 4: Match Experiment Type Keywords (if keywords are defined)
            if self.match_experiment_type_keywords:
                text_to_search = (item.get("title", "") + " " + item.get("summary", "")).lower()
                found_keyword = False
                for keyword in self.match_experiment_type_keywords:
                    if keyword.lower() in text_to_search:
                        found_keyword = True
                        break
                if not found_keyword:
                    passes_filter = False
                    filter_reasons.append(f"Did not match any experiment type keywords: {self.match_experiment_type_keywords}")

            if passes_filter:
                filtered_list.append(item)
            else:
                self.logger.info(f"GSE {gse_id} filtered out. Reasons: {'; '.join(filter_reasons)}")

        self.logger.info(f"Heuristic filtering complete. {len(filtered_list)} out of {len(gse_metadata_list)} candidates passed.")
        return filtered_list


    def _analyze_candidates_with_llm(self, user_query: str, candidates_metadata: List[Dict[str, Any]]) -> List[DatasetCandidate]:
        """
        使用LLM对筛选后的候选数据集进行深度分析和排序。 (对应任务6.3)
        """
        self.logger.debug(f"Performing LLM-based deep analysis on {len(candidates_metadata)} candidates.")
        if not self.llm_gateway:
            self.logger.error("LLMGateway not available for candidate analysis.")
            raise StrategyCreationError("LLMGateway not configured, cannot analyze candidates.")

        # Placeholder for actual LLM call - to be implemented in Step 10 (plan)
        # ranked_candidates = self.llm_gateway.get_structured_response(
        # prompt_name="analyze_candidates", # from prompts_m1.toml
        #     context={
        # "user_query": user_query,
        # "candidate_list_json": json.dumps(candidates_metadata) # Or however it needs to be formatted
        #     },
        #     output_schema=List[DatasetCandidate]
        # )
        # return ranked_candidates
        if not candidates_metadata:
            self.logger.info("No candidates to analyze with LLM, returning empty list.")
            return []

        try:
            # Ensure candidates_metadata (list of dicts) is properly serialized to a JSON string for the prompt
            # The prompt expects a JSON string for 'candidate_list_json'
            # The elements of candidates_metadata are dicts from NCBIClient, not DatasetCandidate Pydantic models yet.
            # We only need specific fields for the LLM as per prompt:
            # gse_id, srp_id (optional), title, summary, species, sample_count.
            simplified_candidates_for_llm = [
                {
                    "gse_id": cand.get("gse_id"),
                    "srp_id": cand.get("srp_id"),
                    "title": cand.get("title"),
                    "summary": cand.get("summary"),
                    "species": cand.get("species"),
                    "sample_count": cand.get("sample_count")
                }
                for cand in candidates_metadata
            ]
            candidate_list_json_string = json.dumps(simplified_candidates_for_llm, ensure_ascii=False, indent=2)

            self.logger.debug(f"Sending to LLM for analysis - User Query: {user_query[:100]}..., Candidates JSON (first 200 chars): {candidate_list_json_string[:200]}...")

            ranked_candidate_list = self.llm_gateway.get_structured_response(
                prompt_name="analyze_candidates", # from prompts_m1.toml
                context={
                    "user_query": user_query,
                    "candidate_list_json": candidate_list_json_string
                },
                output_schema=List[DatasetCandidate] # Expecting a list of DatasetCandidate objects
            )

            if not isinstance(ranked_candidate_list, list) or \
               not all(isinstance(item, DatasetCandidate) for item in ranked_candidate_list):
                self.logger.error(f"LLM call for 'analyze_candidates' returned unexpected type or list content: {type(ranked_candidate_list)}")
                # Attempt to see if it's a single item that should have been a list
                if isinstance(ranked_candidate_list, DatasetCandidate):
                     self.logger.info("LLM returned a single DatasetCandidate, wrapping in a list.")
                     return [ranked_candidate_list] # type: ignore

                raise StrategyCreationError(
                    "LLM did not return a valid list of DatasetCandidate objects for candidate analysis."
                )

            self.logger.info(f"LLM analysis completed. Received {len(ranked_candidate_list)} ranked candidates.")
            return ranked_candidate_list

        except (LLMAPIError, LLMOutputValidationError, PromptTemplateError, LLMGatewayConfigurationError) as e:
            if isinstance(e, LLMOutputValidationError):
                validation_details = getattr(e, 'validation_errors', str(e))
                self.logger.error(
                    f"LLM output validation error during candidate analysis. "
                    f"Details: {validation_details}. (Correction loop not implemented).",
                    exc_info=True
                )
            else:
                self.logger.error(f"LLM Gateway error during candidate analysis: {e}", exc_info=True)
            raise StrategyCreationError(f"Failed to analyze candidates via LLM: {str(e)}", original_exception=e) from e
        except Exception as e: # Catch-all for other unexpected errors
            self.logger.error(f"Unexpected error during LLM candidate analysis: {e}", exc_info=True)
            raise StrategyCreationError(f"Unexpected error analyzing candidates: {str(e)}", original_exception=e) from e


    def create_proposal_from_query(self, user_query: str) -> AnalysisProposal:
        """
        模块主入口函数，执行完整的策略生成工作流。

        接收用户原始查询，经过意图解析、数据检索、过滤、LLM分析等步骤，
        最终生成一个包含推荐数据集的结构化分析方案。

        :param user_query: 用户的原始自然语言查询。
        :return: 一个 `AnalysisProposal` 对象，包含数据集建议。
        :raises NoDataFoundError: 如果在流程中未能找到合适的数据。
        :raises StrategyCreationError: 如果在处理过程中发生不可恢复的错误。
        """
        self.logger.info(f"Received request to create proposal for query: \"{user_query[:100]}...\"")

        try:
            # Step 0: Check cache
            # Using user_query directly as key for simplicity. Consider hashing for long/complex queries.
            cache_key = user_query
            if cache_key in self.cache:
                expiry_timestamp, cached_proposal = self.cache[cache_key]
                if time.time() < expiry_timestamp:
                    self.logger.info(f"Cache hit for query: \"{user_query[:100]}...\". Returning cached proposal {cached_proposal.proposal_id}.")
                    return cached_proposal
                else:
                    self.logger.info(f"Cached proposal for query \"{user_query[:100]}...\" found but expired. Recalculating.")
                    del self.cache[cache_key] # Remove expired entry
            else:
                self.logger.info(f"Cache miss for query: \"{user_query[:100]}...\". Proceeding with generation.")

            # Step 1: Interpret user query (LLM Call)
            self.logger.info("Step 1: Interpreting user query...")
            query_intent: QueryIntent = self._interpret_user_query(user_query)
            self.logger.info(f"Query interpreted: {query_intent.model_dump_json(indent=2)}") # Log full JSON

            # Step 2: Translate intent to search query (LLM Call)
            self.logger.info("Step 2: Translating intent to GEO search query...")
            geo_search_string: str = self._translate_intent_to_search_query(query_intent)
            self.logger.info(f"GEO search string generated: \"{geo_search_string}\"")

            # Step 3: Search GEO (NCBI Client Call) - Now Active
            self.logger.info(f"Step 3: Searching GEO with query: \"{geo_search_string}\"")
            gse_ids: List[str] = self.ncbi_client.search_geo(geo_search_string)
            if not gse_ids:
                self.logger.warning(f"No GEO Series IDs found for search query: {geo_search_string}")
                raise NoDataFoundError(f"No GEO Series IDs found for search: {geo_search_string}")
            self.logger.info(f"Found {len(gse_ids)} GSE IDs initially: {gse_ids[:10]}...") # Log first few

            # Step 4: Get GEO Summaries (NCBI Client Call) - Now Active
            self.logger.info(f"Step 4: Fetching summaries for {len(gse_ids)} GSE IDs.")
            # Consider chunking gse_ids if very large for get_geo_summaries to avoid overly long URLs
            # For now, passing all as per NCBIClient current implementation
            gse_metadata_list: List[Dict[str, Any]] = self.ncbi_client.get_geo_summaries(gse_ids)
            if not gse_metadata_list: # NCBIClient.get_geo_summaries returns [] if input is empty or all fail
                self.logger.warning(f"No metadata retrieved for {len(gse_ids)} GSE IDs ({gse_ids[:10]}...). This might be due to invalid IDs or API issues for all.")
                # Depending on strictness, this could be a NoDataFoundError or proceed if some IDs might be invalid.
                # For now, if gse_ids was populated but summaries are empty, it's an issue.
                if gse_ids: # only raise if we expected summaries
                     raise NoDataFoundError(f"No metadata found for the {len(gse_ids)} retrieved GSE IDs (e.g., {gse_ids[:10]}).")
            self.logger.info(f"Retrieved metadata for {len(gse_metadata_list)} GSEs.")

            # Step 5: Heuristic filtering - Now Active
            self.logger.info("Step 5: Applying heuristic filters...")
            filtered_gse_metadata: List[Dict[str, Any]] = self._filter_candidates(gse_metadata_list)
            if not filtered_gse_metadata: # Check after actual filtering
                self.logger.warning("No suitable datasets found after heuristic filtering.")
                raise NoDataFoundError("No suitable datasets found after heuristic filtering.")
            self.logger.info(f"Reduced to {len(filtered_gse_metadata)} candidates after heuristic filtering.")
            # Log a summary of filtered candidates if the list is not too long
            if filtered_gse_metadata:
                 self.logger.debug(f"Filtered candidates (first 3): {[cand.get('gse_id') for cand in filtered_gse_metadata[:3]]}")


            # Step 6: LLM Candidate Analysis (LLM Call) - Now Active
            self.logger.info(f"Step 6: Performing deep analysis on {len(filtered_gse_metadata)} candidates...")
            ranked_candidates: List[DatasetCandidate] = self._analyze_candidates_with_llm(user_query, filtered_gse_metadata)
            if not ranked_candidates: # Check after actual LLM call
                self.logger.warning("LLM analysis did not yield any ranked candidates, or list was empty.")
                # Depending on desired behavior, this could be an error or proceed with empty candidates.
                # For now, let's assume an empty list is possible but log it.
                # If it must yield candidates, then raise NoDataFoundError here.
            self.logger.info(f"LLM analysis yielded {len(ranked_candidates)} ranked candidates.")
            if ranked_candidates:
                self.logger.debug(f"Ranked candidates (first 3 with scores): {[(cand.gse_id, cand.rank_score) for cand in ranked_candidates[:3]]}")


            # Step 7: Assemble proposal (Logic depends on prior steps, especially ranked_candidates) - Now Active (but depends on ranked_candidates)
            self.logger.info("Step 7: Assembling final proposal...")
            proposal_id = str(uuid.uuid4())
            # Use the analysis type suggested by the initial query interpretation.
            # This could be refined later if needed, e.g., based on the chosen datasets.
            derived_analysis_type = query_intent.suggested_analysis_type

            analysis_proposal = AnalysisProposal(
                proposal_id=proposal_id,
                user_query=user_query,
                derived_analysis_type=str(derived_analysis_type), # Ensure string
                top_candidates=ranked_candidates # This now uses the result from _analyze_candidates_with_llm
            )
            self.logger.info(f"Analysis proposal {proposal_id} created successfully for user query '{user_query}'.")
            self.logger.debug(f"Final AnalysisProposal: {analysis_proposal.model_dump_json(indent=2)}")


            # Step 8: Store in cache
            expiry_time = time.time() + self.cache_ttl_seconds
            self.cache[cache_key] = (expiry_time, analysis_proposal)
            self.logger.info(f"Stored proposal {analysis_proposal.proposal_id} in cache. Expires at {time.ctime(expiry_time)}.")

            return analysis_proposal

        except NoDataFoundError as e:
            self.logger.warning(f"StrategyService: NoDataFoundError for query \"{user_query[:100]}...\": {e}")
            raise # Re-raise to be handled by the caller
        except NCBIAPIError as e:
            self.logger.error(f"StrategyService: NCBIAPIError for query \"{user_query[:100]}...\": {e}", exc_info=True)
            raise StrategyCreationError(f"NCBI API interaction failed: {e.message if hasattr(e, 'message') else str(e)}", original_exception=e) from e
        except (LLMAPIError, LLMOutputValidationError, PromptTemplateError, LLMGatewayConfigurationError) as e:
            self.logger.error(f"StrategyService: LLMGateway related error for query \"{user_query[:100]}...\": {e}", exc_info=True)
            # Specific error handling for LLM errors is now within the private methods.
            # This top-level catch can be more generic or removed if private methods handle all LLM errors by wrapping them.
            # For now, re-raise as StrategyCreationError if not already one.
            if isinstance(e, StrategyCreationError):
                 raise
            raise StrategyCreationError(f"An LLM Gateway operation failed: {str(e)}", original_exception=e) from e
        except NotImplementedError as e:
             self.logger.error(f"A core component of StrategyService is not yet implemented: {e}", exc_info=True)
             raise StrategyCreationError(f"Core functionality not implemented: {e}", original_exception=e) from e
        except Exception as e:
            self.logger.error(f"StrategyService: Unexpected error for query \"{user_query[:100]}...\": {e}", exc_info=True)
            raise StrategyCreationError(f"An unexpected error occurred during strategy creation: {str(e)}", original_exception=e) from e

# Example of how StrategyService might be instantiated and used (for illustration)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#
#     # Mock LLMGateway and config for standalone testing
#     class MockLLMGateway:
#         def get_structured_response(self, prompt_name, context, output_schema):
#             logger.info(f"MockLLMGateway.get_structured_response called for {prompt_name}")
#             if prompt_name == "interpret_query":
#                 # Return a dummy QueryIntent
#                 return QueryIntent(primary_disease="test_disease", species=["Homo sapiens"], research_goal="DEG", suggested_analysis_type="RNA-seq")
#             elif prompt_name == "analyze_candidates":
#                 # Return dummy DatasetCandidate list
#                 return [DatasetCandidate(gse_id="GSE_MOCK", srp_id="SRP_MOCK", title="Mock Title", summary="Mock Summary", species=["Homo sapiens"], sample_count=10, llm_recommendation_reason="Mock reason", rank_score=8.0)]
#             raise NotImplementedError(f"Mock behavior for {prompt_name} not defined.")
#
#         def get_text_response(self, prompt_name, context):
#             logger.info(f"MockLLMGateway.get_text_response called for {prompt_name}")
#             if prompt_name == "translate_to_search":
#                 return "test_disease AND rna-seq AND homo sapiens"
#             raise NotImplementedError(f"Mock behavior for {prompt_name} not defined.")

#     mock_llm_gateway = MockLLMGateway() if LLMGateway is None else LLMGateway("path/to/llm_config.yaml", "path/to/prompts.toml") # Adjust instantiation
#
#     # Dummy module_1_config.yaml content
#     dummy_module_config = {
#         "ncbi_api_key": None, # Or load from env for real test
#         "heuristic_filtering": {
#             "allowed_species": ["Homo sapiens", "Mus musculus"],
#             "min_sample_count": 5,
#             "require_srp_id": False,
#             "match_experiment_type_keywords": ["rna-seq", "transcriptome"]
#         },
#         "cache_ttl_seconds": 3600,
#         "ncbi_client_settings": { # Example of passing specific settings to NCBIClient
#             "ncbi_retry_attempts": 2
#         }
#     }
#
#     # Instantiate the service
#     strategy_service = StrategyService(
#         module_config=dummy_module_config,
#         llm_gateway=mock_llm_gateway, # type: ignore
#         ncbi_api_key=dummy_module_config.get("ncbi_api_key")
#     )
#
#     test_user_query = "Find RNA-seq data for Alzheimer's in humans."
#     try:
#         print(f"\nAttempting to create proposal for: \"{test_user_query}\"")
#         # This will currently raise NotImplementedError as private methods are placeholders
#         # proposal = strategy_service.create_proposal_from_query(test_user_query)
#         # print("\n--- Generated Proposal ---")
#         # print(proposal.model_dump_json(indent=2))
#         print("Skeleton service.py created. Full execution requires implementing private methods.")
#     except StrategyCreationError as e:
#         print(f"\nError creating proposal: {e}")
#     except Exception as e_main_test:
#         print(f"\nUnexpected error in main test block: {e_main_test}", exc_info=True)

```
