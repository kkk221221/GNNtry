# -*- coding: utf-8 -*-
"""
Unit tests for the StrategyService in Module 1.
"""
import unittest
from unittest.mock import Mock, patch, call # Using patch for more complex mocks if needed
import time # For testing cache expiry

# Modules to test
from aibiowflow.module_1_strategy.service import StrategyService
from aibiowflow.module_1_strategy.data_models import (
    QueryIntent,
    DatasetCandidate,
    AnalysisProposal,
    ResearchGoal, # Import Literal types if needed for constructing mock QueryIntent
    SuggestedAnalysisType
)
from aibiowflow.module_1_strategy.exceptions import (
    StrategyCreationError,
    NCBIAPIError,
    NoDataFoundError
)
# Mock LLM Gateway exceptions if they are distinct and need to be simulated
try:
    from aibiowflow.llm_gateway import LLMGateway # To mock its type hint
    from aibiowflow.llm_gateway.exceptions import LLMAPIError, LLMOutputValidationError, PromptTemplateError
except ImportError:
    LLMGateway = None
    LLMAPIError = type('MockLLMAPIError', (Exception,), {})
    LLMOutputValidationError = type('MockLLMOutputValidationError', (Exception,), {})
    PromptTemplateError = type('MockPromptTemplateError', (Exception,), {})


class TestStrategyService(unittest.TestCase):
    """
    Test suite for the StrategyService.
    """

    def setUp(self):
        """
        Set up common resources for each test.
        This will involve creating mock objects for dependencies like LLMGateway and NCBIClient.
        """
        self.mock_llm_gateway = Mock(spec=LLMGateway) if LLMGateway else Mock()

        # NCBIClient is instantiated within StrategyService, so we might need to patch its instantiation
        # or mock its methods if it were passed in directly.
        # For now, we'll mock the methods called on the instance created within StrategyService.
        # This will be done using @patch on individual test methods or the class.

        self.sample_user_query = "Find RNA-seq data for Alzheimer's in humans."

        self.sample_module_config = {
            "ncbi_api_key": "fake_ncbi_key", # Though NCBIClient itself might get it from env
            "heuristic_filtering": {
                "allowed_species": ["Homo sapiens", "Mus musculus"],
                "min_sample_count": 6,
                "require_srp_id": True,
                "match_experiment_type_keywords": ["rna-seq", "transcriptome"]
            },
            "cache_ttl_seconds": 3600, # 1 hour TTL for testing
            "ncbi_client_settings": {
                "ncbi_retry_attempts": 1 # Reduce retries for faster tests
            }
        }

        # We will create StrategyService instance in each test or a helper,
        # potentially with a patched NCBIClient.
        # self.strategy_service = StrategyService(
        # module_config=self.sample_module_config,
        # llm_gateway=self.mock_llm_gateway,
        # ncbi_api_key=self.sample_module_config.get("ncbi_api_key")
        # )

        # Example successful QueryIntent from LLM
        self.mock_query_intent = QueryIntent(
            primary_disease="Alzheimer's disease",
            species=["Homo sapiens"],
            research_goal="DEG",
            suggested_analysis_type="RNA-seq"
        )
        # Example successful GEO search string from LLM
        self.mock_geo_search_string = "(\"Alzheimer's disease\"[MeSH Terms] OR \"Alzheimer's disease\"[All Fields]) AND (\"Homo sapiens\"[Organism]) AND (\"rna seq\"[All Fields] OR \"transcriptome\"[All Fields])"

        # Example GSE IDs from NCBIClient.search_geo
        self.mock_gse_ids = ["GSE123", "GSE456"]

        # Example metadata from NCBIClient.get_geo_summaries
        self.mock_gse_metadata_list = [
            {"gse_id": "GSE123", "srp_id": "SRP111", "title": "RNA-seq of AD brain", "summary": "Relevant study 1", "species": ["Homo sapiens"], "sample_count": 10},
            {"gse_id": "GSE456", "srp_id": "SRP222", "title": "Mouse model of AD", "summary": "Relevant study 2", "species": ["Mus musculus"], "sample_count": 8},
            {"gse_id": "GSE789", "srp_id": None, "title": "Other study", "summary": "Less relevant", "species": ["Rattus norvegicus"], "sample_count": 5}, # Will be filtered
        ]

        # Example ranked candidates from LLM (_analyze_candidates_with_llm)
        self.mock_ranked_candidates = [
            DatasetCandidate(gse_id="GSE123", srp_id="SRP111", title="RNA-seq of AD brain", summary="Relevant study 1", species=["Homo sapiens"], sample_count=10, llm_recommendation_reason="Highly relevant", rank_score=9.0),
            DatasetCandidate(gse_id="GSE456", srp_id="SRP222", title="Mouse model of AD", summary="Relevant study 2", species=["Mus musculus"], sample_count=8, llm_recommendation_reason="Good model study", rank_score=7.5),
        ]

    def _create_service_with_mocked_ncbi(self):
        """Helper to create StrategyService with a mocked NCBIClient instance's methods."""
        # Patch the NCBIClient that StrategyService will instantiate
        # This is a common pattern: patch where it's looked up, not where it's defined.
        # So we patch 'aibiowflow.module_1_strategy.service.NCBIClient'
        self.patcher = patch('aibiowflow.module_1_strategy.service.NCBIClient')
        self.MockNCBIClientClass = self.patcher.start()
        self.addCleanup(self.patcher.stop) # Ensure patch is stopped after test

        self.mock_ncbi_client_instance = self.MockNCBIClientClass.return_value

        service = StrategyService(
            module_config=self.sample_module_config,
            llm_gateway=self.mock_llm_gateway,
            ncbi_api_key=self.sample_module_config.get("ncbi_api_key")
        )
        return service

    def test_successful_proposal_creation_workflow(self):
        """
        Test the full workflow for a successful proposal creation.
        This requires mocking all external calls (LLM Gateway, NCBIClient).
        """
        strategy_service = self._create_service_with_mocked_ncbi()

        # --- Mock LLM Gateway responses ---
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent, # For _interpret_user_query
            self.mock_ranked_candidates # For _analyze_candidates_with_llm
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string # For _translate_intent_to_search_query

        # --- Mock NCBIClient responses ---
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        # Simulate filtering: only GSE123 and GSE456 pass heuristic rules from self.mock_gse_metadata_list
        expected_metadata_for_llm_analysis = [
            self.mock_gse_metadata_list[0], # GSE123
            self.mock_gse_metadata_list[1]  # GSE456
        ]
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = self.mock_gse_metadata_list


        # --- Call the main method ---
        proposal = strategy_service.create_proposal_from_query(self.sample_user_query)

        # --- Assertions ---
        self.assertIsInstance(proposal, AnalysisProposal)
        self.assertEqual(proposal.user_query, self.sample_user_query)
        self.assertEqual(proposal.derived_analysis_type, self.mock_query_intent.suggested_analysis_type)
        self.assertEqual(len(proposal.top_candidates), 2)
        self.assertEqual(proposal.top_candidates[0].gse_id, "GSE123")
        self.assertEqual(proposal.top_candidates[1].gse_id, "GSE456")

        # Verify LLM calls
        self.mock_llm_gateway.get_structured_response.assert_any_call(
            prompt_name="interpret_query",
            context={"user_query": self.sample_user_query},
            output_schema=QueryIntent
        )
        self.mock_llm_gateway.get_text_response.assert_called_once_with(
            prompt_name="translate_to_search",
            context={"intent_json": self.mock_query_intent.model_dump_json()}
        )
        # This assertion needs to be more specific about the content of candidate_list_json
        # For simplicity, we'll check the prompt name and output schema for the second structured call.
        self.mock_llm_gateway.get_structured_response.assert_any_call(
            prompt_name="analyze_candidates",
            context=unittest.mock.ANY, # Or construct the exact expected context
            output_schema=List[DatasetCandidate]
        )

        # Verify NCBI calls
        self.mock_ncbi_client_instance.search_geo.assert_called_once_with(self.mock_geo_search_string)
        self.mock_ncbi_client_instance.get_geo_summaries.assert_called_once_with(self.mock_gse_ids)


    def test_no_gse_ids_found_from_ncbi(self):
        """Test scenario where NCBI ESearch returns no GSE IDs."""
        strategy_service = self._create_service_with_mocked_ncbi()

        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = [] # No IDs found

        with self.assertRaisesRegex(NoDataFoundError, "No GEO Series IDs found for search"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_no_summaries_found_for_gse_ids(self):
        """Test scenario where ESummary returns no metadata for found GSE IDs."""
        strategy_service = self._create_service_with_mocked_ncbi()

        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [] # No summaries

        with self.assertRaisesRegex(NoDataFoundError, "No metadata found for the .* retrieved GSE IDs"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_all_candidates_filtered_by_heuristics(self):
        """Test scenario where all candidates are filtered out by heuristic rules."""
        strategy_service = self._create_service_with_mocked_ncbi()

        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = ["GSE789"] # Only the filterable one
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[2]] # Metadata for GSE789

        with self.assertRaisesRegex(NoDataFoundError, "No suitable datasets found after heuristic filtering"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_llm_fails_to_rank_candidates(self):
        """Test scenario where LLM analysis returns no ranked candidates or invalid data."""
        strategy_service = self._create_service_with_mocked_ncbi()

        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent, # For _interpret_user_query
            []                       # Empty list for _analyze_candidates_with_llm
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0]] # One valid candidate passes heuristics

        # Current behavior is to log a warning if ranked_candidates is empty, and proceed.
        # If strictness demands NoDataFoundError here, this test needs adjustment.
        # For now, let's assume it proceeds and top_candidates will be empty.
        proposal = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertEqual(len(proposal.top_candidates), 0)
        # Add a specific test if LLM returns completely invalid (non-DatasetCandidate list) data for ranking later.

    def test_cache_hit(self):
        """Test that a cached result is returned for an identical query within TTL."""
        strategy_service = self._create_service_with_mocked_ncbi()

        # Populate mocks for the first call
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent, self.mock_ranked_candidates
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0], self.mock_gse_metadata_list[1]]

        # First call - populates cache
        proposal1 = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertIsNotNone(proposal1)

        # Reset mocks to ensure they are not called again for the cached hit
        self.mock_llm_gateway.reset_mock()
        self.mock_ncbi_client_instance.reset_mock() # MockNCBIClientClass.return_value.reset_mock()

        # Second call - should hit cache
        proposal2 = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertIsNotNone(proposal2)
        self.assertEqual(proposal1.proposal_id, proposal2.proposal_id) # Check if it's the same object or has same ID
        self.assertEqual(proposal1.top_candidates[0].gse_id, proposal2.top_candidates[0].gse_id)


        # Verify external services were not called the second time
        self.mock_llm_gateway.get_structured_response.assert_not_called()
        self.mock_llm_gateway.get_text_response.assert_not_called()
        self.mock_ncbi_client_instance.search_geo.assert_not_called()
        self.mock_ncbi_client_instance.get_geo_summaries.assert_not_called()


    def test_cache_expiry(self):
        """Test that an expired cache entry is recalculated."""
        strategy_service = self._create_service_with_mocked_ncbi()
        strategy_service.cache_ttl_seconds = 0.1 # Very short TTL for testing

        # --- First call (populates cache) ---
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent, self.mock_ranked_candidates
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0], self.mock_gse_metadata_list[1]]

        proposal1 = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertIsNotNone(proposal1)

        # Wait for cache to expire
        time.sleep(0.2)

        # --- Second call (should recalculate) ---
        # Re-setup mocks as they would be called again
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent, # For interpret_query
            self.mock_ranked_candidates # For analyze_candidates
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        # MockNCBIClientClass.return_value can be used to set up the instance's behavior again
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0], self.mock_gse_metadata_list[1]]


        proposal2 = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertIsNotNone(proposal2)
        self.assertNotEqual(proposal1.proposal_id, proposal2.proposal_id) # Should be a new proposal

        # Verify mocks were called again
        self.assertTrue(self.mock_llm_gateway.get_structured_response.call_count >= 2) # Called for interpret and analyze for 2nd call
        self.assertTrue(self.mock_llm_gateway.get_text_response.call_count >= 1) # Called at least once for 2nd call
        self.assertTrue(self.mock_ncbi_client_instance.search_geo.call_count >= 1)
        self.assertTrue(self.mock_ncbi_client_instance.get_geo_summaries.call_count >= 1)


    def test_llm_interpret_query_fails_validation(self):
        """Test handling when LLM returns data that fails QueryIntent validation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = LLMOutputValidationError("LLM output validation failed for QueryIntent")

        with self.assertRaisesRegex(StrategyCreationError, "Failed to interpret user query via LLM"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_llm_interpret_query_api_error(self):
        """Test handling when LLM API call fails during query interpretation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = LLMAPIError("Simulated API error", status_code=500)

        with self.assertRaisesRegex(StrategyCreationError, "Failed to interpret user query via LLM"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    # Add more tests for:
    # - NCBIClient API errors (e.g., ncbi_client.search_geo raises NCBIAPIError)
    # - LLM errors at each stage (translate_to_search, analyze_candidates)
    # - Different heuristic filtering outcomes.
    # - Edge cases for input data.

if __name__ == '__main__':
    unittest.main()
```
