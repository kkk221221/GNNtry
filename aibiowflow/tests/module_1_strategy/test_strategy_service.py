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

    # --- Tests for Heuristic Filtering Logic ---

    def _run_filter_test(self, service_config_override, input_metadata, expected_passing_gse_ids):
        """
        Helper to test filtering logic by providing specific service config
        and input metadata, then checking which GSE IDs pass.
        """
        config = {**self.sample_module_config, **service_config_override}

        # Patch NCBIClient instantiation within a specific service instance for this test
        with patch('aibiowflow.module_1_strategy.service.NCBIClient') as MockNCBIClientClass:
            mock_ncbi_instance = MockNCBIClientClass.return_value
            strategy_service = StrategyService(
                module_config=config,
                llm_gateway=self.mock_llm_gateway,
                ncbi_api_key=config.get("ncbi_api_key")
            )

            # Mock the preceding steps to isolate filtering
            self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
            self.mock_llm_gateway.get_text_response.return_value = "any_search_query"
            mock_ncbi_instance.search_geo.return_value = [item['gse_id'] for item in input_metadata] # Dummy IDs
            mock_ncbi_instance.get_geo_summaries.return_value = input_metadata

            # Mock the LLM call for candidate analysis to avoid errors after filtering
            # It should receive only the candidates that passed heuristic filtering
            def mock_llm_analysis(user_query, candidates_metadata_for_llm):
                # This is where we can assert what `_filter_candidates` produced
                passing_gse_ids = [cand['gse_id'] for cand in candidates_metadata_for_llm]
                self.assertCountEqual(passing_gse_ids, expected_passing_gse_ids,
                                     "GSE IDs passed to LLM analysis do not match expected.")

                # Return minimal valid data to allow proposal creation to complete
                return [DatasetCandidate(gse_id=gid, srp_id="SRP_MOCK", title="Mock", summary="Mock",
                                         species=["Homo sapiens"], sample_count=10,
                                         llm_recommendation_reason="Mock", rank_score=1.0)
                        for gid in passing_gse_ids]

            self.mock_llm_gateway.get_structured_response.side_effect = [
                 self.mock_query_intent, # For _interpret_user_query
                 mock_llm_analysis       # For _analyze_candidates_with_llm
            ]


            if not expected_passing_gse_ids:
                with self.assertRaises(NoDataFoundError):
                    strategy_service.create_proposal_from_query(self.sample_user_query)
            else:
                proposal = strategy_service.create_proposal_from_query(self.sample_user_query)
                self.assertCountEqual([cand.gse_id for cand in proposal.top_candidates], expected_passing_gse_ids)


    def test_filter_allowed_species(self):
        """Test species filtering."""
        config_override = {"heuristic_filtering": {"allowed_species": ["Homo sapiens"], "min_sample_count": 1, "require_srp_id": False}}
        metadata = [
            {"gse_id": "GSE1", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S1"},
            {"gse_id": "GSE2", "species": ["Mus musculus"], "sample_count": 5, "srp_id": "S2"}, # Filtered out
            {"gse_id": "GSE3", "species": ["Homo sapiens", "Mus musculus"], "sample_count": 5, "srp_id": "S3"}, # Passes
        ]
        self._run_filter_test(config_override, metadata, ["GSE1", "GSE3"])

    def test_filter_min_sample_count(self):
        """Test minimum sample count filtering."""
        config_override = {"heuristic_filtering": {"allowed_species": ["Homo sapiens"], "min_sample_count": 10, "require_srp_id": False}}
        metadata = [
            {"gse_id": "GSE1", "species": ["Homo sapiens"], "sample_count": 12, "srp_id": "S1"},
            {"gse_id": "GSE2", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S2"}, # Filtered out
            {"gse_id": "GSE3", "species": ["Homo sapiens"], "sample_count": 10, "srp_id": "S3"},
        ]
        self._run_filter_test(config_override, metadata, ["GSE1", "GSE3"])

    def test_filter_require_srp_id(self):
        """Test SRP ID requirement filtering."""
        config_override = {"heuristic_filtering": {"allowed_species": ["Homo sapiens"], "min_sample_count": 1, "require_srp_id": True}}
        metadata = [
            {"gse_id": "GSE1", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S1"},
            {"gse_id": "GSE2", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": None}, # Filtered out
            {"gse_id": "GSE3", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": ""},   # Filtered out (empty string)
        ]
        self._run_filter_test(config_override, metadata, ["GSE1"])

    def test_filter_match_experiment_keywords(self):
        """Test experiment type keyword matching."""
        config_override = {
            "heuristic_filtering": {
                "allowed_species": ["Homo sapiens"], "min_sample_count": 1, "require_srp_id": False,
                "match_experiment_type_keywords": ["RNA-seq", "transcriptome"]
            }
        }
        metadata = [
            {"gse_id": "GSE1", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S1", "title": "Study of RNA-seq in cancer", "summary": ""},
            {"gse_id": "GSE2", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S2", "title": "Proteomics study", "summary": "Whole transcriptome analysis"},
            {"gse_id": "GSE3", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S3", "title": "Methylation patterns", "summary": "No keywords here"}, # Filtered
        ]
        self._run_filter_test(config_override, metadata, ["GSE1", "GSE2"])

    def test_filter_no_keywords_defined(self):
        """Test filtering when no experiment keywords are defined (should pass all)."""
        config_override = {
            "heuristic_filtering": {
                "allowed_species": ["Homo sapiens"], "min_sample_count": 1, "require_srp_id": False,
                "match_experiment_type_keywords": [] # Empty list
            }
        }
        metadata = [
            {"gse_id": "GSE1", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S1", "title": "Study of RNA-seq in cancer", "summary": ""},
            {"gse_id": "GSE2", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S2", "title": "Proteomics study", "summary": "Whole transcriptome analysis"},
            {"gse_id": "GSE3", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S3", "title": "Methylation patterns", "summary": "No keywords here"},
        ]
        # All should pass keyword filter because no keywords are specified for matching
        self._run_filter_test(config_override, metadata, ["GSE1", "GSE2", "GSE3"])


    def test_filter_all_rules_combined(self):
        """Test combination of all filtering rules."""
        config_override = {
            "heuristic_filtering": {
                "allowed_species": ["Homo sapiens"],
                "min_sample_count": 10,
                "require_srp_id": True,
                "match_experiment_type_keywords": ["RNA-seq"]
            }
        }
        metadata = [
            # Passes all
            {"gse_id": "GSE_PASS", "species": ["Homo sapiens"], "sample_count": 12, "srp_id": "SRP_PASS", "title": "Human RNA-seq study", "summary": ""},
            # Fails species
            {"gse_id": "GSE_FAIL_SPECIES", "species": ["Mus musculus"], "sample_count": 12, "srp_id": "SRP1", "title": "Mouse RNA-seq", "summary": ""},
            # Fails sample count
            {"gse_id": "GSE_FAIL_SAMPLES", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "SRP2", "title": "Human RNA-seq low N", "summary": ""},
            # Fails SRP ID
            {"gse_id": "GSE_FAIL_SRP", "species": ["Homo sapiens"], "sample_count": 12, "srp_id": None, "title": "Human RNA-seq no SRA", "summary": ""},
            # Fails keyword
            {"gse_id": "GSE_FAIL_KEYWORD", "species": ["Homo sapiens"], "sample_count": 12, "srp_id": "SRP4", "title": "Human ChIP-seq study", "summary": ""},
        ]
        self._run_filter_test(config_override, metadata, ["GSE_PASS"])

    def test_filter_empty_metadata_input(self):
        """Test filtering with an empty list of metadata."""
        config_override = self.sample_module_config # Use default filters
        metadata = []
        # Expect NoDataFoundError before LLM analysis is even attempted if filtering results in empty
        # The _run_filter_test will assert NoDataFoundError if expected_passing_gse_ids is empty
        self._run_filter_test(config_override, metadata, [])

    # --- Tests for LLM Interactions & Error Handling ---

    def test_interpret_query_llm_output_validation_error(self):
        """Test LLMOutputValidationError during query interpretation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = LLMOutputValidationError("Invalid QueryIntent format")

        with self.assertRaisesRegex(StrategyCreationError, "Failed to interpret user query via LLM.*Invalid QueryIntent format"):
            strategy_service.create_proposal_from_query(self.sample_user_query)
        self.mock_llm_gateway.get_structured_response.assert_called_once_with(
            prompt_name="interpret_query", context={"user_query": self.sample_user_query}, output_schema=QueryIntent
        )

    def test_interpret_query_llm_api_error(self):
        """Test LLMAPIError during query interpretation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = LLMAPIError("LLM API down", status_code=503)

        with self.assertRaisesRegex(StrategyCreationError, "Failed to interpret user query via LLM.*LLM API down"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_interpret_query_llm_prompt_template_error(self):
        """Test PromptTemplateError during query interpretation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = PromptTemplateError("Missing var in prompt")

        with self.assertRaisesRegex(StrategyCreationError, "Failed to interpret user query via LLM.*Missing var in prompt"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_translate_intent_llm_text_response_error(self):
        """Test LLM error during intent to search query translation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent # Step 1 success
        self.mock_llm_gateway.get_text_response.side_effect = LLMAPIError("LLM API for text failed")

        with self.assertRaisesRegex(StrategyCreationError, "Failed to translate intent to search query via LLM.*LLM API for text failed"):
            strategy_service.create_proposal_from_query(self.sample_user_query)
        self.mock_llm_gateway.get_text_response.assert_called_once_with(
             prompt_name="translate_to_search", context={"intent_json": self.mock_query_intent.model_dump_json()}
        )

    def test_translate_intent_llm_returns_empty_string(self):
        """Test when LLM returns an empty string for search query translation."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent # Step 1 success
        self.mock_llm_gateway.get_text_response.return_value = "" # Empty string

        with self.assertRaisesRegex(StrategyCreationError, "LLM did not return a valid search string"):
            strategy_service.create_proposal_from_query(self.sample_user_query)


    def test_analyze_candidates_llm_output_validation_error(self):
        """Test LLMOutputValidationError during candidate analysis."""
        strategy_service = self._create_service_with_mocked_ncbi()
        # Setup mocks for successful steps before candidate analysis
        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent # step 1
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string # step 2
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids # step 3
        # Provide one valid candidate that passes heuristic filtering (default config)
        passing_candidate_metadata = [self.mock_gse_metadata_list[0]] # GSE123 - Homo sapiens, 10 samples, SRP111, RNA-seq title
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = passing_candidate_metadata # step 4 (after filtering, this is what's passed)

        # Configure the second call to get_structured_response (for analyze_candidates) to fail
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent,
            LLMOutputValidationError("Invalid DatasetCandidate list")
        ]

        with self.assertRaisesRegex(StrategyCreationError, "Failed to analyze candidates via LLM.*Invalid DatasetCandidate list"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

        # Check that interpret_query was called
        self.mock_llm_gateway.get_structured_response.assert_any_call(
            prompt_name="interpret_query", context={"user_query": self.sample_user_query}, output_schema=QueryIntent
        )
        # Check that analyze_candidates was called (and was the one that raised the error)
        # The context for analyze_candidates would be the JSON string of passing_candidate_metadata
        # For simplicity, we check ANY context here, knowing the error came from this call.
        self.mock_llm_gateway.get_structured_response.assert_any_call(
            prompt_name="analyze_candidates", context=unittest.mock.ANY, output_schema=List[DatasetCandidate]
        )

    def test_analyze_candidates_llm_api_error(self):
        """Test LLMAPIError during candidate analysis."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent,
            LLMAPIError("Analyze API down")
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0]]


        with self.assertRaisesRegex(StrategyCreationError, "Failed to analyze candidates via LLM.*Analyze API down"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_analyze_candidates_llm_returns_non_list(self):
        """Test when LLM returns a single object instead of a list for analyze_candidates."""
        strategy_service = self._create_service_with_mocked_ncbi()
        single_candidate = DatasetCandidate(gse_id="GSE_SINGLE", srp_id="SRP_S", title="T", summary="S", species=["H.sapiens"], sample_count=10, llm_recommendation_reason="R", rank_score=8)

        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent,
            single_candidate # LLM returns a single object, not a list
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = ["GSE_SINGLE"]
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [{"gse_id": "GSE_SINGLE", "srp_id": "SRP_S", "title": "T", "summary": "S", "species": ["H.sapiens"], "sample_count": 10}]

        # The service currently wraps a single DatasetCandidate in a list if returned by LLM.
        proposal = strategy_service.create_proposal_from_query(self.sample_user_query)
        self.assertEqual(len(proposal.top_candidates), 1)
        self.assertEqual(proposal.top_candidates[0].gse_id, "GSE_SINGLE")

    def test_analyze_candidates_llm_returns_list_with_wrong_items(self):
        """Test when LLM returns a list with non-DatasetCandidate items for analyze_candidates."""
        strategy_service = self._create_service_with_mocked_ncbi()

        self.mock_llm_gateway.get_structured_response.side_effect = [
            self.mock_query_intent,
            [{"gse_id": "GSE_DICT"}] # List of dicts, not DatasetCandidate objects
        ]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = ["GSE_DICT"]
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [{"gse_id": "GSE_DICT", "srp_id": "SRP_D", "title": "T", "summary": "S", "species": ["H.sapiens"], "sample_count": 10}]

        # This should be caught by Pydantic validation in LLMGateway or by the service's check.
        # Current service code: `if not isinstance(ranked_candidate_list, list) or not all(isinstance(item, DatasetCandidate) for item in ranked_candidate_list):`
        with self.assertRaisesRegex(StrategyCreationError, "LLM did not return a valid list of DatasetCandidate objects"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_ncbi_client_search_geo_api_error(self):
        """Test NCBIAPIError during ncbi_client.search_geo."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.side_effect = NCBIAPIError("NCBI ESearch down", status_code=500)

        with self.assertRaisesRegex(StrategyCreationError, "NCBI API interaction failed.*NCBI ESearch down"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    def test_ncbi_client_get_summaries_api_error(self):
        """Test NCBIAPIError during ncbi_client.get_geo_summaries."""
        strategy_service = self._create_service_with_mocked_ncbi()
        self.mock_llm_gateway.get_structured_response.return_value = self.mock_query_intent
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids # Search succeeds
        self.mock_ncbi_client_instance.get_geo_summaries.side_effect = NCBIAPIError("NCBI ESummary down", status_code=503)

        with self.assertRaisesRegex(StrategyCreationError, "NCBI API interaction failed.*NCBI ESummary down"):
            strategy_service.create_proposal_from_query(self.sample_user_query)

    # --- Tests for Configuration Impact & Cache ---

    def test_config_impact_heuristic_rules_species(self):
        """Test that changing 'allowed_species' in config affects filtering."""
        # Original config in setUp allows "Homo sapiens", "Mus musculus"
        # Test with a more restrictive species list
        config_override = {
            "heuristic_filtering": {
                "allowed_species": ["Mus musculus"], # Only Mus musculus
                "min_sample_count": 1, # Keep other rules permissive
                "require_srp_id": False,
                "match_experiment_type_keywords": []
            }
        }
        metadata = [
            {"gse_id": "GSE_HS", "species": ["Homo sapiens"], "sample_count": 5, "srp_id": "S1", "title":"HS study"},
            {"gse_id": "GSE_MM", "species": ["Mus musculus"], "sample_count": 5, "srp_id": "S2", "title":"MM study"},
            {"gse_id": "GSE_BOTH", "species": ["Homo sapiens", "Mus musculus"], "sample_count": 5, "srp_id": "S3", "title":"Both study"},
        ]
        # Expected: GSE_MM and GSE_BOTH should pass as they include Mus musculus
        self._run_filter_test(config_override, metadata, ["GSE_MM", "GSE_BOTH"])

    def test_config_impact_heuristic_rules_min_samples(self):
        """Test that changing 'min_sample_count' in config affects filtering."""
        config_override = {
            "heuristic_filtering": {
                "allowed_species": ["Homo sapiens", "Mus musculus"],
                "min_sample_count": 20, # Higher sample count
                "require_srp_id": False,
                "match_experiment_type_keywords": []
            }
        }
        metadata = [
            {"gse_id": "GSE_LOW_N", "species": ["Homo sapiens"], "sample_count": 10, "srp_id": "S1", "title":"Low N"},
            {"gse_id": "GSE_HIGH_N", "species": ["Mus musculus"], "sample_count": 25, "srp_id": "S2", "title":"High N"},
            {"gse_id": "GSE_EQ_N", "species": ["Homo sapiens"], "sample_count": 20, "srp_id": "S3", "title":"Equal N"},
        ]
        self._run_filter_test(config_override, metadata, ["GSE_HIGH_N", "GSE_EQ_N"])

    def test_config_impact_cache_ttl(self):
        """Test that 'cache_ttl_seconds' in config affects cache expiry."""
        # This test is similar to test_cache_expiry but explicitly sets TTL via config
        # to ensure the service reads it correctly.

        # Create a service instance with a very short TTL via config
        short_ttl_config = self.sample_module_config.copy()
        short_ttl_config['cache_ttl_seconds'] = 0.05 # Very short

        # Patch NCBIClient for this specific test scenario
        with patch('aibiowflow.module_1_strategy.service.NCBIClient') as MockNCBIClientClass_TTLTest:
            mock_ncbi_instance_ttl = MockNCBIClientClass_TTLTest.return_value

            strategy_service_short_ttl = StrategyService(
                module_config=short_ttl_config,
                llm_gateway=self.mock_llm_gateway,
                ncbi_api_key=short_ttl_config.get("ncbi_api_key")
            )

            # --- First call (populates cache) ---
            # Reset and re-configure mocks for the first call
            self.mock_llm_gateway.reset_mock()
            self.mock_llm_gateway.get_structured_response.side_effect = [
                self.mock_query_intent, self.mock_ranked_candidates
            ]
            self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
            mock_ncbi_instance_ttl.search_geo.return_value = self.mock_gse_ids
            mock_ncbi_instance_ttl.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0], self.mock_gse_metadata_list[1]]

            proposal1 = strategy_service_short_ttl.create_proposal_from_query(self.sample_user_query + "_ttl_test") # Unique query
            self.assertIsNotNone(proposal1)

            # Verify mocks were called for the first proposal
            self.mock_llm_gateway.get_structured_response.assert_any_call(prompt_name="interpret_query", context=unittest.mock.ANY, output_schema=QueryIntent)
            self.mock_llm_gateway.get_text_response.assert_called_once()
            mock_ncbi_instance_ttl.search_geo.assert_called_once()
            mock_ncbi_instance_ttl.get_geo_summaries.assert_called_once()

            # Wait for cache to expire (longer than the 0.05s TTL)
            time.sleep(0.1)

            # --- Second call (should recalculate due to short TTL from config) ---
            # Reset and re-configure mocks for the second call
            self.mock_llm_gateway.reset_mock() # Reset all mock call counts etc.
            mock_ncbi_instance_ttl.reset_mock()

            self.mock_llm_gateway.get_structured_response.side_effect = [
                self.mock_query_intent, # For interpret_query
                self.mock_ranked_candidates # For analyze_candidates
            ]
            self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string + "_recalc" # Ensure different if needed
            mock_ncbi_instance_ttl.search_geo.return_value = self.mock_gse_ids # Can be same
            mock_ncbi_instance_ttl.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0]] # Slightly different to ensure new obj

            proposal2 = strategy_service_short_ttl.create_proposal_from_query(self.sample_user_query + "_ttl_test") # Same unique query
            self.assertIsNotNone(proposal2)
            self.assertNotEqual(proposal1.proposal_id, proposal2.proposal_id, "Proposal ID should be different after cache expiry and recalc.")
            # If data changed, we can assert that too:
            self.assertEqual(len(proposal2.top_candidates), 1, "Recalculated proposal should have different data if mock changed.")

            # Verify mocks were called again for the second proposal
            self.mock_llm_gateway.get_structured_response.assert_any_call(prompt_name="interpret_query", context=unittest.mock.ANY, output_schema=QueryIntent)
            self.mock_llm_gateway.get_text_response.assert_called_once()
            mock_ncbi_instance_ttl.search_geo.assert_called_once()
            mock_ncbi_instance_ttl.get_geo_summaries.assert_called_once()

    def test_cache_different_queries_no_collision(self):
        """Test that different queries result in different cache entries."""
        strategy_service = self._create_service_with_mocked_ncbi() # Uses default TTL

        query1 = self.sample_user_query + " (Query 1)"
        query2 = self.sample_user_query + " (Query 2)"

        # --- Call for Query 1 ---
        self.mock_llm_gateway.get_structured_response.side_effect = [self.mock_query_intent, self.mock_ranked_candidates]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string
        self.mock_ncbi_client_instance.search_geo.return_value = self.mock_gse_ids
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [self.mock_gse_metadata_list[0]]

        proposal1 = strategy_service.create_proposal_from_query(query1)
        call_count_interpret1 = self.mock_llm_gateway.get_structured_response.call_count
        call_count_text1 = self.mock_llm_gateway.get_text_response.call_count
        call_count_search1 = self.mock_ncbi_client_instance.search_geo.call_count
        call_count_summary1 = self.mock_ncbi_client_instance.get_geo_summaries.call_count


        # --- Call for Query 2 (should not hit cache from Query 1) ---
        # Create specific mock data for query2 that will pass default filters
        mock_metadata_q2_pass = {
            "gse_id": "GSE_Q2",
            "srp_id": "SRP_Q2",
            "title": "RNA-seq study for Query 2", # Contains "RNA-seq"
            "summary": "Relevant study for query 2",
            "species": ["Mus musculus"],
            "sample_count": 10
        }
        # Ensure only one candidate is returned by LLM for simplicity in assertion
        mock_ranked_candidates_q2 = [
            DatasetCandidate(gse_id="GSE_Q2", srp_id="SRP_Q2", title=mock_metadata_q2_pass["title"],
                             summary=mock_metadata_q2_pass["summary"], species=mock_metadata_q2_pass["species"],
                             sample_count=mock_metadata_q2_pass["sample_count"],
                             llm_recommendation_reason="Relevant for Q2", rank_score=8.0)
        ]

        self.mock_llm_gateway.get_structured_response.side_effect = [self.mock_query_intent, mock_ranked_candidates_q2]
        self.mock_llm_gateway.get_text_response.return_value = self.mock_geo_search_string + "_q2"
        self.mock_ncbi_client_instance.search_geo.return_value = [mock_metadata_q2_pass["gse_id"]]
        self.mock_ncbi_client_instance.get_geo_summaries.return_value = [mock_metadata_q2_pass]

        proposal2 = strategy_service.create_proposal_from_query(query2)

        self.assertNotEqual(proposal1.proposal_id, proposal2.proposal_id)
        # proposal1 should have 1 candidate from self.mock_gse_metadata_list[0] if mock_ranked_candidates has 1 for it
        # For simplicity, let's ensure proposal1 also expects one candidate based on its mocks.
        # The initial mock_ranked_candidates has 2 items. Let's adjust the first call's mock for consistency of this test.
        # However, the key is that proposal2 is successfully created.
        self.assertTrue(len(proposal1.top_candidates) > 0, "Proposal 1 should have candidates")
        self.assertEqual(len(proposal2.top_candidates), 1, "Proposal 2 should have one candidate")
        self.assertEqual(proposal2.top_candidates[0].gse_id, "GSE_Q2")


        # Ensure mocks were called again for the second query
        self.assertEqual(self.mock_llm_gateway.get_structured_response.call_count, call_count_interpret1 + 2) # interpret + analyze for Q2
        self.assertEqual(self.mock_llm_gateway.get_text_response.call_count, call_count_text1 + 1)
        self.assertEqual(self.mock_ncbi_client_instance.search_geo.call_count, call_count_search1 + 1)
        self.assertEqual(self.mock_ncbi_client_instance.get_geo_summaries.call_count, call_count_summary1 + 1)


if __name__ == '__main__':
    unittest.main()
```
