# -*- coding: utf-8 -*-
"""
Pydantic Data Models for Module 1: Literature & Strategy Service.

These models define the structured data used within the module, especially for
LLM interactions and the final analysis proposal output.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Literal types for enum-like fields as described in the specification
ResearchGoal = Literal['DEG', 'Variant', 'Epigenomics', 'Other']
SuggestedAnalysisType = Literal['RNA-seq', 'WGS', 'ChIP-seq', 'Other', 'Unknown']

class QueryIntent(BaseModel):
    """
    Represents the structured understanding of the user's initial query,
    as interpreted by an LLM (Prompt 1 in the specification).
    """
    primary_disease: str = Field(
        ...,
        description="主要关注的疾病或生物学状况。"
    )
    primary_gene: Optional[str] = Field(
        None,
        description="主要关注的基因 (如果明确提及)。"
    )
    species: List[str] = Field(
        ...,
        description="研究涉及的物种列表 (例如 ['Homo sapiens', 'Mus musculus'])。"
    )
    research_goal: ResearchGoal = Field(
        ...,
        description="用户的核心研究目标 (例如差异表达分析、变异检测等)。"
    )
    suggested_analysis_type: SuggestedAnalysisType = Field(
        ...,
        description="根据用户查询建议的生物信息学分析类型 (例如 RNA-seq, WGS)。"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "primary_disease": "Alzheimer's disease",
                    "primary_gene": "APOE",
                    "species": ["Homo sapiens"],
                    "research_goal": "DEG",
                    "suggested_analysis_type": "RNA-seq"
                }
            ]
        }
    }


class DatasetCandidate(BaseModel):
    """
    Represents a single candidate dataset (e.g., a GEO Series) identified
    and evaluated by the service.
    """
    gse_id: str = Field(
        ...,
        description="GEO Series ID (例如 'GSE12345')."
    )
    srp_id: Optional[str] = Field( # Made optional as per heuristic rule FR-M1-05 potentially filtering some without it initially
        None,
        description="关联的SRA项目ID (例如 'SRP123456')，如果存在。"
    )
    title: str = Field(
        ...,
        description="数据集的标题。"
    )
    summary: str = Field(
        ...,
        description="数据集的摘要信息。"
    )
    species: List[str] = Field(
        default_factory=list,
        description="数据集中涉及的物种列表。"
    )
    sample_count: int = Field(
        ...,
        description="数据集中的样本总数。"
    )
    llm_recommendation_reason: str = Field(
        ...,
        description="LLM 对为何推荐此数据集的解释。"
    )
    rank_score: float = Field(
        ...,
        description="LLM 评估的推荐分数 (例如 1-10)，分数越高越推荐。"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gse_id": "GSE12345",
                    "srp_id": "SRP123456",
                    "title": "RNA-seq analysis of Alzheimer's disease brain tissue",
                    "summary": "This study performs RNA-sequencing on post-mortem brain samples...",
                    "species": ["Homo sapiens"],
                    "sample_count": 20,
                    "llm_recommendation_reason": "Directly relevant to Alzheimer's, good sample size for RNA-seq.",
                    "rank_score": 9.5
                }
            ]
        }
    }


class AnalysisProposal(BaseModel):
    """
    Represents the final output of the Literature & Strategy Service.
    It contains the original query, derived analysis type, and a list of
    top recommended dataset candidates.
    """
    proposal_id: str = Field(
        ...,
        description="分析方案的唯一标识符 (例如 UUID)。"
    )
    user_query: str = Field(
        ...,
        description="用户输入的原始科研问题。"
    )
    derived_analysis_type: str = Field(# Could also use SuggestedAnalysisType here if it's always one of those
        ...,
        description="从用户查询和选定数据集中推断出的主要分析类型。"
    )
    top_candidates: List[DatasetCandidate] = Field(
        default_factory=list,
        description="经过筛选和LLM评估后，最被推荐的数据集候选列表。"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "proposal_id": "prop_2a7d3ef8-12a7-4b8c-8f92-3a4c1d78b9f0",
                    "user_query": "I want to find differentially expressed genes in Alzheimer's disease brain tissue.",
                    "derived_analysis_type": "RNA-seq",
                    "top_candidates": [
                        {
                            "gse_id": "GSE12345",
                            "srp_id": "SRP123456",
                            "title": "RNA-seq analysis of Alzheimer's disease brain tissue",
                            "summary": "This study performs RNA-sequencing on post-mortem brain samples...",
                            "species": ["Homo sapiens"],
                            "sample_count": 20,
                            "llm_recommendation_reason": "Directly relevant to Alzheimer's, good sample size for RNA-seq.",
                            "rank_score": 9.5
                        },
                        {
                            "gse_id": "GSE67890",
                            "srp_id": "SRP678901",
                            "title": "Transcriptomic study of AD models",
                            "summary": "Mouse models of Alzheimer's were analyzed using RNA-seq.",
                            "species": ["Mus musculus"],
                            "sample_count": 12,
                            "llm_recommendation_reason": "Relevant model organism study, though not human tissue.",
                            "rank_score": 7.0
                        }
                    ]
                }
            ]
        }
    }
