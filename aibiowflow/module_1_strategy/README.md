# 模块一: 文献与数据策略服务 (Literature & Strategy Service)

## 1. 模块概述

本模块 (`aibiowflow.module_1_strategy`) 是 AIBioFlow 系统的认知与决策起点。其核心使命是将用户输入的自然语言科研问题，转化为一个具体、可执行、有数据支撑的分析计划。它通过与 LLM 网关交互进行意图解析和内容分析，并调用 NCBI 等外部数据库接口获取相关文献与数据集信息。

最终，本模块旨在生成一份结构化的分析方案 (`AnalysisProposal`)，其中包含推荐的数据集，供用户审核。

## 2. 主要组件

本模块主要包含以下组件：

*   **`service.py`**: 包含核心的 `StrategyService` 类，负责编排整个文献检索、数据筛选和策略生成的工作流。
*   **`ncbi_client.py`**: 包含 `NCBIClient` 类，封装了与 NCBI E-utilities API (如 ESearch, ESummary) 的交互逻辑，用于检索 GEO 数据集等信息。
*   **`data_models.py`**: 定义了本模块使用的核心 Pydantic 数据模型，如 `AnalysisProposal`, `DatasetCandidate`, 和 `QueryIntent`，用于规范化数据结构。
*   **`exceptions.py`**: 定义了本模块特定的自定义异常，如 `StrategyCreationError`, `NCBIAPIError`, `NoDataFoundError`，用于错误处理。
*   **`prompts_m1.toml`** (位于 `aibiowflow/llm_gateway/`): 存储本模块与 LLM 网关交互时使用的特定 Prompt 模板。 (路径待最终确认)
*   **`module_1_config.yaml`** (位于 `aibiowflow/config/`): 存储本模块的特定配置，例如启发式过滤规则、NCBI API参数、缓存设置等。

## 3. 核心职责

*   **查询理解与翻译**: 将用户自然语言查询转化为结构化意图和数据库搜索查询。
*   **数据检索**: 从 NCBI GEO 等公共数据库检索相关数据集。
*   **启发式初筛**: 根据预设规则（如物种、样本数）对检索结果进行初步筛选。
*   **候选集深度分析**: 利用 LLM 对初筛后的候选数据集进行相关性评估和推荐。
*   **分析方案生成**: 将所有信息整合成结构化的 `AnalysisProposal` 对象。
*   **触发人工审核**: 准备好将生成的方案提交给用户进行最终决策。

## 4. 使用说明

### 4.1 初始化 `StrategyService`

`StrategyService` 需要模块特定的配置字典、一个 `LLMGateway` 实例以及可选的 NCBI API 密钥来进行初始化。

```python
from aibiowflow.llm_gateway import LLMGateway
from aibiowflow.module_1_strategy.service import StrategyService
import yaml # 用于加载配置

# 1. 初始化 LLM 网关 (假设已配置好)
# llm_config_path = "path/to/llm_gateway_config.yaml"
# llm_prompts_path = "path/to/llm_gateway_prompts.toml" # 包含 prompts_m1.toml 的内容或单独加载
# llm_gateway = LLMGateway(config_path=llm_config_path, prompt_path=llm_prompts_path)
# 对于测试，可以使用 MockLLMGateway
mock_llm_gateway = unittest.mock.Mock(spec=LLMGateway)


# 2. 加载模块1的配置
try:
    with open("aibiowflow/config/module_1_config.yaml", 'r', encoding='utf-8') as f:
        module_1_config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: module_1_config.yaml not found. Please ensure it exists in aibiowflow/config/")
    module_1_config = {} # 或者提供一个默认的最小配置

# 3. 获取 NCBI API 密钥 (通常从环境变量或更高级别的配置管理获取)
# ncbi_api_key = os.getenv("NCBI_API_KEY")
ncbi_api_key = module_1_config.get("ncbi_api_key") # Can also be "env:VAR" if handled by a config loader

# 4. 实例化服务
# strategy_service = StrategyService(
#     module_config=module_1_config,
#     llm_gateway=llm_gateway, # 使用真实的 LLM 网关
#     ncbi_api_key=ncbi_api_key
# )

# 使用Mock LLM Gateway进行示例
strategy_service_example = StrategyService(
    module_config=module_1_config,
    llm_gateway=mock_llm_gateway, # 使用 Mock 对象
    ncbi_api_key=ncbi_api_key
)
print("StrategyService 示例初始化成功 (使用Mock LLM网关)")
```

### 4.2 调用主要方法

模块的主要功能通过 `create_proposal_from_query` 方法暴露：

```python
user_query = "我想研究与阿尔茨海默病相关的人类脑组织中的差异表达基因"

# try:
#     # 以下调用会实际执行整个流程 (LLM调用, NCBI调用等)
#     # 为避免实际API调用，此示例将不直接运行此行，除非llm_gateway和ncbi_client被正确mock
#     # analysis_proposal = strategy_service.create_proposal_from_query(user_query)
#     # print(f"生成的分析方案ID: {analysis_proposal.proposal_id}")
#     # for candidate in analysis_proposal.top_candidates:
#     #     print(f"  候选数据集: {candidate.gse_id} - {candidate.title} (推荐分: {candidate.rank_score})")
#     print(f"若要实际运行，请确保LLM网关和NCBI客户端已正确配置和（如果需要）mock。")
# except NoDataFoundError as e:
#     print(f"未能找到数据: {e}")
# except StrategyCreationError as e:
#     print(f"创建分析方案时出错: {e}")
# except Exception as e:
#     print(f"发生意外错误: {e}")
```
上述代码片段演示了如何初始化和调用服务。在实际应用中，`LLMGateway` 和 `NCBIClient` 的交互将是实时的（或通过单元测试中的 mock 对象模拟）。

### 4.3 配置项说明 (`module_1_config.yaml`)

模块的行为可以通过 `aibiowflow/config/module_1_config.yaml` 文件进行配置。关键配置项包括：

*   `ncbi_api_key`: NCBI API 密钥 (推荐使用 `env:NCBI_API_KEY` 形式通过环境变量设置)。
*   `ncbi_esearch_url`, `ncbi_esummary_url`: NCBI 服务端点，通常不需要修改。
*   `ncbi_retry_attempts`, `ncbi_backoff_factor`: NCBI 客户端的重试策略。
*   `heuristic_filtering`:
    *   `allowed_species`: 允许的物种列表 (例如 `["Homo sapiens", "Mus musculus"]`)。
    *   `min_sample_count`: 数据集的最小样本数。
    *   `require_srp_id`: 是否必须包含 SRA 项目 ID。
    *   `match_experiment_type_keywords`: 用于匹配实验类型的关键词列表。
*   `cache_ttl_seconds`: 查询结果的缓存时间 (秒)。

## 5. 模块状态与未来工作

*   **当前状态**:
    *   核心工作流已实现，包括查询解析、NCBI数据检索、启发式过滤和LLM辅助的候选集分析。
    *   Pydantic数据模型、自定义异常和基础的NCBI客户端已完成。
    *   支持通过配置文件进行参数调整。
    *   实现了初步的内存缓存机制。
    *   已添加较为详细的日志记录。
*   **未来工作/待增强**:
    *   **单元测试**: 进一步完善 `test_strategy_service.py` 中的单元测试，覆盖更多场景和错误处理。
    *   **NCBIClient 增强**:
        *   实现对 EFetch API 的更完整支持（如果需要除摘要外的更详细信息）。
        *   考虑对 `get_geo_summaries` 中的ID列表进行分块处理，以避免超长URL或请求过载。
    *   **LLM 交互优化**:
        *   针对 `LLMOutputValidationError` 实现更智能的修正请求循环（如果LLM输出格式轻微错误）。
        *   根据实际效果调整和优化 `prompts_m1.toml` 中的 Prompt 模板。
    *   **缓存增强**: 考虑将内存缓存替换为更持久和可扩展的方案，如 Redis。
    *   **错误处理细化**: 根据实际运行中遇到的问题，进一步细化错误分类和处理逻辑。
    *   **与核心调度器集成**: 明确与核心调度器的交互点，特别是 `HITL-Checkpoint-1` 的触发。

---
*注意：本文档描述了模块的当前设计和实现状态，将随着AIBioFlow项目的进展而持续更新。*
