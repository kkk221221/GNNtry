# LLM Gateway (`aibiowflow.llm_gateway`) Documentation

## 1. Overview

The `aibiowflow.llm_gateway` module serves as a centralized and standardized interface for interacting with various Large Language Models (LLMs). It abstracts the complexities of individual LLM provider APIs, offering a unified approach to configuration, prompt management, API calls, response handling, and error management. This gateway is designed to be extensible, allowing for the addition of new LLM providers with minimal effort.

## 2. Features

*   **Centralized LLM Access**: Provides a single `LLMGateway` class as the entry point for all LLM interactions.
*   **Multi-Provider Support**: Easily switch between different LLM providers (e.g., Google Gemini, Alibaba Qwen) through configuration.
*   **Configuration Management**:
    *   Externalized configuration via `config.yaml` for API keys, model definitions, and retry policies.
    *   Secure API key handling by referencing environment variables.
    *   Provider-specific settings can be configured.
*   **Prompt Templating**:
    *   Manages prompt templates through a `prompts.toml` file.
    *   Supports dynamic filling of templates with context variables.
*   **Model Aliasing**: Allows defining user-friendly aliases for specific provider models, making it easy to switch underlying models without code changes.
*   **Robust API Calls**:
    *   Implements an automatic retry mechanism for transient API errors with configurable backoff strategies.
*   **Structured Output**:
    *   Supports requesting JSON-formatted outputs from LLMs.
    *   Validates structured responses against Pydantic schemas to ensure data integrity.
*   **Customizable Logging**: Detailed logging of requests, responses, errors, and performance metrics, with support for both human-readable and structured (JSON) log formats.
*   **Standardized Error Handling**: A clear hierarchy of custom exceptions for predictable error management.
*   **Extensibility**: Designed with a `BaseProvider` abstraction to easily integrate new LLM providers.

## 3. Configuration

The LLM Gateway relies on two main configuration files: `config.yaml` and `prompts.toml`.

### 3.1. `config.yaml`

This YAML file manages the core settings for the gateway.

```yaml
# API 密钥配置
api_keys:
  google_gemini: "env:GEMINI_API_KEY"
  qwen: "env:DASHSCOPE_API_KEY"
  # anthropic_claude: "env:ANTHROPIC_API_KEY_DUMMY" # Example for another provider

# Provider 特定配置 (可选)
provider_specific_configs:
  google_gemini: # Matches the key in api_keys or the provider's PROVIDER_ID
    safety_settings:
      HARM_CATEGORY_HARASSMENT: "BLOCK_NONE"
      # HarmCategory and HarmBlockThreshold string names
    default_model_kwargs: # Default parameters for GenerationConfig
      top_p: 0.9
  qwen:
    # base_url: "https://custom-qwen-compatible-url/v1" # If not using default
    default_model_kwargs: # Default parameters for OpenAI client create call
      top_p: 0.85

# 模型定义
models:
  - name: "gemini-1.5-pro-latest"      # Actual model name used by the provider
    provider: "google"                # Corresponds to provider's PROVIDER_ID
    aliases: ["smart_model", "default_text_model"] # User-friendly aliases
  - name: "gemini-1.5-flash-latest"
    provider: "google"
    aliases: ["fast_model"]
  - name: "qwen-plus"
    provider: "qwen"
    aliases: ["qwen_plus_model"]
  # ... more models

# 重试策略配置
retry_policy:
  max_retries: 3
  backoff_factor: 2.0
  initial_wait_seconds: 1.0
  max_wait_seconds: 60.0
```

**Key Sections:**

*   `api_keys`:
    *   Maps provider identifiers (e.g., `google_gemini`, `qwen`) to their API keys.
    *   It's highly recommended to use the `env:YOUR_ENV_VARIABLE_NAME` syntax to load keys from environment variables for security.
*   `provider_specific_configs` (Optional):
    *   Allows passing specific configurations to individual providers during their initialization.
    *   The key (e.g., `google_gemini`) should match the provider's unique ID or the key used in `api_keys`.
    *   Refer to individual provider documentation for supported options (e.g., `safety_settings` for Gemini, `base_url` for Qwen).
*   `models`:
    *   A list defining available models. Each entry includes:
        *   `name`: The actual model identifier used by the LLM provider (e.g., "gemini-1.5-pro-latest").
        *   `provider`: The identifier of the LLM provider (must match a key in `api_keys` and the `PROVIDER_ID` of a registered provider class).
        *   `aliases`: A list of strings representing user-friendly names to refer to this model in your code. Aliases should be unique across all model definitions.
*   `retry_policy`:
    *   `max_retries`: Maximum number of retry attempts after an initial failed call.
    *   `backoff_factor`: Multiplier for increasing wait time between retries (e.g., 2 means wait times double).
    *   `initial_wait_seconds`: Base wait time before the first retry.
    *   `max_wait_seconds`: Maximum cap for a single retry wait interval.

### 3.2. `prompts.toml`

This TOML file stores named prompt templates.

```toml
[propose_strategy]
# Template for converting a research question to NCBI GEO search terms.
# Variables: user_query (the user's original question)
# Expected output: JSON format with 'pubmed_query' and 'analysis_type' fields.
template = """
将以下科研问题转化为NCBI GEO数据库的搜索关键词。
问题: "{user_query}"
请以JSON格式返回，包含'pubmed_query'和'analysis_type'字段。
"""

[generate_samplesheet]
# Template for creating a Samplesheet CSV for Nextflow RNA-Seq pipeline.
# Variables: file_list (a string describing the list of files in a specific format)
# Expected output: Raw CSV text.
template = """
我正在使用Nextflow流程'nf-core/rnaseq'。请为我创建一个samplesheet.csv。
必需的列是'sample', 'fastq_1', 'fastq_2', 'strandedness'。
'strandedness'列可设为'auto'。
我的文件列表如下：
{file_list}
请仅输出CSV格式的原始文本。
"""

[example_summarize_text]
template = "请总结以下文本：\n\"{text_to_summarize}\""
```

**Structure:**

*   Each prompt template is defined under a unique key (e.g., `propose_strategy`).
*   The `template` field contains the actual prompt string.
*   Placeholders for dynamic content are specified using curly braces (e.g., `{user_query}`). These are filled using Python's string formatting capabilities.

## 4. Core Components

### 4.1. `LLMGateway` Class

This is the main class for interacting with the LLM gateway.

**Initialization:**

```python
from aibiowflow.llm_gateway import LLMGateway

gateway = LLMGateway(
    config_path="path/to/your/config.yaml",
    prompt_path="path/to/your/prompts.toml"
)
```

Upon initialization, the gateway loads configurations, initializes available providers (those with valid API keys), and builds the model alias map.

**Methods:**

*   **`get_text_response(prompt_name: str, context: Dict[str, Any], model_alias: str = "fast_model", **provider_kwargs: Any) -> str`**
    *   Retrieves and formats a prompt template specified by `prompt_name` using the `context` dictionary.
    *   Selects an LLM based on the `model_alias`.
    *   Sends the request to the chosen LLM provider.
    *   Returns the LLM's text response as a string.
    *   `provider_kwargs`: Allows passing additional, provider-specific parameters directly to the LLM API call (e.g., `temperature`, `top_p`, `max_tokens`).

    ```python
    summary = gateway.get_text_response(
        prompt_name="example_summarize_text",
        context={"text_to_summarize": "Some long text here..."},
        model_alias="fast_model", # Defined in config.yaml
        temperature=0.5
    )
    print(summary)
    ```

*   **`get_structured_response(prompt_name: str, context: Dict[str, Any], output_schema: Type[BaseModel], model_alias: str = "smart_model", **provider_kwargs: Any) -> BaseModel`**
    *   Similar to `get_text_response`, but is used when a structured (JSON) response is expected.
    *   The `output_schema` parameter must be a Pydantic `BaseModel` class that defines the expected JSON structure.
    *   The gateway instructs the LLM (via prompt engineering and/or provider features) to return JSON.
    *   The raw JSON string from the LLM is then parsed and validated against the `output_schema`.
    *   Returns an instance of the `output_schema` Pydantic model.
    *   Raises `LLMOutputValidationError` if the LLM output is not valid JSON or does not conform to the schema.

    ```python
    from pydantic import BaseModel
    from typing import List

    class PaperInfo(BaseModel):
        title: str
        authors: List[str]
        year: int

    paper_details = gateway.get_structured_response(
        prompt_name="propose_strategy", # Assuming this prompt asks for JSON
        context={"user_query": "Find papers on LLMs in bioinformatics"},
        output_schema=PaperInfo,
        model_alias="smart_model",
        temperature=0.1 # Lower temperature often better for JSON
    )
    print(f"Title: {paper_details.title}")
    for author in paper_details.authors:
        print(f"Author: {author}")
    ```

### 4.2. Provider System (`providers/`)

The gateway uses an extensible provider system to interact with different LLM APIs.

*   **`BaseProvider` (Abstract Base Class):** Defines the common interface that all concrete provider classes must implement. Key methods include `generate_text()` and `generate_structured_text()`, and a `provider_name` property.
*   **`ProviderResponse`:** A data class that standardizes the response received from any provider. It includes:
    *   `text_content` (str): The main text output from the LLM.
    *   `input_tokens` (Optional[int]): Number of tokens in the input prompt.
    *   `output_tokens` (Optional[int]): Number of tokens in the generated response.
    *   `cost` (Optional[float]): Estimated cost of the API call (if available).
    *   `raw_response` (Optional[Any]): The original, raw response object from the provider's SDK (for debugging or advanced use).
    *   `model_name` (Optional[str]): The actual model name used for the call.
*   **Concrete Providers:**
    *   **`GeminiProvider`**: Interacts with Google Gemini models.
        *   Supports Gemini-specific configurations like `safety_settings` and `GenerationConfig` parameters passed via `provider_kwargs` or `provider_specific_configs`.
    *   **`QwenProvider`**: Interacts with Alibaba Cloud's Qwen (通义千问) models using their OpenAI-compatible API.
        *   Supports OpenAI-style parameters (e.g., `system_prompt`, `messages_override` in `provider_kwargs`).
        *   Can request JSON output directly using `response_format={"type": "json_object"}` (handled internally when `get_structured_response` is called).

### 4.3. Exception Handling

The gateway defines a set of custom exceptions for more specific error handling:

*   **`LLMGatewayError(Exception)`**: Base class for all gateway-specific errors.
*   **`LLMAPIError(LLMGatewayError)`**: Raised for issues during communication with the LLM API (e.g., network errors, authentication failures, rate limits, server errors).
    *   `status_code` (Optional[int]): HTTP status code if available.
*   **`LLMOutputValidationError(LLMGatewayError)`**: Raised when the LLM's output for a structured response request is not valid JSON or does not conform to the provided Pydantic schema.
    *   `validation_errors` (Optional[list | dict]): Detailed validation errors from Pydantic.
*   **`PromptTemplateError(LLMGatewayError)`**: Raised if a prompt template cannot be found or if there's an error formatting it (e.g., missing context variables).
    *   `template_name` (Optional[str]): The name of the problematic template.
*   **`ConfigurationError(LLMGatewayError)`**: Raised for errors in loading or parsing `config.yaml` or `prompts.toml`, or if the configuration content is invalid.
    *   `config_file_path` (Optional[str]): Path to the problematic configuration file.

**Example Error Handling:**

```python
from aibiowflow.llm_gateway import LLMGateway, LLMAPIError, LLMOutputValidationError, PromptTemplateError

try:
    gateway = LLMGateway(config_path="config.yaml", prompt_path="prompts.toml")
    response = gateway.get_text_response(
        prompt_name="non_existent_prompt",
        context={},
        model_alias="fast_model"
    )
except PromptTemplateError as e:
    print(f"Prompt error: {e}")
except LLMAPIError as e:
    print(f"API error: {e} (Status: {e.status_code})")
except LLMOutputValidationError as e:
    print(f"Output validation error: {e}")
    # print(f"Details: {e.validation_errors}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

## 5. Logging

The LLM Gateway implements comprehensive logging using Python's built-in `logging` module.

*   **Log Structure**: For each LLM request, the gateway logs:
    *   A human-readable summary line (INFO/ERROR level).
    *   A detailed JSON object (DEBUG level) containing:
        *   Timestamp
        *   Log type (`text_response` or `structured_response`)
        *   Prompt name, model alias, actual model name
        *   Request duration (ms)
        *   Masked prompt content (to avoid logging sensitive data)
        *   Status (success/failure)
        *   Token counts (input/output) and estimated cost (if available from provider)
        *   Error details (type, message, status code) if an error occurred.
        *   Target output schema name (for structured responses).
*   **Configuration**: Logging behavior (level, format, handlers) should be configured at the application level where the `LLMGateway` is used.

Example of setting up logging for the gateway:
```python
import logging

# Configure basic logging for the application
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose gateway logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Optionally, set a more specific level for the gateway's logger
logging.getLogger("aibiowflow.llm_gateway").setLevel(logging.DEBUG)

# ... then initialize and use the gateway
```

## 6. Extending the Gateway (Adding New Providers)

To add support for a new LLM provider:

1.  **Create a Provider Class**:
    *   In the `aibiowflow/llm_gateway/providers/` directory, create a new Python file (e.g., `my_new_provider.py`).
    *   Define a class that inherits from `BaseProvider`.
    *   Implement the required abstract methods:
        *   `__init__(self, api_key: str, provider_config: Optional[Dict[str, Any]] = None)`: Initialize the provider with API key and any specific configs.
        *   `generate_text(...) -> ProviderResponse`: Logic to call the new provider's API for text generation.
        *   `generate_structured_text(...) -> ProviderResponse`: Logic to call the new provider's API for structured (JSON) output. Ensure the response's `text_content` is a JSON string.
        *   `provider_name` (property): Return a unique string identifier for this provider (e.g., "my_new_llm"). This ID will be used in `config.yaml`.
    *   Handle API errors from the new provider's SDK and wrap them in `LLMAPIError`.
    *   Populate and return a `ProviderResponse` object.

2.  **Register the Provider**:
    *   In `aibiowflow/llm_gateway/gateway.py`, import your new provider class.
    *   Add it to the `supported_provider_classes` dictionary within the `LLMGateway._init_providers()` method:
        ```python
        # In LLMGateway._init_providers():
        from .providers import GeminiProvider, QwenProvider, MyNewProvider # Add import

        supported_provider_classes: Dict[str, Type[BaseProvider]] = {
            GeminiProvider.PROVIDER_ID: GeminiProvider,
            QwenProvider.PROVIDER_ID: QwenProvider,
            MyNewProvider.PROVIDER_ID: MyNewProvider, # Add your new provider
            # ...
        }
        ```

3.  **Update Configuration (`config.yaml`)**:
    *   Add an entry for the new provider's API key under `api_keys`:
        ```yaml
        api_keys:
          # ... other keys ...
          my_new_llm: "env:MY_NEW_LLM_API_KEY" # Use the provider_name as key
        ```
    *   If your provider has specific configurations, add a section under `provider_specific_configs`.
    *   Define models that use this new provider under the `models` section:
        ```yaml
        models:
          # ... other models ...
          - name: "actual-model-name-for-new-provider"
            provider: "my_new_llm" # Must match PROVIDER_ID
            aliases: ["alias_for_new_model"]
        ```

4.  **Environment Variable**: Ensure the corresponding environment variable (e.g., `MY_NEW_LLM_API_KEY`) is set with the actual API key.

## 7. Example Usage (Conceptual)

```python
# main_script.py
import logging
from pydantic import BaseModel
from typing import List

from aibiowflow.llm_gateway import LLMGateway
from aibiowflow.llm_gateway.exceptions import LLMGatewayError

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# For detailed gateway logs:
# logging.getLogger("aibiowflow.llm_gateway").setLevel(logging.DEBUG)


# Define a Pydantic model for structured output
class AnalysisResult(BaseModel):
    analysis_id: str
    status: str
    findings: List[str]
    confidence_score: float


def run_bio_analysis_pipeline():
    try:
        # Initialize the gateway
        # Ensure config.yaml and prompts.toml are correctly set up
        gateway = LLMGateway(config_path="./config.yaml", prompt_path="./prompts.toml")

        # --- Example 1: Get a simple text response (e.g., code generation/explanation) ---
        code_explanation_prompt_context = {
            "code_snippet": "def hello():\n  print('Hello, world!')",
            "language": "Python"
        }
        # Assuming a prompt like "explain_code" exists in prompts.toml:
        # template = "Explain the following {language} code snippet:\n```\n{code_snippet}\n```"
        # explanation = gateway.get_text_response(
        #     prompt_name="explain_code",
        #     context=code_explanation_prompt_context,
        #     model_alias="fast_model" # Choose an appropriate model alias
        # )
        # print("Code Explanation:\n", explanation)

        # --- Example 2: Get a structured response (e.g., parsing experimental data) ---
        experimental_data_context = {
            "raw_data_text": "Patient ID: P001, Gene: BRCA1, Expression: Up, Significance: High",
            "analysis_type": "gene_expression_summary"
        }
        # Assuming a prompt "parse_experimental_data" exists, asking for JSON output
        # that matches the AnalysisResult Pydantic model.
        parsed_result = gateway.get_structured_response(
            prompt_name="parse_experimental_data", # This prompt must ask for JSON output
            context=experimental_data_context,
            output_schema=AnalysisResult,
            model_alias="smart_model" # A model good at following structured instructions
        )
        print(f"\nParsed Analysis (ID: {parsed_result.analysis_id}):")
        print(f"  Status: {parsed_result.status}")
        print(f"  Findings: {', '.join(parsed_result.findings)}")
        print(f"  Confidence: {parsed_result.confidence_score}")

    except LLMGatewayError as e:
        logging.error(f"LLM Gateway operation failed: {e}", exc_info=True)
    except FileNotFoundError:
        logging.error("Error: config.yaml or prompts.toml not found. Please check paths.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    run_bio_analysis_pipeline()
```

This documentation provides a comprehensive guide to understanding, configuring, and using the `aibiowflow.llm_gateway` module.
