# -*- coding: utf-8 -*-
"""
LLM 网关模块的自定义异常。
"""

class LLMGatewayError(Exception):
    """LLM 网关模块所有自定义异常的基类。"""
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception
        # 可以添加更多通用属性，如错误代码等

class LLMAPIError(LLMGatewayError):
    """
    当与 LLM API 的交互失败时抛出。
    这可能包括网络问题、认证失败、API 限流、服务器端错误等。
    """
    def __init__(self, message: str, status_code: int | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.status_code = status_code # HTTP 状态码或其他错误代码

    def __str__(self):
        if self.status_code:
            return f"LLM API Error (Status {self.status_code}): {super().__str__()}"
        return f"LLM API Error: {super().__str__()}"

class LLMOutputValidationError(LLMGatewayError):
    """
    当 LLM 返回的输出未能通过预定义的结构（例如 Pydantic schema）验证时抛出。
    """
    def __init__(self, message: str, validation_errors: list | dict | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.validation_errors = validation_errors # Pydantic ValidationError.errors() 或类似结构

    def __str__(self):
        base_message = super().__str__()
        if self.validation_errors:
            return f"{base_message} Validation Details: {self.validation_errors}"
        return base_message

class PromptTemplateError(LLMGatewayError):
    """
    当 Prompt 模板加载、解析或格式化失败时抛出。
    """
    def __init__(self, message: str, template_name: str | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.template_name = template_name

    def __str__(self):
        if self.template_name:
            return f"Prompt Template Error (Template: {self.template_name}): {super().__str__()}"
        return f"Prompt Template Error: {super().__str__()}"


class ConfigurationError(LLMGatewayError):
    """
    当网关配置（config.yaml 或 prompts.toml）加载或解析失败，
    或配置内容不符合预期格式时抛出。
    """
    def __init__(self, message: str, config_file_path: str | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.config_file_path = config_file_path

    def __str__(self):
        if self.config_file_path:
            return f"Configuration Error (File: {self.config_file_path}): {super().__str__()}"
        return f"Configuration Error: {super().__str__()}"

# 可以在这里根据需要添加更多特定的异常类型，例如：
# class RateLimitError(LLMAPIError): ...
# class AuthenticationError(LLMAPIError): ...
# class ModelNotAvailableError(LLMAPIError): ...
