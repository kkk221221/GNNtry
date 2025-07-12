# -*- coding: utf-8 -*-
"""
Custom Exceptions for Module 1: Literature & Strategy Service.

These exceptions are specific to the operations and error conditions encountered
within this module.
"""

class StrategyCreationError(Exception):
    """
    基类异常，表示在文献与数据策略服务模块中生成分析方案时发生错误。
    所有本模块抛出的更具体的、可捕获的异常都应继承自此类。
    """
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception
        # 可以考虑添加其他通用属性，如错误发生的阶段、上下文信息等

    def __str__(self):
        if self.original_exception:
            return f"{super().__str__()} (Caused by: {type(self.original_exception).__name__}: {self.original_exception})"
        return super().__str__()


class NCBIAPIError(StrategyCreationError):
    """
    当与 NCBI (National Center for Biotechnology Information) 的 API 交互失败时抛出。
    这可能包括网络问题、API限流、无效的查询参数、或 NCBI 服务器端错误。
    """
    def __init__(self, message: str, status_code: int | None = None, original_exception: Exception | None = None):
        super().__init__(message, original_exception)
        self.status_code = status_code  # HTTP 状态码或其他相关的错误代码

    def __str__(self):
        base_message = super().__str__()
        if self.status_code:
            return f"NCBI API Error (Status: {self.status_code}): {base_message}"
        return f"NCBI API Error: {base_message}"


class NoDataFoundError(StrategyCreationError):
    """
    当在文献检索或数据筛选过程中，未能找到任何符合条件的候选数据集时抛出。
    这可以帮助上层调用者区分“没有结果”和“发生了错误”。
    """
    def __init__(self, message: str = "No suitable data found matching the criteria.", original_exception: Exception | None = None):
        super().__init__(message, original_exception)

    # __str__ 方法继承自 StrategyCreationError 即可


# 可以在此根据需要添加更多特定的异常类型，例如：
# class InvalidFilterConfigError(StrategyCreationError): ...
# class LLMCorrectionLoopError(StrategyCreationError): ...

[end of aibiowflow/module_1_strategy/exceptions.py]
