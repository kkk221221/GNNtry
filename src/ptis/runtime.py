"""运行时辅助函数，负责从环境变量构建 PTIS 配置。"""

from __future__ import annotations

import os
from .config import CacheConfig, ObservabilityConfig, PTISConfig, QwenProviderConfig


def load_env_config(
    *,
    api_key_env: str = "DASHSCOPE_API_KEY",
    timeout_env: str = "DASHSCOPE_TIMEOUT_S",
    retries_env: str = "DASHSCOPE_MAX_RETRIES",
) -> PTISConfig:
    """根据环境变量构建 :class:`PTISConfig`。

    - ``DASHSCOPE_API_KEY``（或 ``api_key_env`` 指定的变量）必须存在；
    - ``DASHSCOPE_TIMEOUT_S`` 可以覆写默认超时；
    - ``DASHSCOPE_MAX_RETRIES`` 可以覆写最大重试次数。
    """

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"环境变量 {api_key_env} 未设置，无法访问 DashScope API。"
        )

    timeout_s = _read_float_env(timeout_env, default=30.0)
    max_retries = _read_int_env(retries_env, default=3)

    qwen_config = QwenProviderConfig(
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    return PTISConfig(
        qwen=qwen_config,
        observability=ObservabilityConfig(enable_tracing=False, enable_metrics=False),
        cache=CacheConfig(enabled=False),
    )


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - 配置错误是运行时问题
        raise RuntimeError(f"环境变量 {name} 必须是浮点数") from exc


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError(f"环境变量 {name} 必须是整数") from exc
    if parsed < 0:
        raise RuntimeError(f"环境变量 {name} 必须是非负整数")
    return parsed
