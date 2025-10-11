# PTIS (Parallel Thinking Inference System)

该项目基于 `docs/qwen_ptis_overview.md` 中的方案实现了一个面向通义千问 Qwen API 的并行推理服务。核心特性包括：

- **编排与路由**：`Planner` 模块按照任务类型、风险等级和预算生成分支/回合配置，并在模型族之间智能路由。
- **并行推理引擎**：`ParallelReasoningEngine` 负责分支生成、自检/互评、预算管理与裁决，支持工具调用证据化。
- **工具体系**：内置 RAG 和受限 Python 沙箱，可被分支独立调用，为推理提供可验证证据。
- **安全与治理**：`SafetyCenter` 对输入输出与内部思考进行审计，满足高风险场景的安全要求。
- **可观测性**：轻量级 `Tracer` 与 `MetricsCollector` 提供链路追踪与指标采集，支撑调试与评估。
- **状态与缓存**：`StateStore` 记录推理过程，`TTLCache` 复用中间结果以降低延迟与成本。
- **HTTP API**：`ptis.api` 基于标准库实现 `/v1/infer` 与 `/v1/traces/{id}` 接口，便于与现有系统集成。

## 目录结构

```
src/ptis/
  adapters/        # Qwen API 适配层
  reasoning/       # 并行推理核心组件
  tools/           # RAG、Python 沙箱等工具
  api.py           # HTTP 服务入口
  orchestrator.py  # 端到端编排器
  ...
```

## 本地运行

1. 安装 Python 3.11 及以上版本。
2. 设置环境变量 `PYTHONPATH=src`。
3. 运行测试套件：

   ```bash
   PYTHONPATH=src python -m unittest discover tests
   ```

4. （可选）使用真实 DashScope API 执行一次联调：

   ```bash
   export DASHSCOPE_API_KEY="<你的 DashScope API Key>"
   # 可选：覆写使用的模型，默认 qwen-plus
   export PTIS_MODEL="qwen-plus"
   PYTHONPATH=src python examples/run_real_infer.py
   ```

5. 启动 HTTP 服务（可选）：

   ```bash
   PYTHONPATH=src python -c "from ptis.api import run_server, PTISRequestHandler; from ptis.config import PTISConfig, QwenProviderConfig; server = run_server(PTISConfig(qwen=QwenProviderConfig(api_key='YOUR_KEY'))); input('Press Enter to stop...'); server.shutdown()"
   ```

> **提示**：默认单元测试会覆盖调用逻辑，但不会访问真实 Qwen API。若需自定义联机推理，可参考 `examples/run_real_infer.py`。
