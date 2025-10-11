# 概述（Qwen API 版本）

面向“并行思考 / Parallel Thinking（Deep Think/并行测试时计算, PTC）”范式，现将 **PTIS**（Parallel Thinking Inference System）落地为**优先调用通义千问 Qwen 的 API**的工程化方案。该版本：

* 以 **DashScope（阿里云百炼）OpenAI 兼容接口** 为首选接入方式，可一键迁移既有 OpenAI SDK 代码；
* 根据任务类型自动在 **qwen-plus/qwen-max/qwen-turbo**、**qwen-math-*****、qwen-coder-*****、qwen-vl-***、**qwen-long**、以及**QwQ（推理加强）**等模型族间路由；
* 在系统层实现“**多分支并行—互评—修订—裁决**”，必要时结合 Qwen3 的 `enable_thinking`（仅在支持的模型上）获取更稳健的中间推理，但**默认不对终端返回“思维过程”文本**（安全与合规）。

---

# 1. 设计目标与非功能需求

**功能目标**

* 并行推理：自适应分支数 `b` 与回合 `r`，闭环“生成→批评→修订→裁决”。

* 工具增强：每条分支均可独立调用 RAG、代码沙箱、计算器、联网搜索、数据库/Graph 查询。

* 成本/延迟约束：在 `latency_budget_ms` 与 `cost_cap_usd` 内进行预算分配、早停、分层并行。

* 可观测与评测：全链路 Trace、分支级指标、离线/在线评测与回放。

**非功能需求**

* 可伸缩：无状态编排 + 有状态存储；分支执行器水平扩展。
* 弹性与容错：超时降级、跨地域自动回退（北京/新加坡/金融云），断路器与速率控制。
* 数据保护：最小化收集，PII 脱敏/加密，策略化留存；可选专有域名与 VPC egress 控制。

---

# 2. 总体架构

```
+------------------+          +-------------------+        +------------------+
|  客户端/上游API  |--HTTP--> |  API Gateway      | -----> |  AuthN/AuthZ     |
+------------------+          +-------------------+        +------------------+
                                         |                         \
                                         v                          \
                               +-------------------+                 \
                               |  Orchestrator     | <----(Policy)----+
                               |  (编排/治理)      |-----------------+
                               +---------+---------+                 |
                                         |                           |
                +------------------------+-----------------------+   |
                v                                                v   v
      +---------------------+                         +----------------------+
      | Planner & Router    |                         | Safety & Governance  |
      |(任务分解/自适应并行)|                         |(输入/中间/输出审计) |
      +----------+----------+                         +----------+-----------+
                 |                                                    |
                 v                                                    v
        +--------+------------------------------------+   +-------------------+
        |      Parallel Reasoning Engine (PRE)        |   | Observability &   |
        |  - Branch Generator (多分支生成)            |   | Evaluation (OTel) |
        |  - Critics/Debaters (相互评析/辩论)        |   +-------------------+
        |  - Aggregators/Judges (裁决/合并)          |
        |  - Budget Manager (预算/早停/复用)          |
        +--------+------------------+-----------------+
                 |                  |
                 v                  v
       +----------------+   +----------------------------+
       | Tools & RAG    |   | Providers Layer (Qwen 优先) |
       |(检索/执行/计算)|   | - DashScope OpenAI 兼容     |
       +----------------+   | - DashScope SDK（可选）     |
                 |          | - 本地/开源兜底（Qwen 系列）|
                 v          +----------------------------+
          +-------------+     +------------+
          | State Store |     | Cache      |
          | (DAG/Trace) |     | (语义缓存) |
          +-------------+     +------------+
```

---

# 3. 关键子系统

## 3.1 API Gateway

* 统一入口，OpenAPI Schema 校验、认证、配额、速率限制；支持 `DASHSCOPE_API_KEY` 租户隔离与地域选择。
* 关键参数（示例）：

  * `task_type`、`risk_level`、`latency_budget_ms`、`cost_cap_usd`、`parallelism_max`、`models_preferred[]`（Qwen 型号）、`tools_allowed[]`、`region`（`cn`/`intl`/`finance`）、`explain_level`、`trace_return`。

## 3.2 Planner & Router（任务分解与路由）

* **难度估计与并行判定**：轻量模型/规则综合，给出 `b0`、`r`、分支 token 预算 `T0`。
* **模型选型策略（Qwen）**：

  * 通用对话/综合推理：`qwen-plus` ⇒ 高质量；低延迟：`qwen-turbo`。
  * 数学：`qwen-math-plus`（或 `qwen2.5-math-*` 开源/私有化场景）。
  * 代码：`qwen-coder-plus` / `qwen-coder-turbo`。
  * 多模态视觉：`qwen-vl-plus`/`qwen-vl-max`（VL）。
  * 长文理解：`qwen-long`。
  * 强化推理：`qwq-plus`
* **地域与合规路由**：

  * **北京：**`https://dashscope.aliyuncs.com/compatible-mode/v1`
  * **金融云：**`https://dashscope-finance.aliyuncs.com/compatible-mode/v1`

## 3.3 Parallel Reasoning Engine（PRE）

**组成**

* **Branch Generator**：并行采样多条候选（差异化温度/系统指令/工具计划/提示风格）。
* **Critics/Debaters**：

  * *Self-critique* 自检（计算/引用/边界）。
  * *Cross-critique* 分支互评（结构化指出漏洞与修补）。
  * *Debate*（可选）多轮“论点—反驳—再论证”。
* **Aggregators/Judges**：以证据覆盖、逻辑一致、可验证性为准则，投票/评分/置信回归聚合；必要时 **progressive widening** 再展开。
* **Budget Manager**：

  * **Successive Halving（SH）**：初始 `b0` → 评分 Top-p% 追加预算 → 其余早停。
  * **UCB/Thompson**：将“提示×工具计划×模型”视为臂，在线分配预算。
  * **分层并行**：先小宽度 `b1` 探路，聚焦潜力臂纵向加深。
* **Tool Hooks**：每分支可调用 RAG/计算器/代码沙箱/联网搜索；工具结果作为“证据节点”入库（强约束：无证据的推断降权）。
* **Thinking 控制（可选）**：对 **Qwen3 系列**可通过 `extra_body.enable_thinking` 控制内部思考；**仅存档，不回传**给用户端输出。

## 3.4 Providers Layer（Qwen 专用抽象）

* **首选接入：OpenAI 兼容接口**

  * Base URL：见 3.2；Header：`Authorization: Bearer $DASHSCOPE_API_KEY`。
  * API：`POST /chat/completions`；支持 `stream`、`tools`（函数调用）、`tool_choice`。
* **可选接入：DashScope 原生 SDK**（Python/Java 等）用于异步任务或特定增强能力。
* **可靠性工程**：超时/重试/退避、429 限流处理、跨地域自动回退、模型降级（如 `qwen-plus` → `qwen-turbo`）。

## 3.5 Tools & RAG

* **RAG**：BM25+向量召回+重排序；引用对齐与证据评分；per-branch 检索域隔离。
* **执行工具**：Python 沙箱、数值/符号计算、网络检索、SQL/Graph 查询。
* **工具治理**：Schema 校验、单元测试化验证、速率与配额、审计日志。

## 3.6 Safety & Governance

* **三层审计**：输入预审（敏感/恶意）、中间产物审计（工具越权/数据外泄）、输出后审（有害/幻觉/泄漏）。
* **安全完成**：优先提供无害替代、注意事项和后续步骤。
* **Thinking 隐私**：若启用 `enable_thinking`，仅内部存档，不出现在 `answer`；对外仅给“可验证事实与证据指针”。

## 3.7 Observability & Evaluation

* **可观测**：OpenTelemetry Trace，分支/工具/Provider 作为 `span`；统一 `trace_id` 串接请求→分支→裁决。
* **指标**：正确率@场景、pass@k、成本/延迟分布、证据覆盖率、早停收益、工具成功率。
* **评测**：离线基准（数学/代码/多步推理/多模态）、在线 A/B（影子模式），以及回放测试（固定外部依赖时间线）。

## 3.8 State Store & Cache

* **State Store**：保存推理 DAG、提示/输出、工具证据、评分元数据；支持可重放（deterministic seed + provider snapshot）。
* **缓存**：语义缓存（查询/上下文）、片段缓存（子问题复用，多租户隔离）。

---

# 4. 端到端流程（E2E）

**输入**：`POST /v1/infer`

```json
{
  "task_type": "math_proof",
  "prompt": "证明对任意...",
  "latency_budget_ms": 12000,
  "cost_cap_usd": 0.35,
  "risk_level": "high",
  "parallelism_max": 6,
  "region": "intl",
  "models_preferred": ["qwen-math-plus", "qwq-plus", "qwen-plus"],
  "tools_allowed": ["rag", "python"],
  "explain_level": "trace_ptrs"
}
```

**时序（简化）**

1. Gateway 校验 → Orchestrator 生成 `trace_id`。
2. Planner 估计难度并给出 `(b0=4, r=2, T0=1200)` 与工具计划，选择 `qwq-plus` 或 `qwen-math-plus`。
3. PRE 第 0 轮：并行生成 4 条分支；必要时触发 RAG/计算；自检形成“错误/证据 TODO”。
4. 互评：两两指出漏洞与修补建议（结构化 JSON）。
5. 修订：对 Top-2 分支追加预算（SH 50% 筛），其余早停。
6. 裁决：评分器（覆盖/一致/可证/工具通过率），必要时再展开。
7. Safety：输出审计；若高风险，转安全完成模板。
8. 返回：答案 + 关键证据指针 + 可选简版推理摘要（不暴露思维文本）。

---

# 5. 预算与并行调度策略

**目标**：在 `latency_budget_ms` 与 `cost_cap_usd` 下最大化正确率。

**算法草案**

```text
输入：Bmax(并行上限), T(总token预算), r(最大回合)
初始化：b=b0, 为每分支分配 T0 = T/(b*r)
for round in 1..r:
  并行生成/修订所有存活分支 (各自用 T0 或加权 T_i)
  评分 = w1*逻辑一致 + w2*证据覆盖 + w3*可验证性 + w4*多样性
  存活 = Top-K 或分位数筛选
  若剩余预算 < 阈值或触发高置信：early stop
  自适应加宽：若分数接近且仍有预算 → 新增探索分支
输出：最佳分支或加权汇总
```

**实现建议**

* **UCB1/Thompson**：把“提示风格×工具计划×模型选型”当作拉臂，在线估计收益。
* **分层并行**：先 `b_small` 探索 → 聚焦潜力臂 → 纵向加深；保证低延迟场景体验。
* **早停阈值**：当“所有可验证证据”满足且互评无有效反驳，即刻返回。

---

# 6. Prompt 与协议（Qwen 适配）

**分支提示模版（简化）**

```
[系统] 你是严谨的推理专家。先计划→分步推理→必要时调用工具。
[约束] 不编造证据；缺证必须标注并触发检索/计算。
[目标] 解决子任务 {{subtask_id}}：{{subtask_text}}
[输出JSON]
{
  "plan": ["..."],
  "steps": [{"thought": "...", "tool_call?": {"name": "python|search|sql", "args": {}}}],
  "assumptions": ["..."],
  "intermediate_results": {"...": "..."},
  "provisional_answer": "...",
  "uncertainties": ["..."],
  "needs_more_evidence": true
}
```

**互评模版**

```
[任务] 评审分支 A 对子任务 {{sid}} 的方案。
[标准] 逻辑、证据、边界、可验证性。
[输出JSON] {"fatal_flaws": [...], "minor_issues": [...], "fix_suggestions": [...], "score": 0..100}
```

**裁决模版**

```
[任务] 汇总分支 {A,B,...} 的结论。
[要求] 仅使用经验证证据；结论不一致时给出分情形或加权结论。
[输出] 最终答案 + 关键证据引用（State Store 指针）
```

---

# 7. 外部 API 设计（OpenAPI 草案）

```yaml
paths:
  /v1/infer:
    post:
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferRequest'
      responses:
        '200': { $ref: '#/components/schemas/InferResponse' }
components:
  schemas:
    InferRequest:
      type: object
      properties:
        task_type: { type: string }
        prompt: { type: string }
        latency_budget_ms: { type: integer }
        cost_cap_usd: { type: number }
        risk_level: { type: string, enum: [low, medium, high] }
        parallelism_max: { type: integer, default: 4 }
        region: { type: string, enum: [cn, intl, finance] }
        models_preferred: { type: array, items: { type: string } }
        tools_allowed: { type: array, items: { type: string } }
        explain_level: { type: string, enum: [none, brief, trace_ptrs] }
    InferResponse:
      type: object
      properties:
        answer: { type: string }
        confidence: { type: number }
        trace_pointers: { type: array, items: { type: string } }
        cost_actual_usd: { type: number }
        latency_ms: { type: integer }
```

---

# 8. Provider 适配层（以 TypeScript/Node 为例）

**OpenAI 兼容：QwenAdapter**

```ts
import OpenAI from "openai";

export interface ChatOpts {
  model: string;           // 如 qwen-plus / qwen-math-plus / qwen-vl-plus
  messages: any[];         // OpenAI 兼容 messages
  temperature?: number;
  stream?: boolean;
  tools?: any[];           // OpenAI tools（函数调用）
  tool_choice?: string | {type: string, function?: {name: string}};
  extra_body?: Record<string, any>; // 如 { enable_thinking: true }
}

export class QwenAdapter {
  private client: OpenAI;
  constructor(apiKey: string, region: 'cn'|'intl'|'finance' = 'cn') {
    const baseURL = region === 'intl'
      ? 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
      : region === 'finance'
      ? 'https://dashscope-finance.aliyuncs.com/compatible-mode/v1'
      : 'https://dashscope.aliyuncs.com/compatible-mode/v1';
    this.client = new OpenAI({ apiKey, baseURL });
  }
  async chat(opts: ChatOpts) {
    return await this.client.chat.completions.create({
      model: opts.model,
      messages: opts.messages,
      temperature: opts.temperature,
      stream: opts.stream,
      tools: opts.tools,
      tool_choice: opts.tool_choice as any,
      extra_body: opts.extra_body as any,
    });
  }
}
```

**Python（流式 + enable_thinking 示例）**

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus",  # 或 qwq-plus / qwen-math-plus / qwen-coder-plus
    messages=[
        {"role": "system", "content": "You are a rigorous reasoner."},
        {"role": "user", "content": "请分步推理解此题，并在需要时调用工具。"},
    ],
    stream=True,
    # 对 Qwen3 系列模型可传：extra_body={"enable_thinking": True}
)
for chunk in resp:
    print(chunk.model_dump_json())
```

**Vision（Qwen-VL，多图输入）**

```python
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
completion = client.chat.completions.create(
  model="qwen-vl-plus",
  messages=[{
    "role": "user",
    "content": [
      {"type": "text", "text": "这些是什么？"},
      {"type": "image_url", "image_url": {"url": "https://.../dog_and_girl.jpeg"}},
      {"type": "image_url", "image_url": {"url": "https://.../tiger.png"}}
    ]
  }]
)
print(completion.model_dump_json())
```

---

# 9. 成本、延迟与可靠性工程

* **成本守护**：token→成本函数；调用前预扣预算；热点子任务 TTL 缓存；跨分支结果复用。
* **延迟控制**：分层并行与早停；检索/重排与 LLM 解耦；I/O 工具与 CPU 工具分池并发。
* **可靠性**：限流与断路器；429/5xx 重试与退避；跨地域健康探测与自动回退；模型级降级路径。

---

# 10. 可观测性与评测

* **Trace**：请求→分支→工具→裁决；记录 token、时延、状态、提示签名、地域/模型信息。
* **Dashboard**：正确率、pass@k、成本/延迟直方图、分支存活率、早停节省、工具成功率、地域回退统计。
* **评测**：离线（数学/代码/推理/多模态、不含污染）+ 在线（分流 A/B、影子流），回放（固定时间线）。

# 14. 路线图

* **v0（2–4 周）**：PRE 核心闭环；手动并行度；最小工具集；观测基础；Qwen `qwen-plus/qwen-turbo` 接入。
* **v1（6–10 周）**：自适应并行（SH/UCB）；RAG 强化；安全完成与策略中心；A/B 框架；缓存与回放；接入 `qwen-math-*/qwen-coder-*` 与 `qwen-long`。
* **v2（10–16 周）**：多模态（`qwen-vl-*`）；跨地域智能路由；成本-质量多目标优化；高风险域模板库；组织级合规与审计；可选 QwQ 推理模型集成。

---

# 15. MVP 清单（Qwen 版）

* **核心**：Orchestrator、Planner、PRE（Branch/Critic/Judge/Budget）、QwenAdapter、State Store、基础安全审计。
* **接口**：`/v1/infer`、`/v1/traces/{id}`（仅元数据/指针）。
* **运行**：DashScope OpenAI 兼容；至少一种数学/代码/通用模型；本地向量库 RAG；观测与日志。
* **评测**：小型公开集 + 自建业务集；SxS 与成本/延迟曲线。
