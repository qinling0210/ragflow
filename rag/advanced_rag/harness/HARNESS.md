# RAGFlow Agentic RAG Harness — 结构解析与 Go 移植指南

> 本文档描述 `rag/advanced_rag/harness/` 的行为、文件关系与关键数据流，
> 并给出将其移植到 Go（`internal/agent` 风格）时的包划分与实现要点。
> 所有结论均基于当前代码（2026-09），改动前请对照最新源码复核。

---

## 1. 概述

`harness` 是 RAGFlow Advanced RAG 的 **Agentic 编排层**（"引擎室"）。它把
"检索 → 判断 → 补充 → 回答"的循环拆成可复用零件：

- **决策循环**（`action_session.py`）：基于 LangGraph 的 ReAct 式行动会话，
  驱动 LLM 决策 + 工具调用，直到产出草稿或预算耗尽。
- **能力层**（`tools/`）：检索 / 导航 / 图谱探索 / 文本处理等工具实现。
- **编排零件**（`orchestrator/`）：直搜、充分性检查（SCA）、查询重写。
- **支撑模块**：配置、统计、关键词、算术、证据窄化、记忆、结构 QA。

外部入口是 `rag/advanced_rag/agentic_rag.py` / `agentic_rag_graph.py`，
它们按 `tools.thinking_mode`（由 `config.py` 解析）选择策略，再调用本目录。

---

## 2. 目录结构

```
harness/
├── action_session.py          # 核心：Agentic 行动会话（LangGraph ReAct 循环）
├── config.py                  # 思考模式配置（唯一权威：low/medium/high/ultra/naive）
├── keywords.py                # 加权关键词抽取（entity x3 / qualifier x3）
├── arithmetic.py              # 安全计算工具（AST 白名单求值）
├── grep_sed_narrow.py         # grep/sed 式证据窄化（零额外 LLM 调用）
├── memory.py                  # 检索记忆（会话内原始 chunk 缓存 + 相关度检索）
├── stats.py                   # 阶段计时 + LLM 用量统计（CountingChatModel 代理）
├── structure_qa.py            # 基于结构图（entities+relations）的 QA 助手
├── chunk_utils.py             # chunk 字段访问器（别名归一，str/int 去重一致）
├── prompts/
│   ├── __init__.py
│   └── report_prompt.py       # 报告生成提示词（FINAL_ANSWER_SYSTEM）
├── orchestrator/              # 顶层编排零件（被外部 agentic 图调用）
│   ├── __init__.py
│   ├── direct.py              #   low 模式：单次直搜
│   ├── query_rewriter.py      #   SCA gap → 新检索查询
│   └── sufficient_context.py  #   统一充分性检查代理（SCA）
└── tools/                     # 行动会话可绑定的工具实现
    ├── __init__.py
    ├── search.py              #   hybrid/vector/bm25/web/grep/list_chunks/structured
    ├── navigation.py          #   数据集导航树 + 文档结构导航（TOC 钻孔）
    ├── exploration.py         #   graph_explore（KG 广度优先）+ wiki_query
    ├── compiled_expansion.py  #   编译结构扩展（page_index/tree/KG/wiki 合成页）
    └── text_processing.py     #   句子切分/词干化/关键词窄化/高亮
```

---

## 3. 核心数据流

```
外部 agentic_rag_graph / agentic_rag（harness 之外）
   │  按 tools.thinking_mode → config.resolve_mode → ModeSpec
   ├─ low  ────────────────→ orchestrator/direct.py（单次 hybrid_search）
   └─ medium/high/ultra ───→ action_session.run_action_session
          │
          ├─ 0) initialize_state     槽位表初始化（LLM 分解问题 → slots + first_queries）
          ├─ 1) run_nav_prefix       导航链预跑（AUTO 步骤代码化，注入历史）
          ├─ 2) LangGraph 循环
          │      _run_action_node  ──LLM 决策──▶ tool_calls | <state>|<answer>
          │           │
          │           ▼
          │      _tool_node  ──execute_tool──▶ tools/*（dispatch by name）
          │           │        ├─ 近重复查询抑制（Jaccard 去重）
          │           │        ├─ 同参数工具缓存
          │           │        ├─ 状态策略（OK/EMPTY/MISS/POOR/REDUNDANT/ERROR）
          │           │        └─ 编译工具"strike" 禁用机制
          │           ▼
          │      _finalize_node  预算耗尽 → 无工具收尾调用 + 松线索收割
          │           │
          │           ▼
          │      _route / _route_after_tool（轮次预算 / 收敛判断）
          │
          ├─ 3) 产出草稿 claims（写入 tools.kbinfos）
          └─ 4) 外部再调 orchestrator/sufficient_context.py
                 └─ 不充分 → orchestrator/query_rewriter.py → 下一轮
```

共享状态载体：**`tools.kbinfos`**（chunks / doc_aggs / memory 等），
贯穿检索、行动会话、SCA、终答全流程。

### 3.1 模块依赖图（内外双向）

**内部依赖**（箭头 = 导入方向；`tools/navigation.py` 与 `tools/exploration.py`
存在循环导入，规避方式不对称：**navigation→exploration 是模块级 re-export**
（navigation.py:760 `from ...tools.exploration import (...)`），而
**exploration→navigation 是函数体内惰性导入**（exploration.py:351））：

```
action_session.py
   ├─ config.py                resolve_mode → ModeSpec（4.1）
   ├─ tools/search.py          grep/hybrid/list_chunks/_chunk_id/_base_chat_mdl…
   ├─ tools/navigation.py      _navigate_tree_impl / _navigate_structure_impl / graph_explore
   ├─ arithmetic.py            compute_from_facts
   └─ stats.py                 record_external_response（action_session.py:1067/1086）

tools/search.py
   ├─ tools/text_processing.py     句子切分/词干化/窄化/高亮
   ├─ tools/compiled_expansion.py  编译结构扩展（1 跳实体图 + 合成页）
   ├─ memory.py                    检索结果写入记忆
   └─ grep_sed_narrow.py           grep/sed 证据窄化

tools/navigation.py
   ├─ chunk_utils.py               字段别名归一
   └─ tools/exploration.py         graph_explore / _kg_scopes（模块级 re-export）

tools/exploration.py
   ├─ structure_qa.py              _ask_structure（结构大纲判定，exploration.py:27 模块级导入）
   └─ tools/navigation.py          _doc_aggs / _load_chunks_by_ids（函数体内惰性导入，exploration.py:351）

orchestrator/{direct,query_rewriter,sufficient_context}.py
   ├─ tools/search.py              hybrid_search
   └─ stats.py                     in_phase（direct.py:5 / query_rewriter.py:25 / sufficient_context.py:35）
```

**向外依赖**（harness → 上层 rag / 基础设施模块）：

| 被依赖方 | 用途 |
|---|---|
| `common.settings.retriever`（全局单例） | 所有检索后端（hybrid/vector/bm25） |
| `rag.prompts.template.load_prompt` / `rag.prompts.generator.gen_json` / `PROMPT_JINJA_ENV` | 提示词加载与 JSON 生成（query_rewriter / sufficient_context） |
| `rag.llm.tool_decorator.tool` | `@tool(timeout=120)` 装饰（navigation 工具） |
| `api.db.services.dataset_api_service` | 导航树 / 编译结构 / doc 元数据读取 |
| `json_repair` | 容错 JSON 解析（structure_qa / navigation） |
| `thread_pool_exec` | 阻塞式 doc store 调用的线程池封装 |

**外部调用者**（谁调用 harness）：

```
api/db/services/dialog_service.py        ← Chat API（RAGTools，用户真实入口）
   └─ rag/advanced_rag/agentic_rag.py            RAGTools.answer（low 直搜 + SCA 收尾）
        └─ rag/advanced_rag/agentic_rag_graph.py      run_agentic_rag（planner/fanout 展开）
             └─ harness/*（resolve_mode → orchestrator/direct 或 action_session）
test/unit_test/rag/advanced_rag/*.py      ← pytest（RAGTools 行为测试，harness 的间接消费者）
```

### 3.2 边界数据契约（对外 DTO，Go struct 需逐字段对齐）

harness 的对外边界是**函数签名**，不是 HTTP/DTO：

| 函数 | 入参 | 出参 / 副作用 |
|---|---|---|
| `config.resolve_mode(tools)` | `RAGTools`（读 `tools.thinking_mode`） | `ModeSpec`（frozen dataclass，见 4.1） |
| `action_session.run_action_session(tools, direction, parent_state, deadline_left=None, base_summary="", shared_tool_cache=None)` | 问题 + 已分解 direction（+ 截止剩余时间 / 汇总草稿 / 工具缓存） | `Result`（new_states / found_answer / messages / evidence ids） |
| `action_session.initialize_state(tools, question, fanout_hint, deadline_left=None)` | 问题 + fanout 提示（+ 截止剩余时间） | `list[Variable]`（槽位表） |
| `orchestrator/direct.py: direct_search(...)` | 问题 | 写 `tools.kbinfos`，无结果标记 `empty_result` |
| `orchestrator/sufficient_context.py: sufficient_context_agent(...)` | 各 claim 草稿 + 证据锚点 + 整体草稿 | verdict（is_sufficient/confidence/contradictions/claims/sub_queries） |
| `orchestrator/query_rewriter.py: rewrite_gap_to_query(...)` | forward gap（what+hint） | 新查询；失败返回空表由调用方回退原 gap |

内部核心结构（不跨 harness 边界，但 Go 移植需照搬语义）：
`Variable` / `State` / `Result` / `ToolOutcome(status,reason,metrics)` /
`_SessionState` / `NavResult`；`ToolOutcome.status ∈ {OK,EMPTY,MISS,POOR,REDUNDANT,ERROR}`。

### 3.3 控制流细节：超时 / 重试 / 日志钩子

| 机制 | 位置 | 行为 |
|---|---|---|
| 工具超时 | `@tool(timeout=120)`（navigation 两个 impl） | 120s 上限，超时视为工具失败 |
| 温和重试 | `grep_sed_narrow.py` | 主术语无命中 → 兜底术语机械重试一次（零 LLM） |
| LLM 失败回退 | `arithmetic.compute_from_facts` / `keywords.py` | LLM 异常或 JSON 解析失败 → 返回空值，不中断会话 |
| 兜底构造 | `action_session.initialize_state` | LLM 分解失败 → 用 fanout hint 直接构造槽位表 |
| 异常分支 | 各编排函数 | 全部 `except Exception: _LOG.info/exception` 吞掉，保证单轮失败不炸掉整个会话 |
| 日志钩子 | 每模块 `_LOG = logging.getLogger(__name__)` | `[Hybrid search]` / `[Direct search]` / `[Memory]` / `[SCA]` / `[Compute]` 等前缀 |
| 统计代理 | `stats.CountingChatModel` | 代理 LLM 调用并计数（usage 归入 `LLMUsageStats`），阶段计时由 `in_phase` 记录 |

---

## 4. 模块明细

### 4.1 `config.py` — 模式配置（单一权威）

定义 `ModeSpec`（frozen dataclass），集中 5 个可变维度：

| 字段 | 含义 |
|---|---|
| `label` | 模式自身名称（`low`/`medium`/`high`/`ultra`/`naive`） |
| `agentic` | 是否跑 agentic 图（false → 直搜） |
| `enable_sca` | 是否启用充分性检查循环 |
| `sca_max_rounds` | SCA→rewriter 迭代预算 |
| `use_fanout` | 是否启用 planner + 多槽分解 |
| `action_max_turns` | 单行动会话内轮次预算 |
| `tools` | 模型可见工具集（frozenset[str]） |

模式矩阵（`THINKING_MODES`）：

| 模式 | agentic | SCA | fanout | action_max_turns | 工具 |
|---|---|---|---|---|---|
| `low` | false | off | off | 4 | （无） |
| `medium` | true | 3 轮 | off | 4 | `_ALL_TOOLS`（7 个） |
| `high` | true | 3 轮 | on | 4 | `_ALL_TOOLS` |
| `ultra` | true | 5 轮 | on | 6 | `_ALL_TOOLS` + `graph_explore` |
| `naive`（未知标签回退） | false | off | off | 4 | （无） |

`_ALL_TOOLS` = retrieve, search_chunks, list_chunks, navigate_tree,
navigate_structure, calculate, web_search。

**Go 移植要点**：等价于一个 `ModeSpec` struct + 查表；`resolve_mode(tools)`
读 `tools.thinking_mode` 字符串。建议在 Go 侧定义 `ModeSpec` + `GetMode(label)`，
未知标签同样回退 naive，绝不 fail request。

---

### 4.2 `action_session.py` — 行动会话（核心，约 2100 行）

#### 4.2.1 数据结构

| 类型 | 作用 |
|---|---|
| `Variable` | 一个待求解槽位：`id`（不可变）、`type`、`question_clues`、`discovered_clues`、`candidate`、`candidate_strength` |
| `State` | 会话状态：`state: list[Variable]`、`depth`、`id`、`retrieved_evidence_ids`；由模块级函数 `apply_patch(base, branch_patches)`（action_session.py:574）修改——只允许改可变字段，不允许新增变量 |
| `Result` | 一次 `run_action_session` 的产出（new_states / found_answer / messages / evidence ids） |
| `ToolOutcome` | 一次工具调用的结构化结果：`payload`（模型可见）、`evidence_ids`、`status`、`reason`、`metrics` |
| `_SessionState` | LangGraph 状态（messages / tools / mdl / 预算 / 各种计数器） |

`ToolOutcome.status` 语义（这是策略决策的关键）：

| 状态 | 含义 | 触发动作 |
|---|---|---|
| `OK` | 正常命中 | 清除该工具的 strike 记录 |
| `EMPTY` | 数据集级：无此编译结构（`reason="no_structure"`） | 累计 2 次后禁用该工具（`_disable_tool`） |
| `MISS` | 查询级：本查询无命中（`reason="no_doc"`） | 不禁用 |
| `POOR` | 有输出但太弱 | 交给路由/梯子回退 |
| `REDUNDANT` | 执行成功但无新证据 | 显式提示模型换角度 |
| `ERROR` | 基础设施/参数失败 | 记日志 |

#### 4.2.2 工具面（`_TOOL_MAP` + `_active_tool_specs`）

- `_TOOL_MAP`：8 个工具名 → OpenAI function schema。
- `_active_tool_specs(tools)`：按模式裁剪工具面；`web_search` 无 provider 时
  隐藏；已被 `_disabled_tools` 标记的编译工具移除。
- `_disable_tool`：把"数据集无此编译结构"记到 `tools._disabled_tools`，
  后续 `execute_tool` 直接短路返回提示。

#### 4.2.3 工具执行器（`_exec_*` + `execute_tool`）

| 执行器 | 后端 | 说明 |
|---|---|---|
| `_exec_retrieve` | `grep_search` | 精确词定位、紧凑片段；`nav_hint` 作为 BM25 软提示（"导航是提示，不是约束"） |
| `_exec_search_chunks` | `hybrid_search` | 语义检索 + 编译结构扩展（`use_compiled=True`） |
| `_exec_web_search` | provider | 无 provider 时返回显式提示而非空结果 |
| `_exec_list_chunks` | `list_chunks` | 深读整篇文档；未知 doc_id 只算 MISS |
| `_exec_navigate_tree` | `_navigate_tree_impl` | 数据集级文档路由 |
| `_exec_navigate_structure` | `_navigate_structure_impl` | 文档内结构导航；`chunk_ptrs==0` → POOR |
| `_exec_calculate` | `arithmetic.compute_from_facts` | 安全算术 |
| `_exec_graph_explore` | `graph_explore` | 图谱探索；空结果 = 数据集级 `no_structure` |

`execute_tool` 是统一分发点（deepsearch ToolNode 等价物）。

#### 4.2.4 证据管理

- `_seed_evidence` / `_admit_evidence`：chunk 注册进会话输出 + 共享证据池
  （`tools.kbinfos["chunks"]`）。表格 chunk 不被截断（保留全表，防答案行被切掉）。
- `_run_search` / `_search_outcome`：查询类工具共享驱动；**无新证据 → REDUNDANT**。
- 近重复查询抑制：Jaccard token 重叠 ≥ 0.8 的检索查询被跳过（模型换措辞反复搜
  同一意图会烧轮次）。

#### 4.2.5 导航链（`_NAV_RULES` 梯子，`_NavRule`@1627，`_NAV_RULES`@1767）

```
locate  navigate_tree         路由到 top-n 文档，暴露摘要
drill   retrieve+structure    全库检索（nav 摘要作 BM25 软提示）+ navigate_structure
                              root→chunk 路径按 chunk_id 合并 + 路由文档置顶重排
global  retrieve              兜底（仅当 drill 本身为空/失败）
```

- `_NavRule`：`mode`（AUTO=代码驱动 / LLM=模型驱动）、`run`（组合步骤如
  `_run_drill_merge`）、`args`、`when`、`next`（按 status 路由）。
- `run_nav_prefix`：会话开始前以代码跑梯子到第一个 LLM 步骤，注入已完成
  assistant/tool 消息对（模型看到"已完成的链"，无法跳过）。
- 会话中 `_tool_node` 里若模型某轮结果弱（非 OK），梯子自动继续。

#### 4.2.6 会话图与路由

- `_build_session_graph`：START → run_action → {tool | finalize | END} → …
- `_route`：先处理 pending tool_calls；到达 `_action_max_turns` → finalize；
  近重复跳过 ≥ 2 次 → 提前收敛到 finalize。
- `_route_after_tool`：执行工具后必须回放 tool 响应，再查预算（防死循环）。

#### 4.2.7 入口

- `run_action_session(tools, direction, parent_state, ...)`：单 direction 的有界会话。
  先跑导航前缀，再 `_SESSION_GRAPH.ainvoke`，返回 `Result`。
- `initialize_state(tools, question, fanout_hint)`：LLM 分解问题 → `slots` +
  `first_queries`；失败时用 fanout hint 兜底构造槽位表。

---

### 4.3 `tools/` — 工具实现层

#### 4.3.1 `search.py` — 检索核心

| 函数 | 说明 |
|---|---|
| `hybrid_search` | 混合（向量+BM25）检索，可开 `use_compiled` 扩展；写入 `tools.search_cache` 防重复检索；结果写进 memory |
| `vector_search` | 纯向量（`vector_similarity_weight=1.0`） |
| `bm25_search` | 纯关键词 |
| `web_search` | 外部 provider（无则空） |
| `structured_query` | 结构化表 KB → SQL |
| `grep_search` | BM25 候选池 + 正则定位 + 上下文窗口；表格 chunk 不窄化 |
| `list_chunks` | 按 doc_id 深读整篇（≤30 chunk，`[:30]`） |
| `_base_chat_mdl` | 解析 `RAGTools.chat_mdl` 链到最内层 tool-calling 模型 |
| `_get_kb_ids` / `_tools_slot` / `_tools_ref` | 请求级工具实例槽（navigation 等模块读它） |

所有检索均通过 `settings.retriever.retrieval`（`common/settings` 全局单例）。

#### 4.3.2 `navigation.py` — 导航

- **数据集级**：`dataset_navigation_by_tree` / `dataset_navigation_search` /
  `_nav_search_titled`（`search_dataset_layers(mode="navigation_tree")` 混合 BFS
  beam 下钻，向量+BM25 双腿）。
- **文档级**：`ontology_navigate` / `mindmap_navigate` → `_navigate_within_doc`
  （读取编译 entity/relation，LLM 选相关实体，拉 source_chunk_ids 对应 chunk）。
- **工具实现**：`_navigate_tree_impl` / `_navigate_structure_impl`
  （`@tool(timeout=120)` 装饰，返回 `NavResult`：`text`+结构化字段+`empty_reason`）。
- **TOC 钻孔**：`_embed_query` → `_load_entities_with_vectors` →
  `_drill_kept_nodes`（向量 beam：每层保留 top-K 余弦相似节点下钻）→
  `_render_toc_drilldown`。`_expand_related_via_structure` 供 search_chunks 扩相关 chunk。
- 两种编译行形状并存：graph blob（`knowledge_graph_kwd="graph"`）与 per-entity/
  relation 行；`compile_kwd`（非 `knowledge_graph_kwd`）区分编译类型。

#### 4.3.3 `exploration.py` — 图谱与 wiki

- `graph_explore`：KG 种子实体（dense match ≥0.8 + mention_count 排序）→
  BFS `_KG_HOPS` 跳 → `_ask_structure` 判断子图是否直接回答；
  不足则返回相关实体/关系的 source chunks。
- `wiki_query`：编译 wiki 页检索（BM25 + dense fusion）。
- `_kg_scopes` / `_kg_search`：被 compiled_expansion 复用。

#### 4.3.4 `compiled_expansion.py` — 编译结构扩展

`_expand_with_compiled` 对每个 KB 依次跑：knowledge_graph / mind_map / timeline /
page_index 模板的 1 跳实体图扩展（`_expand_compiled_strategy`）+ tree 编译行 +
wiki/artifact/essence 合成页（`_expand_wiki_page_strategy`）。零额外 LLM。

#### 4.3.5 `text_processing.py` — 文本处理

- 句子切分（HTML/markdown 表格整体原子化，`_split_sentences`）。
- 词干化（`_stem`：优先 nltk Porter，兜底后缀剥离）。
- 关键词窄化（`_narrow_by_keywords` / `_narrow_or_keep` / `_highlight_keywords`；
  表格整表保留；事实密集句保留）。
- `_is_fact_dense_sentence`：数字/年份/百分比/专有名词启发式。

---

### 4.4 `orchestrator/` — 顶层编排零件

| 文件 | 行为 |
|---|---|
| `direct.py` | low 模式：`direct_search` 一次 hybrid → 合并进 `tools.kbinfos`；无结果标记 `empty_result` |
| `query_rewriter.py` | `rewrite_gap_to_query`：SCA forward gap（what+hint）→ 用 `sca_query_rewrite` 提示词生成新查询（支持多跳锚定 bridge value），失败返回空表由调用方回退原 gap |
| `sufficient_context.py` | `sufficient_context_agent`：一次 LLM 审查（`sca_select` 提示词），输入=各 claim 草稿+其引用证据锚点+整体草稿；输出统一 verdict（is_sufficient/confidence/contradictions/claims/sub_queries）。`to_boost`/`to_grounded` 适配旧决策阶梯 |

---

### 4.5 支撑模块

| 文件 | 核心行为 |
|---|---|
| `keywords.py` | `extract_weighted_keywords(llm, question)` → `(weighted_query, union_keywords)`；entity/qualifier 各重复 3 次喂 BM25 |
| `arithmetic.py` | `compute(expr)`：AST 白名单校验后 `eval`（无 builtins）；`compute_from_facts`：LLM 判断是否需推导并生成表达式 |
| `grep_sed_narrow.py` | `narrow_by_terms`：术语→正则→行定位+上下文窗口；`grep_sed_narrow`：从 claim 机械提取术语（零 LLM）；多级兜底不丢答案 |
| `memory.py` | `add`/`grep`/`search`/`size`/`clear`；存于 `tools.kbinfos["memory"]`；`search` 用语言无关显著项打分（拉丁词+数字+CJK 3-gram） |
| `stats.py` | `phase()`/`in_phase()`（阶段计时）、`LLMUsageStats`、`CountingChatModel`（代理 LLMBundle 计数）、`record_external_*` |
| `structure_qa.py` | `_ask_structure`：渲染 entity/relation 大纲 → LLM 判是否直接回答 + 返回相关实体名 |
| `chunk_utils.py` | chunk 字段别名归一（`_chunk_id`/`_chunk_text`/`_doc_id`/`_doc_title`/`_dataset_id`/`_snippet`/`_xml_escape`） |
| `prompts/report_prompt.py` | 终答系统提示词（citation / 属性保真 / 语言一致 / 兜底） |

---

## 5. Go 移植映射

### 5.0 移植状态（2026-09-02 更新）

`internal/rag/advanced_rag/harness` 的重写已完成。旧版移植（2026-08-28 停止演进）针对的是**已被
删除的旧版 Python harness**（见 5.0.1），已整体清除并重写。

| 项 | 状态 |
|---|---|
| 旧架构文件清理 | ✅ 删除 15 个过时 `.go` + 10 个失效测试 |
| 能力层复用 | ✅ `navigation.go` / `keywords.go` / `arithmetic.go` / `compiled_expansion.go` 保留 |
| 核心编排层 | ✅ 见 5.1 文件表 |
| **`agentic_rag_graph.py` 外层循环** | ✅ 已移植（父目录 `agentic_rag*.go`，package `advanced_rag`，见 5.1） |
| **首尾两个 LLM 节点** | ✅ Phase 0 `Formalize` + Phase 5 答案合成 |
| 对外接口对齐 | ✅ `Run` / `RunActionSession` / `InitializeState` 签名与语义对齐 Python |

**四种模式均已端到端对齐**（`low` / `medium` / `high` / `ultra`，含 `naive` 回退）：
`formalize → 研究 → 答案合成`，每种都返回带 `[n]` 引用的最终答案。

**收尾项已完成**：`loadStructureEntities` 已接入 `ParseCompiledStructure`，读取
graph-blob **与** per-entity/relation 两种行形状（对齐 Python `_load_compiled_structure`），
并投影回 `structureEntity`（见 `navigation.go`）。

**逐项核验 9 处行为差异并已修复**：见 5.0.3。

**测试状态**：`bash build.sh --test ./internal/...` 下 harness 相关包全部 `ok`
（`internal/rag/advanced_rag/harness`、`internal/rag/advanced_rag/harness/orchestrator`、
`internal/rag/advanced_rag/harness/prompts`、`internal/agent/tool`、
`internal/agent/runtime`、`internal/agent/component`）。
`internal/handler` 的 `TestDownloadAttachment_OK` 失败是**既有的**（pre-existing），
该包不 import harness，与本次改动无关。

#### 5.0.1 旧版移植为何废弃（历史记录）

旧版移植的是 `route → planner → orchestrator → formalize_answer` 显式 DAG，配
`ExecutionStrategy` + decision-ladder `CHigh/CLow/LLMFloor` + `SufficiencyVerdict` 五态
+ 14 个工具（含 4 个 `inspector_*`，见旧 `pipeline.go`:182-201）。其注释引用的 Python
原型 `harness/types.py`、`harness/pipeline.py`、`harness/gating.py`、
`orchestrator/agentic.py` **在 Python 重写后全部消失**，且旧版从未接入生产
（全仓库无 handler/service import）。断层是架构级的，故整体重写而非增量对齐。

#### 5.0.2 逐文件处置（已执行）

| Go 文件 | 处置 | 说明 |
|---|---|---|
| `navigation.go` / `datasetnav.go` / `kg_explore.go` | ✅ 保留 | 零依赖旧符号，直接沿用 |
| `compilation.go` | ✅ 保留 | 门控概念需按「扩展」语义重做（见 §5.6.1 未完项） |
| `types.go` / `pipeline.go` / `agentic.go` / `agentic_rag.go` / `route.go` / `planner.go` / `orchestrator.go` / `sufficiency*.go` / `grounded_llm.go` / `inspector.go` / `production.go` / `answer.go` / `research_agent.go` | ❌ 已删除 | 原型已删除或范式废弃 |
| `Kbinfos` / `chunkKey` / `chunkText` / `splitSentences` / `normalizeWebResults` / `unmarshalModelJSON` / `installChat` | ⚠️ 已迁移 | 原定义在待删文件中，已移至 `kbinfos.go` / `textproc.go` / `websearch.go` / `llmutil.go` / `harness_test.go` |

对应地，**`internal/harness/`（通用 Pregel/agentcore 框架）与本次移植无关**，它是独立
的通用 agent 框架，不引用也不被 `internal/rag/advanced_rag/harness` 引用。当前实现用显式状态循环，
未引入图引擎依赖。

### 5.1 移植后的 package 划分（实际态）

```
internal/rag/advanced_rag/            ← rag/advanced_rag/（2026-09-04 由 internal/agent 迁入，与 Python 同目录结构）
├── agentic_rag.go          ← agentic_rag.py           RAGTools 依赖/配置 + 方法归属总表
│                             + formalize(:412) Phase 0 问句形式化 + 关键词压缩
│                             + rag(:817) 外层 Run + 模式分派
├── agentic_rag_graph.go    ← agentic_rag_graph.py      图节点/路由/fanout/SCA 草稿/回退草稿
│                             + _compose_answer_from_evidence(:604)
│                             + _naive_rag(:1517) 合成半部（kb/cite 见 internal/rag/prompts）

├── agentic_rag_util.go     （本地小工具：trunc / deadlineToDuration）
└── harness/                ← rag/advanced_rag/harness/（package harness，约 9800 行）
    ├── config.go            ← config.py             ModeSpec + THINKING_MODES（5 模式含 naive）
    ├── action_session.go    ← action_session.py     会话类型(Variable/State/Result) + ApplyPatch/
    │                                                extract_json/extract_tag + 工具 schema/Toolset/
    │                                                ToolExecutor + _NAV_RULES 导航梯子 + ReAct 循环
    ├── tool_executor.go     ← agentic_rag*.py       顶层 Run 入口 + 工具接线 + RuntimeRetriever 适配
    │                                                + 证据发布（harness↔advanced_rag 的桥）
    ├── search.go            ← tools/search.py       Retriever 接口 + HybridSearch + doc 聚合
    │                                                + 缓存 + CompiledExpander
    ├── kbinfos.go           ← tools/kbinfos         共享存储 + chunkKey/chunkText + SearchParams
    ├── memory.go            ← memory.py             原始 chunk 无损存储 + grep
    ├── exploration.go       ← tools/exploration.py  graph_explore / web/wiki 工具接线
    │                                                （WebSearcher / WikiRetriever）
    ├── structure_qa.go      ← structure_qa.py       RenderStructure / AskStructure
    │                                                （_render_structure / _ask_structure，供 graph_explore 委托）
    ├── grep_sed_narrow.go   ← grep_sed_narrow.py    EscapeTerm / TermsToPatterns / NarrowByTerms /
    │                                                SplitFallbackTerms / GrepSedNarrow
    ├── stats.go             ← stats.py              LLMUsageStats + CountingInvoker
    ├── text_processing.go   ← text_processing.py    句子切分（表格原子）+ 窄化 + 高亮
    ├── chunk_utils.go       ← chunk_utils.py        字段别名归一
    ├── keywords.go / arithmetic.go / compiled_expansion.go / navigation.go
    │                        （工具/能力层，逐函数归属见文件内注释）
    ├── prompts.go           ← rag/prompts/*.md      内置协议兜底（loader 无法解析名字时）
    ├── report_prompt.go     ← report_prompt.py      FINAL_ANSWER_SYSTEM / PARTIAL_ANSWER_PREAMBLE
    ├── orchestrator/
    │   ├── direct.go              ← orchestrator/direct.py
    │   ├── sufficient_context.go  ← orchestrator/sufficient_context.py
    │   ├── query_rewriter.go      ← orchestrator/query_rewriter.py
    │   └── prompts.go             ← rag/prompts/sca_select.md / sca_query_rewrite.md
    └── prompts/
        ├── loader.go              ← rag/prompts/template.py::load_prompt（go:embed）
        └── *.md                   rag/prompts/ 的 canonical 内嵌副本（action_run /
                                   action_initialize_state / sca_select / sca_query_rewrite）
```

**为何外层循环是 harness 的父级包（package `advanced_rag`）**：Python 中 `agentic_rag*.py`
与 `harness/` 同级（都在 `rag/advanced_rag/`）；Go 保持同构——外层循环放 `harness/` 的父目录，
import `harness` + `harness/orchestrator`；harness 反向不 import 外层循环（靠 `tool_executor.go` 的
RunDeps/上下文/回调解耦），依赖全部单向，无循环。旧设计中独立的 `agentic/` 子包
与 `sessiontypes.go`/`session.go`/`run.go`/`navchain.go`/`toolschema.go`/`extract.go`/
`formalize.go`/`answer_compose.go` 等拆分已取消，统一收敛到上述布局。

### 5.2 关键类型映射（实际实现）

| Python | Go 实现 | 位置 |
|---|---|---|
| `ModeSpec`（frozen dataclass） | `type ModeSpec struct` | `config.go` |
| `Variable` / `State` / `Result` | 同名 struct | `sessiontypes.go` |
| `apply_patch` | `ApplyPatch(base, patches) *State`（纯函数） | `sessiontypes.go` |
| `ToolOutcome(status, reason, metrics)` | 同名 struct + `Status*` 常量 | `sessiontypes.go` |
| `_TOOL_MAP` / `_active_tool_specs` | `ToolMap` + `(*Toolset).ActiveToolSpecs()` | `toolschema.go` |
| `execute_tool` | `ExecuteTool(ctx, tools, name, args) ToolOutcome` | `session.go` |
| `_NavRule` / `_NavContext` / `_NAV_RULES` | `NavRule` / `NavContext` / `NavRules` | `navchain.go` |
| `run_action_session` / `initialize_state` | 同名导出函数 | `run.go` |
| `tools.kbinfos` | `Kbinfos{Chunks, DocAggs, Memory}` | `kbinfos.go` |
| `settings.retriever` | `Retriever` 接口 + `RuntimeRetriever` 适配 | `search.go` / `runtime.go` |
| `extract_json` / `extract_tag` | `ExtractJSON` / `ExtractTag` | `extract.go` |
| `LLMUsageStats` / `CountingChatModel` | 同名 / `CountingInvoker` | `stats.go` |
| `RAGTools.answer` 分流 | `Run(ctx, deps, req) *RunResponse` | `tool_executor.go` |

**与 Python 的三处结构性差异**（均已隔离在单个 seam，行为等价，非偏差）：

1. **LangGraph → 显式循环**：图是固定三节点，编译图无收益。节点体、路由谓词、顺序逐分支
   照搬，`sessionLoop` 注释标明每条边的对应关系。
2. **tool calling**：Python 用 provider 原生 `tools=[...]`；Go 的 `chat.Request` 无 Tools
   字段，故抽象为 `SessionModel` 接口，默认 `InvokerSessionModel` 用 prompt-based JSON 块。
3. **ContextVar → context.Context**：Python 用 ContextVar 做阶段隔离，Go 用 ctx 携带。

> 这三类均为环境约束驱动的等价实现，不构成与 Python 的行为偏差。剩余的 `web_search` /
> `wiki_query` 仅受"外部 provider 是否注入"影响（能力缺口，见 §5.4），`compilation.go`
> 门控需按「扩展」语义重做（见 §5.6 未完项），二者亦非移植不一致。

### 5.3 状态机（已实现）

显式状态循环替代 LangGraph（`session.go`）：

```
runActionNode → route() → {END | tool | finalize | runAction}
             tool → routeAfterTool() → {runAction | finalize}
             finalize → END
```

四条守卫逻辑**全部保留并锁定测试**：

1. pending tool_calls 必须先回放 tool 响应，再查轮次预算（防 Q86 死循环）；
2. 近重复查询（Jaccard ≥ 0.8）跳过 ≥2 次 → 提前收敛 finalize；
3. EMPTY(no_structure) 累计 2 次 → 禁用该工具（strike 机制）；
4. REDUNDANT（无新证据）显式提示模型。

导航梯子（`navchain.go`）在会话启动前以**代码**跑 `locate → drill → global`，把完成的
assistant/tool 交换注入历史，使模型无法跳过路由步骤。

### 5.4 工具注册与 schema（已实现）

`ToolMap` 注册表（9 个工具，schema 描述逐字照搬 Python）+ `Toolset.ActiveToolSpecs()`
动态裁剪：

- `web_search` 无 provider（未配置 `HasWebSearch`）→ 隐藏（而非仅劝阻）；
- `wiki_query`：已移植 handler 但未接线（生产无 `SearchWiki` 后端、不在活跃工具集），与 Python 一致不暴露；
- `graph_explore` 仅 ultra 可见；
- 运行时确认无编译结构 → `DisableTool` 从工具面移除。

当前接线状态（`tool_executor.go`）：`retrieve` / `search_chunks` / `grep_*` / `navigate_tree` /
`navigate_structure` / `list_chunks` / `calculate` / `graph_explore` 均已实现并接线；
`web_search`（经可选 `WebSearcher` provider）已接线，无 provider 时 `Toolset` 隐藏该工具、
`Execute` 返回干净 MISS；`wiki_query` **仅移植了 handler**（对齐 `tools/exploration.py::wiki_query`，
payload 形状一致），**未接线**：生产无 `SearchWiki` 后端，且已从 `allTools` 移除，因此与 Python
一致**不暴露于**模型工具面（Python 的 `wiki_query` 也从未注册进 action session）。`WikiRetriever`
接口 + `wikiQuery` 执行分支作为未接线的扩展 seam 保留。

这属**能力缺口而非移植不一致**——与 Python 一致：Python 的 `web_search` 无 provider
时返回显式提示、`wiki_query` 仅在数据集存在 `wiki_page_draft` 编译时才真正工作。

> 三处结构性差异（LangGraph→显式循环、tools=[...]→JSON-block 调用、import→依赖注入）
> 见 §5.0.3 与 §5.2 末尾说明；行为等价，非偏差。

### 5.5 依赖与注意点

| 依赖 | Go 侧处理 |
|---|---|
| `common.settings.retriever` | `Retriever` 接口；`RuntimeRetriever` 适配 `runtime.RetrievalService` |
| `api.db.services.dataset_api_service` | 编译结构读取**已移植**：`navigation.go` 经 `nav.NavService` + `ParseCompiledStructure` 读取 graph/entity/relation 三种行形状（见 §5.0） |
| `rag.prompts.*` | `PromptLoader` 接口；未接线时用 `prompts.go` 内置兜底 |
| `json_repair` | `llmutil.unmarshalModelJSON` + `extract.go` 的 brace-matching |
| `thread_pool_exec` | 未移植（Go 侧无对应阻塞调用） |
| 表格处理 | 集中于 `textproc.splitSentences`（表格原子）+ `sca.boundedExcerpt`（表格整体保留） |
| `_tools_ref` 全局槽 | 改为显式参数 / struct 字段传递，无全局可变状态 |
| 并发安全 | `Kbinfos.cache` 用 `sync.Mutex`；`LLMUsageStats` 全字段加锁 |

### 5.6 外层循环（`agentic` 包）与接线方式

`agentic_rag_graph.py` 的五阶段流水线已移植：

```
formalize_question → [planner → prefetch] → rag_agent → draft → sca
    ├─ sufficient ──────────────────────────→ formalize_answer
    └─ insufficient → query_rewrite ────────→ rag_agent（下一轮）
```

**激活方式**：`agentic` 包通过 `init()` 注册到 `harness.SetAgenticLoop`。导入该包
（含 blank import）即激活完整 medium/high/ultra 流水线：

```go
import _ "ragflow/internal/rag/advanced_rag/harness/agentic"
```

未导入时，`harness.Run` 对这些模式退化为**单次行动会话**（仍可用，但无
planner/fan-out/SCA 迭代）。这是依赖倒置：`agentic` 导入 harness，故 harness
不能反向导入它。

**路由守卫**（`_route_sca` / `_route_rewrite`）全部保留：

| 守卫 | 行为 |
|---|---|
| `no_progress` | 立即收敛到 formalize_answer |
| `enable_sca=false` | medium 单趟，verdict 仅作参考 |
| `search_rounds >= sca_max_rounds` | 停止迭代 |
| `remaining <= min_round_headroom(50s)` | 预算不足则不再开新轮 |
| SCA view 重复（view_id 未变） | 判为无效轮，停止 |
| 重写后 0 新增片段且 rounds≥1 | 检索饱和，停止 |

**一处代码与注释矛盾**：Python `agentic_rag_graph.py`:1174 的注释写着
"Prefetch is DISABLED"，但紧随其后的代码是 `use_prefetch = use_fanout`（即
high/ultra 仍启用）。Go 版**按代码行为实现**（fan-out 模式启用 prefetch），并在
`graph.go` 注明该注释已过时。

#### 5.6.1 `compilation.go` 门控（未完项，非移植缺口）

`compilation.go` 保留为能力层，但其"门控"概念尚无对应 Python 语义，需按「**扩展**」
（expansion）语义重做——即对应 Python `compiled_expansion.py` 的 `use_compiled` 扩展
（hybrid_search 时叠加 1 跳实体图 + tree/wiki 合成页），而非简单"有/无该结构"的开关。

当前状态：

- `buildCompilationMap` 已产出 `graph` / `entity` / `relation` / `tree` / `wiki` 五类 token；
- 工具面裁剪（`ActiveToolSpecs` / `DisableTool`）已按"无编译结构则移除工具"实现；
- 但"扩展"语义（检索时真正叠加编译结构）尚未接线，属**待扩展项**，与 Python 的偏差
  不在移植一致性，而在功能完整度。该未完项与 §5.4 的 `web_search`/`wiki_query`
  provider 缺口、§5.2 的结构性差异并列，均**不构成移植不一致**。

### 5.7 首尾两个 LLM 节点

**Phase 0 `Formalize`**（`formalize.go`，对齐 `agentic_rag.py:412`）：把最新用户消息改写为
独立问题并抽取关键词，**一次 LLM 调用**完成。

- **单轮短路**：没有指代需要消解时，问题**保持原样不改写**，也不调 LLM。改写自包含问题
  有静默改变语义的风险，而单轮是绝大多数场景。
- 多轮才走 LLM：解析出 `question` + `keywords`，失败则回退到原始消息（绝不返回空问题）。
- 关键词经 `CompactKeywords` 去重并截断到 15 项——提示词要求每个词给 2-3 个同义词，
  模型常返回 40-60 词的冗余串，直接拼到查询会稀释向量召回并让 BM25 偏到无关文档。
- **默认关闭**（`RunDeps.Formalize`），因为它多一次 LLM 往返；传 `Messages` 并开启才生效。

**Phase 5 答案合成**（`answer_compose.go`，对齐 `_compose_answer_from_evidence` + `kb_prompt`）：

1. 无证据 + 配置了 `empty_response` → 直接返回，**不调 LLM**；
2. 按 `similarity`/`score` 排序，取前 6 个 chunk 作为引用参考集；
3. `KBPrompt` 在 token 预算（默认 8000）内渲染编号证据块（`ID: n` 从 1 开始，与模型的
   `[n]` 引用对应；元数据内的换行按 Python 规则压平为一个空格）；
4. `KB.PreSummary`（SCA 审阅过的事实草稿）作为权威发现前置；partial 时加上
   `PARTIAL_ANSWER_PREAMBLE`；
5. 用 `FINAL_ANSWER_SYSTEM` 调模型，产出带 `[n]` 引用的答案。

**系统提示词的优先级**（照搬 Python，含其长期踩坑的结论）：用户配置的 `system_prompt`
**追加在** agentic 契约**之后**——因为 `FINAL_ANSWER_SYSTEM` 以 `# Language` 规则结尾，
若前置则后写优先会让语言规则静默覆盖用户的"用英文回答"。结尾子句重述优先级：
**语言刻意可被覆盖**（"回答与问题同语言"正是用户想替换的规则），**受保护的是证据契约**
（引用来源、回答所问属性、不得以先验知识替代缺失证据）。

`ComposeNaiveAnswer` 是 naive 路径的独立实现（Python `_naive_rag:1555`）：用扁平的
`[i] content` 列表（每片截断 1500 字符、最多 8 片），**不用** `kb_prompt`。

### 5.8 支撑模块（已全部移植）

| 模块 | 文件 | 支撑 |
|---|---|---|
| `keywords.py` | `keywords.go` | 四面向加权关键词（entity x3 / qualifiers x3） |
| `arithmetic.py` | `arithmetic.go` + `compute.go` | `calculate` 工具（受限 Python 子集求值器） |
| `grep_sed_narrow.py` | `narrow.go` | grep+sed 证据窄化（含回退链） |
| `structure_qa.py` | 复用 `navigation.go` 的 `askStructureSelect` / `kg_explore.go` 的 `askStructureAnswer` | 结构大纲 QA |
| 编译结构读取 | `navtools.go` + `navservice.go` + `ParseCompiledStructure` | `navigate_tree` / `navigate_structure` / `graph_explore` |

**编译结构读取**：`navigate_tree` 已通过 `nav.NavService.Search`（Go 侧对应 Python
的 `search_dataset_layers(..., "navigation_tree", ...)`）**真正工作**；
`navigate_structure` / `graph_explore` 接入了既有的 `NavigateStructure` /
`ExploreGraph`。

**编译结构读取（已对齐）**：Go 的 `loadStructureEntities` 现在与 Python
`_load_compiled_structure` 一致，查询 `graph`/`entity`/`relation` 三种行形状，经
`ParseCompiledStructure`（已实现并测试）合并解析，再投影回 `structureEntity`。
`navigate_structure` 不再对"按 per-entity/relation 行编译"的数据集误报 EMPTY。

#### 5.0.3 与 Python 的行为差异修复（2026-09-02）

对 Go 实现与 Python `action_session.py` 逐条核验，发现并修复 9 处行为不一致（按影响排序）：

| # | 差异 | Python 参照 | Go 修复 | 文件 |
|---|---|---|---|---|
| 1 | ReAct 循环反复搜同一证据（缺 ALREADY RETRIEVED 种子） | `run_action_session:1982-1984` 用 `_extract_relevant_evidence` 注入已有证据 | `SessionDeps` 加 `KB`；`run.go` 新增 `extractRelevantEvidence` 注入 seedUser；`run.go` 的 `runSingleSession` 传 `kb` | `run.go` / `tool_executor.go` |
| 2 | `finalizeNode` 缺 loose-clue harvest 兜底 | `_finalize_node:1497-1518` salvage 失败时扫末条 AI 叙述（≥24 字符）补第一个未填槽 | `finalizeNode` 加 harvest 分支；新增 `lastNarration` | `session.go` |
| 3 | `snippetsPerQuery=4` 是死常量（结果条数失控） | `_run_search:705` 每查询截前 4 条 | `passEvidence` 用 `snippetsPerQuery` 限制每查询条数 | `tool_executor.go` |
| 4 | `retrieve` 与 `search_chunks` 的 `top_n` 未区分 | `:743` retrieve=10、`:763` search_chunks=20 | `Execute` 内 `search_chunks`→20、`retrieve`→10 | `tool_executor.go` |
| 5 | chunk 截断无表格保留（答案行被切） | `_admit_evidence:667-671` 非表截 1200、表格 chunk 留全表 | `passageFromChunk` 表格全文、非表格 1200；`chunk_utils.go` 新增 `IsTableChunk`/`isTableText` | `tool_executor.go` / `chunk_utils.go` |
| 6 | `list_chunks` 无 30 条上限 | `_exec_list_chunks:827` 截 `[:30]` | `executeListChunks` 加 `[:30]` 上限 | `tool_executor.go` |
| 7 | `navigate_structure` schema 多了 `chunk_ids`/`want_chunk` | `_NAVIGATE_STRUCTURE_TOOL_SPEC:350-359` 仅 doc_id/query/kind | 移除两个参数；删除无用 `paramBool`/`paramArrayOfStrings` | `toolschema.go` |
| 8 | `web_search` 暴露但未接线（落默认 MISS） | 真实 provider 路径（provider 缺失时 `action_session.py:463` 从工具面移除） | 新增 `WebSearcher` 接口 + `web_search` 执行分支；未配置（`HasWebSearch=false`）时 `Toolset` 隐藏 + `Execute` 返回干净 MISS；`SearchDeps`/`agentic/register.go` 接 `WebSearch` | `tool_executor.go` / `search.go` / `agentic/register.go` |
| 9 | 单轮 LLM 无独立超时（可吃满整段预算） | `_run_action_node:1207-1210` 每轮 `wall=max(15,min(75,deadline_left))` | `runActionNode` 加每轮 wall 超时 | `session.go` |
| 10 | `wiki_query` 未接线（仅 MISS 占位，无 wiki 页检索） | `tools/exploration.py::wiki_query` 对 `wiki_page_draft` 编译行做 BM25+dense fusion | Go 已移植 handler（`wikiQuery` 执行分支 + `WikiRetriever` 接口，payload 形状对齐 `_exec_wiki_query`）；但生产无 `SearchWiki` 后端、且已从 `allTools` 移除，因此与 Python 一致**不暴露**，仅作未接线扩展 seam 保留 | `exploration.go` / `search.go` |
| 11 | `extractRelevantEvidence` 不按 `direction` 相关度排序、截断 220 | `_extract_relevant_evidence:1939-1964` 按 direction token 相关度排序后取 top-N、截断 300 | `run.go` 新增 `tokenizeDirection`/`directionRelevance`，按相关度 `sort.SliceStable` 取 top-N，截断 300，空 direction 时落 `chunks[-max:]` 兜底 | `run.go` |

**结构性差异（非上述 9 条，属环境约束驱动的等价实现，已在 5.0/5.1 标注）**：
- LangGraph → 显式状态循环（`session.go` 三函数一开关）；
- 原生 `tools=[...]` → `SessionModel` JSON-block 工具调用；
- import 循环 → 依赖注入（`agentic.SetAgenticLoop`）。

**测试**：三包 + `internal/deepdoc/native` 全部 `ok`；`bridge_test.go` 的
`TestPassageFromChunkTruncatesContent` 期望由 600→1200 截断（对齐 Python 非表格上限）。

#### 5.0.4 orchestrator 与边界层的有意偏离（2026-09-07 审查）

对 `orchestrator/`（direct.go / sufficient_context.go / query_rewriter.go）与
`kbinfos.go`、`agentic_rag_graph.go` 并发编排做了逐点核验。以下均为**已确认的有意
偏离或等价设计**，非未被发现的 parity bug，列入此节防止后续误改：

| # | 点 | Python 参照 | Go 实现 | 判定 |
|---|---|---|---|---|
| 1 | **并发模型** | `agentic_rag.py` 主循环顶层 stage（SCA→rewrite→direct→slot）**串行 `await`**，无跨 stage `asyncio.gather`；slot-table research 在 `asyncio.Semaphore(2)` 下并发 | 顶层 stage 同步串行调用；`SlotResearch` 用 `go func` + `chan struct{}` 信号量（`slotSessionConcurrency=2`）+ `sync.WaitGroup`，`wg.Wait()` 后 `sort.Slice` 按 slotID 确定性折叠 | ✅ 行为等价（Go 无 async/await，goroutine+semaphore 是等价范式；并发度未落后 Python）|
| 2 | **SCA 证据锚点文本键优先级** | `sufficient_context.py:188`：`content_with_weight` → `content` → `chunk` | `renderClaimContext` 走独立 `scaChunkText`（`content_with_weight`→`content`→`chunk`），**不复用** memory 路径的 `chunkText`（`content`→`content_with_weight`→`text`） | ✅ 有意区分：SCA 必须读 rerank 后的 `content_with_weight`，与 Python 一致；勿把 `scaChunkText` 简并回 `ChunkTextOf` |
| 3 | **JSONModel `"Output:\n"` 契约** | `gen_json(prompt, "Output:\n", chat_mdl)`：prompt 作 system turn、`"Output:\n"` 作 user turn，每个调用点显式传参 | `JSONModel.GenJSON(ctx, prompt)` 把 `"Output:\n"` **固定在 `jsonModelAdapter.GenJSON` 内部**（user turn 硬编码），调用方不可见 | ✅ 行为一致，但契约隐藏；已由 `TestGenJSONUsesOutputNewlineUserTurn` 锁定首轮 user turn 恰为 `"Output:\n"` |
| 4 | **direct.go nil 守卫** | `direct_search` 直接访问 `tools.kbinfos` / `hybrid_search`，缺字段会 `AttributeError` 抛异常 | `DirectSearch` 在 `deps.KB == nil` / `deps.Search == nil` 时早返回空结果 | ✅ Go 防御式增强；正常（非 nil）调用下与 Python 行为一致 |
| 5 | **dedup 兜底** | `_chunk_key`：`chunk_id` or `id` or `str(id(ck))`（对象内存地址）→ 无 id 的等价 chunk 跨对象永不合并，去重静默失效 | `chunkKey`：`chunk_id`/`id` 缺失时回落 **content 派生 key**（content_with_weight→content，再 doc 引用兜底） | ✅ 有意改进：`chunkKey` doc 注释已显式说明偏离 Python `id(ck)` |
| 6 | **direct.go 失败日志** | `logger.exception(...)` 带完整 traceback | `_LOG.Printf("%v", err)` 仅 error 字符串 | ✅ 语言差异：Go error 不携带堆栈（除非 `fmt.Errorf("%w")` 包装），非行为 gap |
| 7 | **direct.go 同步签名** | `direct_search` 为 `async def` | `DirectSearch` 为同步函数（并发由 orchestration 层 `go` 编排） | ✅ 架构选择，见 #1 |

> 说明：本表与 §5.0.3 的区别在于——§5.0.3 是"曾偏离、已修复的 bug"，本表是"经核验
> 确认正确、且相对 Python 有改进/等价范式差异"的设计点。后续审查若发现其中某项其实
> 是 bug，应提升到 §5.0.3 修复并移出本表。

#### 5.0.5 tool_text_processing 审查（2026-09-08）

对 `tools/text_processing.py`（`_compact_keywords` / `_is_fact_dense_sentence` /
`_highlight_keywords` / `_MD_TABLE`）与 Go `tool_text_processing.go` /
`keywords.go` 的逐点核验：

| # | 点 | Python 参照 | Go 状态 | 处置 |
|---|---|---|---|---|
| 1 | `_compact_keywords` | `text_processing.py:20`：dedupe + cap **15 terms**，压缩 formalize/extract 的同义词 run | **Go 从未移植该函数**（harness 内无 `CompactKeywords`）。`ExtractWeightedKeywords` 直接 `strings.Join` 全量 union/weighted | ✅ **已补 cap-15**（`keywords.go`：union/weighted 各截断至 `maxKeywordTerms=15`，对齐 Python `max 15`）；dedupe 已由 `parseAspects` 跨 aspect `seen` 集覆盖，无需重做 |
| 2 | `IsFactDenseSentence` 专有名词保留 | `_is_fact_dense_sentence` 用 `_FACT_RE` **或** `_PROPER_NOUN_RE`（`\b[A-Z][a-z]{2,}\b`） | Go 原只查 number/quoted/≥6 token，**丢 proper-noun 分支** → 实体答案在 `NarrowContent` 兜底时丢失 | ✅ **已修**（`tool_text_processing.go`：新增 `properNounPattern` 并加入 `IsFactDenseSentence`，保住 "Atlanta Braves"/"Maseru" 类实体句子）|
| 3 | highlight 标记 | `_highlight_keywords` 用 `*…*`（星号包裹） | Go `HighlightKeywords` 用 `<em>…</em>` | ⚠️ **非 gap**：`HighlightKeywords` 在 Go 与 Python 两端**均无任何下游调用**（grep 全仓仅定义），属死代码格式差异；若将来接上游需统一标记符 |
| 4 | `mdTableRe` 宽容度 | `_MD_TABLE` header 为 `[^\n]*\|[^\n]*\r?\n`（认 CRLF） | Go `mdTableRe` header 为 `[^\n]*\|[\n]`（仅 `\n`），separator/body 仍认 `\r?\n` | ⏸️ **低优先级未动**：仅 CRLF 输入的表 header 边界识别略弱；纯 `\n` 环境行为一致 |

> §5.0.5 的 #1/#2 已落实到代码（非"待修"）；#3 经核查为死代码不改；#4 留作已知放宽。

**安全白名单**（`arithmetic.go`）：LLM 写的表达式来自不可信来源，Go 无 eval，且
LLM 写的是 **Python** 语法（`**`、三元、列表字面量），`go/parser` 无法解析——故
自实现词法+Pratt 解析+AST 白名单，**求值前**拒绝白名单外的一切。无反射、无属性
访问、无下标；未列出的构造是解析错误而非运行时沙箱逃逸。

**Python 的又一处注释/代码矛盾**（`_escape_term`）：注释称"短词不加 `\b`（避免
`pop` 匹配不到 `population`）"，但代码是 `len>=3 且两侧字母数字` 才加 `\b`——
`pop` 恰好 3 字符，**会**加 `\b`，因此匹配不到 `population`。Go 版**按代码实现**，
`TestEscapeTerm` 锁定。
| `calculate` 工具 | `arithmetic.py` | ✅ 已接线（`tool_executor.go` 的 `calculate`，安全沙箱求值） |
| `graph_explore` | `tools/exploration.py` | ✅ 已接线（`tool_executor.go` 的 `graphExplore`） |
| `wiki_query` | `tools/exploration.py` | ⚠️ 仅移植 handler（未接线）：`wikiQuery` 执行分支 + `WikiRetriever` 接口体量对齐 `_exec_wiki_query`，但生产无 `SearchWiki` 后端、不在 `allTools`，与 Python 一致不暴露；保留为扩展 seam |
| `keywords.py` | — | ✅ 已等价移植：`RunDeps.Keywords`（`KeywordExtractorFn`） |
| `grep_sed_narrow.py` | — | ✅ 已等价移植：`NarrowOrKeep`（`narrow.go`） |
| `structure_qa.py` | — | ✅ 已等价移植：`navigate_structure`（`navigation.go`） |
| `_fanout_search` 双通道 | BM25+窄化 与 hybrid 旁路 | Go 的 `Retriever` 只暴露单次加权调用，统一走 `NarrowOrKeep` |

**真实剩余未对齐项**（截至 2026-09-02，均已逐项记录，均不构成移植不一致）：
1. `web_search`：已接线但依赖可选 provider 注入（`WebSearcher`）。未配置时 `Toolset` 隐藏该工具、
   `Execute` 返回干净 MISS——属**能力缺口**而非移植偏差（Python 同步：无 provider 显式提示）。
   `wiki_query`：仅移植 handler 未接线（生产无 `SearchWiki` 后端、不在 `allTools`），与 Python 一致
   不暴露于工具面，保留为扩展 seam。
   详见 §5.4。
2. 结构性差异（环境约束等价实现，非行为偏差，见 §5.2 末尾）：LangGraph→显式循环、
   `tools=[...]`→JSON-block、import→依赖注入。
3. `compilation.go` 门控概念需按「扩展」语义重做（见 §5.6.1 未完项标注）。

`Run`/`RunRequest`/`RunResponse` 接口设计为**稳定**，移植外层循环时只需改 `runAgentic` 内部。

---

## 6. 移植状态 checklist

**第 0 步（已完成）**：清理 15 个旧架构文件 + 10 个失效测试；迁移 `Kbinfos`/`chunkKey`/
`splitSentences`/`unmarshalModelJSON`/`installChat` 等基础设施。

- [x] `config.go`：ModeSpec（含 `label`）+ 5 模式表 + naive 回退
- [x] `chunk_utils.go`：字段别名归一（`_chunk_attr`/`_chunk_text`/`_doc_id`/`_doc_title`/`_dataset_id`/`_chunk_id`/`_snippet`/`_xml_escape`，与 Python 完全对齐；`DocTitleOf` 仅 4 个键，无 Go 侧 `doc_name` 别名）
- [x] `textproc.go`：句子切分（表格原子）+ 关键词窄化 + 高亮
- [x] `search.go`：HybridSearch + 缓存 + memory 写入 + doc 聚合
- [x] `memory.go`：原始 chunk 无损存储 + grep（含 prefix-stem 回退、CJK 无 `\b`）
- [x] `orchestrator/direct.go` / `sufficient_context.go` / `query_rewriter.go`
- [x] `session.go`：状态机 + 工具分发 + 四条守卫策略
- [x] `navchain.go`：导航梯子 + run_nav_prefix
- [x] `stats.go`：阶段计时 + 用量统计（等价 CountingChatModel）
- [x] `tool_executor.go`：顶层 `Run` 入口 + 检索/导航工具接线 + 证据发布
- [x] 提示词：`action_run` / `action_initialize_state` / `sca_select` /
      `sca_query_rewrite` 兜底模板（canonical 仍在 `rag/prompts/*.md`）
- [x] **`agentic_rag_graph.py` 外层循环**（planner / prefetch / 槽位研究 /
      SCA↔rewriter 迭代 / 路由守卫）→ `harness/agentic` 包
- [x] Phase 0 `Formalize`（单轮短路 + 关键词压缩，默认关闭）
- [x] Phase 5 答案合成（`_compose_answer_from_evidence` + `kb_prompt` +
      `FINAL_ANSWER_SYSTEM` + `system_prompt` 优先级 + naive 路径）
- [x] `keywords.go`：四面向加权关键词（entity x3 / qualifiers x3）
- [x] `arithmetic.go` + `compute.go`：受限 Python 子集求值器（`calculate` 工具）
- [x] `narrow.go`：grep/sed 证据窄化（含三级回退链）
- [x] `structure_qa`：复用既有 `askStructureSelect` / `askStructureAnswer`
- [x] 编译结构读取：`navigate_tree` 经 `nav.NavService` 真正工作；
      `navigate_structure` / `graph_explore` 已接入
- [x] `loadStructureEntities` 合并两种编译行形状（graph-blob + per-entity/relation），
      经 `ParseCompiledStructure` 接入 reader（对齐 Python `_load_compiled_structure`）
- [x] 9 处与 Python 的行为差异已核验并修复（见 5.0.3）；另移植 `wiki_query` handler 作为未接线扩展 seam（第 10 条）与 `extractRelevantEvidence` 相关度排序（第 11 条）
