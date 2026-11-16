# sf-framework

可扩展的 agent 任务评测基础框架。当前支持 **A 类步进环境**（官方 ALFWorld、WebShop）和 **B 类 Docker 工作区**。两类任务使用独立配置的评分器，并共用 ACP 和 LLM 轨迹格式。

这一阶段用于建立直接运行现成 agent 的基线。框架尚未实现自进化算法，也没有内置手写的购物/家务技能、动作纠错或额外成功判定。后续自进化模块可以消费相同的轨迹，并实现新的 Agent 接口，与冻结基线比较。

## 当前环境

已创建 Conda 环境 `sf-framework`，Python 3.10。当前机器已安装 Codex CLI 0.166.0，原生集成使用其 app-server JSON-RPC 接口。

```bash
cd "/home/hb/projects/general self-evolving framework"
conda activate sf-framework
sf doctor
sf auth status
```

其他机器可执行：

```bash
conda env create -f environment.yml
conda activate sf-framework
bash scripts/bootstrap.sh
# 需要 Node.js 22+ 和可用的 Java；本机验证使用 Java 21。
npm install -g @openai/codex@0.166.0
sf setup alfworld
sf setup webshop
```

`requirements/benchmarks.txt` 是 Python 3.10 的兼容依赖配置；`requirements/installed.lock.txt` 保存本机安装版本。环境源码、任务数据和奖励函数沿用官方，但依赖版本并非 2022 年 WebShop requirements 的逐项复刻，尤其 spaCy 和英文模型版本可能影响文本匹配细节。每次运行都会保存完整包版本。

## Benchmark 数据目录

| Benchmark | 数据位置 | 当前配置 |
|---|---|---|
| ALFWorld | `data/alfworld/json_2.1.1/` | TextWorld 游戏和任务元数据。`test` 读取 `valid_unseen`，筛选后 134 局。 |
| WebShop | `vendor/WebShop/data/`、`vendor/WebShop/search_engine/indexes/` | 商品、属性和人写指令 JSON，以及全文检索的 Lucene 索引。全量商品目录的 `test` 取 500 道题。 |
| Werewolf | 相邻仓库 `../self-evolving/seeh/environments/werewolf_six.py` | 根据 `configs/werewolf.codex.toml` 中的种子与角色预设生成 36 局。 |
| SkillsBench | `data/skillsbench-v1.1/tasks/` | 官方 v1.1 的 87 个任务包，含环境和评分文件。 |
| SkillFlow | `data/skillflow/test_tasks/` | 官方测试集的 20 个家族、166 个任务包。 |
| SpreadsheetBench 2 | `data/spreadsheetbench2/spreadsheetbench-v2/`、`data/spreadsheetbench2/upstream/evaluation/` | 数据目录含题目索引、输入与参考工作簿，另有官方评分代码。当前配置可跑 297 道题，另外 24 道 Visualization 题的数据也已在本地。 |

评测结果、轨迹和运行产物保存在 `runs/`。本地数据和运行结果不纳入源码版本管理。WebShop 在首次创建索引时使用临时转换文件，建好的 Lucene 索引用于后续评测。

## 账号登录与直接运行

复用 Codex 的登录能力，不自行实现 OAuth、不读取或复制账号 token。已登录时可直接运行：

```bash
sf run --config configs/alfworld.codex.toml --episodes 1 --output runs/alfworld-first
sf run --config configs/webshop.codex.toml --episodes 1 --output runs/webshop-first
```

需要登录时，浏览器或设备码方式任选一种：

```bash
sf auth login --method chatgpt
sf auth login --method chatgpt --device-auth
```

当前 ALFWorld 和 WebShop 的账号运行配置均显式使用 `gpt-6-luna`，推理强度为 `medium`，默认结果目录分别为 `runs/alfworld-codex-luna-medium` 和 `runs/webshop-codex-luna-medium`。框架支持通过 `--model` 覆盖模型；删除配置中的 `agent.model` 后会跟随 Codex 用户默认模型。

长期设置推理强度，在对应 TOML 的 `[agent]` 下填写 `reasoning_effort = "medium"`。命令行 `--reasoning-effort` 优先于配置文件；省略这两处时沿用 Codex 配置。这个项目使用的字段名是 `reasoning_effort`，由适配器传给 Codex 的 `turn/start.effort`。明确指定模型、推理强度：

```bash
sf run --config configs/alfworld.codex.toml \
  --model YOUR_AVAILABLE_MODEL_ID --reasoning-effort medium \
  --episodes 10 --output runs/alfworld-model-a
```

指定的模型不会被框架静默替换。实际 model/provider、推理强度、Codex 版本和 token 使用信息写入结果。可使用 `--codex-home /path/to/separate-codex-home` 复用其他登录目录；框架不会修改全局 Codex 配置。

## API 密钥与非 GPT 模型

**Codex + OpenAI API 密钥**可以在单次运行时选择，无需退出现有 ChatGPT 账号：

```bash
# 在终端安全地设置 OPENAI_API_KEY；不要把密钥写进配置或命令参数。
read -rs OPENAI_API_KEY
export OPENAI_API_KEY
sf run --config configs/alfworld.codex.toml --auth api_key \
  --model YOUR_AVAILABLE_API_MODEL_ID --output runs/alfworld-api

# 同样适用于 WebShop
sf run --config configs/webshop.codex.toml --auth api_key \
  --model YOUR_AVAILABLE_API_MODEL_ID --output runs/webshop-api
```

也提供官方 Codex CLI 的持久化 API 登录入口：

```bash
sf auth login --method api-key --api-key-env OPENAI_API_KEY
```

这条命令会由 Codex 保存登录状态，可能改变当前 Codex 登录模式。想分别保存账号和 API 登录，可给 `sf auth login` 和 `sf run` 设置不同的 `--codex-home`。持久化 API 登录后，运行时使用 `--auth existing`。

**Codex + 第三方模型服务**需要该服务实现 Codex 兼容的 Responses 工具调用协议：

```bash
sf run --config configs/alfworld.codex.toml --auth provider \
  --base-url http://localhost:8000/v1 --api-key-env MODEL_API_KEY \
  --model YOUR_MODEL_ID --output runs/alfworld-codex-custom
```

要复用 Codex 用户配置里已有的提供方，可以复制对应 benchmark 的配置，在副本的 `[agent]` 中设定 `auth = "provider"`、`provider` 和 `model`，其中 `provider` 填已有提供方的名称。

**只支持 OpenAI Chat Completions 的第三方模型**使用单独的 `openai_chat` agent 后端：

```bash
sf run --config configs/alfworld.codex.toml --agent openai_chat \
  --base-url http://localhost:8000/v1 --api-key-env MODEL_API_KEY \
  --model YOUR_MODEL_ID --output runs/alfworld-chat-model

sf run --config configs/webshop.codex.toml --agent openai_chat \
  --base-url http://localhost:8000/v1 --api-key-env MODEL_API_KEY \
  --model YOUR_MODEL_ID --output runs/webshop-chat-model
```

该后端有完整消息历史和 function/tool 调用循环，适用于实现兼容工具调用的服务。它是另一种 agent 基线，结果中的 agent 类型与 Codex 分开记录，不把它标成 Codex 成绩。第三方服务需支持工具调用，不能仅有文本补全能力。

## 执行方式与评分

现有 ALFWorld 和 WebShop 配置默认使用以下设置。

```toml
[execution]
kind = "step"

[evaluation]
kind = "environment"
```

`step` 将 `sf_observe`、`sf_step` 接到官方环境，记录步数和结束状态。默认的 `environment` 评分器读取环境给出的分数和成功标志。

B 类可用 [工作区示例配置](configs/workspace.codex.toml) 实跑。Docker 需要可用，配置的镜像应已在本机。示例使用 `ubuntu:24.04`。

```bash
sf run --config configs/workspace.codex.toml --output runs/workspace-demo
```

`task.kind = "workspace"` 从任务目录读取 `instruction.md`，将可选的 `input/` 拷贝到每局独立的 `workspace/`。`execution.kind = "workspace"` 用 `sf_exec({"command": "..."})` 提供容器内的 Shell。命令从 `/workspace` 开始执行，容器使用只读根文件系统、独立工作区和禁用网络。任务用时间和工具调用预算，`run.max_steps` 留空。最终工作区保存在本局尝试目录中。

执行结束且容器停止后，`evaluation.kind = "file_text"` 读取 `evaluation.options.path` 指定的交付物，以去除首尾空白后的 UTF-8 文本与 `expected` 比较。Agent 的结束状态和评分器状态分别保存在 `result.json` 的 `agent_status` 和 `evaluation` 中。执行后端、评分器和镜像 ID 都计入断点续跑校验。

自定义评分器可将 `evaluation.kind` 设为 `module:factory`。工厂接收 `EvaluationConfig`。评分器实现 `metadata(task_metadata) -> dict`（可提供 `score_range`）和 `evaluate(episode: EvaluationInput) -> Grade`。评分输入包含 agent 结果和本局输出目录，并按执行方式提供环境结果或最终工作区。评分器可以独立于 agent 和执行后端更换。

### SkillsBench 1.1

[SkillsBench 配置](configs/skillsbench.codex.toml) 使用官方 BenchFlow `TaskRuntime` 读取 `task.md`、启动任务 Docker 环境，并在 agent 退出后、容器清理前调用官方 verifier。Codex 通过 `sf_exec` 在任务容器内以 root 执行命令，从任务自己的工作目录开始，以支持题目中要求的 `/root` 等绝对路径。配置默认使用 `gpt-6-luna`、`medium`、ChatGPT 登录，以及 SkillsBench 的 `no-skill` 条件。

准备 Docker Compose、Codex 登录和 Python 3.12 或更新版本。可以在这个 Python 环境中执行 `python -m pip install -e '.[skillsbench]'`。本机已有独立的 BenchFlow 0.6.5 环境，也可直接用其 Python 运行框架，无需更换 ALFWorld 所用的 Python 环境。

```bash
# 从本地官方仓库导出固定 v1.1 的任务包
mkdir -p data/skillsbench-v1.1
git -C /home/hb/projects/skillsbench archive v1.1 tasks | tar -xf - -C data/skillsbench-v1.1

sf tasks --config configs/skillsbench.codex.toml
sf run --config configs/skillsbench.codex.toml \
  --task-id econ-detrending-correlation --task-id dialogue-parser --workers 2 \
  --output runs/skillsbench-codex-demo
```

每局保留 `acp_trajectory.jsonl`、`llm_trajectory.jsonl` 和 `result.json`。`benchflow_rollout` 指向本局的 BenchFlow 目录，内有官方 verifier 的奖励、日志和它导出的产物。官方 verifier 返回的标量奖励写入 `score`，满分 1 记为成功。评分器异常单独记入 `evaluation.status`。

### SkillFlow

[SkillFlow 配置](configs/skillflow.codex.toml) 读取官方数据集的 `test_tasks/<家族>/<任务>`，用 Harbor 启动任务的 Docker 环境并运行原版 `tests/test.sh`。任务 ID 保留家族前缀，家族内优先采用发布包里的任务排序。本配置以独立任务基线运行，每局使用新会话，采用 `gpt-6-luna`、`medium` 和 ChatGPT 登录。

Harbor 依赖 Python 3.12 或更新版本和 Docker Compose。本机的项目 `.venv` 已安装 Harbor。新环境可用 `uv venv --python 3.12 .venv` 和 `uv pip install --python .venv/bin/python -e '.[skillflow]'` 安装。

```bash
# 下载验证用的两个任务包。需要整套数据时去掉 --include 及其两个模式。
hf download zhang-ziao/SkillFlow-Task --repo-type dataset \
  --revision ecaadb0e25d5d5cfd87bd86d81e77b4abe3a00bc --local-dir data/skillflow \
  --include 'test_tasks/HWPX-Document-Automation/hwpx-event-announcement/*' \
            'test_tasks/Medical-Data-Standardization/cardio-panel-template-harmonization/*'

.venv/bin/sf tasks --config configs/skillflow.codex.toml
.venv/bin/sf run --config configs/skillflow.codex.toml \
  --task-id HWPX-Document-Automation/hwpx-event-announcement \
  --task-id Medical-Data-Standardization/cardio-panel-template-harmonization \
  --workers 2 --output runs/skillflow-codex-demo
```

本地从任务的 Dockerfile 构建镜像。若基础镜像 `skillevlove/harbor-cli-openhands:ubuntu24.04` 拉取失败，可以用 [SkillFlow 官方构建脚本](https://github.com/ZhangZi-a/SkillFlow/tree/7b49ff5a7e26cd7706e959bfa0dba4746d18440d/docker/harbor-cli-base) 本地构建，调用时把该镜像名作为第一个参数。每局在尝试目录保留两份框架轨迹和 `harbor/verifier/` 评分日志。

### SpreadsheetBench 2

[SpreadsheetBench 2 配置](configs/spreadsheetbench2.codex.toml) 接入 `Debugging`、`Financial_Model` 和 `Template`。任务 ID 采用 `类别/原版 ID`。任务容器接收官方题目和单个只读输入工作簿，通过 `sf_exec` 生成 `<id>_output.xlsx`。评分容器在任务容器关闭后运行官方 `evaluation.py`，二元准确率及改动、回归准确率写入 `result.json`。LibreOffice 用官方 `open_spreadsheet.py` 重算输入、参考答案和提交物的副本。Visualization 有独立的 Windows Excel/WPS 和视觉模型评分流程。

```bash
hf download KAKA22/SpreadsheetBench-v2 spreadsheetbench-v2-09-10.zip --repo-type dataset \
  --revision 5a2215ed4121945ab09d8723df2995602090b042 --local-dir data/spreadsheetbench2
unzip -n data/spreadsheetbench2/spreadsheetbench-v2-09-10.zip -d data/spreadsheetbench2
git clone https://github.com/RUCKBReasoning/SpreadsheetBench-2.git /tmp/spreadsheetbench2-upstream
mkdir -p data/spreadsheetbench2/upstream
git -C /tmp/spreadsheetbench2-upstream archive a7bdc2d358e052d35a94768e1f487c0d9e9c0011 evaluation \
  | tar -xf - -C data/spreadsheetbench2/upstream
docker build -t sf-spreadsheetbench2:a7bdc2d examples/spreadsheetbench2

sf tasks --config configs/spreadsheetbench2.codex.toml
sf run --config configs/spreadsheetbench2.codex.toml --task-id Template/06_05 \
  --task-id Debugging/01_02 --workers 2 --output runs/spreadsheetbench2-codex-demo
```

镜像沿用官方 Dockerfile 的 Debian Bullseye Python 基础镜像，通过 Debian 归档源安装 LibreOffice Calc 和 UNO。完成的工作簿保存在本局的 `workspace/` 目录。

## A 类任务条件

| 条件 | ALFWorld | WebShop |
|---|---|---|
| 环境 | 官方 TextWorld 游戏，ALFWorld 0.4.2 | 官方 `WebAgentTextEnv`，源码 commit `64fa2a5` |
| 默认测试集 | `valid_unseen`，实际识别 134 个任务 | 官方打乱后的目标索引 `0–499` |
| 验证集 | `valid_seen` | 索引 `500–1499` |
| 训练集 | `train` | 索引 `1500` 及以后 |
| 默认环境动作上限 | 50 | 100 |
| 观察 | 文本 | `text_rich`，不含图像特征，不显示 Attributes 页 |
| 动作信息 | 配置 `admissible` 或 `generation` | 页面公开可点击项及搜索接口 |
| 成功判定 | 环境的 `won` 标志 | 购买终止且官方 reward 为 1 |
| 分数 | 0/1 | 官方原始 reward，0–1；另提供换算到 0–100 的均分 |

ALFWorld 默认配置显式选择 `admissible`，每步给出合法动作列表。若比较使用命令生成的模型，应复制配置并改成 `action_mode = "generation"`，此时不会把合法动作列表交给 agent。

WebShop 默认使用完整数据，绝不在缺文件时自动换成 1000 商品预览版。准备工具支持显式 `--catalog small`，相应运行配置也必须写 `catalog = "small"`；这类结果必须单独报告。

WebShop 完整数据从固定 revision 的 `HongbangYuan/webshop` 镜像下载，三个文件有固定 SHA-256 校验。Lucene 文档构造字段、索引选项与官方代码一致。价格/预算生成前设定固定种子，保存数据与索引哈希，以及每局含实际目标文本的初始观察哈希和轨迹。固定种子有助于本框架内比较，不意味着与每篇历史论文的随机价格条件完全一致。

任何字符串形式的动作都会真正送入官方环境；无效动作也消耗一次环境步数。没有免费重试、自动替换动作、目标答案查询或跨回合记忆。agent 只接触 `sf_observe`、`sf_step` 的公开观察，不能重置游戏。Codex 使用新的临时会话，不提供文件/终端环境，关闭无关 MCP、插件、网页搜索及子 agent 能力，避免访问数据文件中的答案。

步进任务初始化后，首次用户消息直接包含原始 `observation`，环境提供可用动作列表时附带 `available_actions`。Agent 可以直接调用 `sf_step` 行动，后续通过 `sf_observe` 重读当前观察。两种工具的正常返回都附带 `done`，执行提示词要求在 `done` 为 `false` 时继续行动。环境终止或步数预算耗尽时，`done` 变为 `true`。运行结束后可从 `result.json` 查看步数、终止状态和停止原因。调用格式错误时返回 `error`。

### 六人狼人杀

[狼人杀配置](configs/werewolf.codex.toml) 从 `task.root` 加载旧 SEEH 项目的 `SixPlayerWerewolfMatch`。原控制器负责夜间技能、白天发言、秘密投票和胜负判定。在受测席位需要决策时，框架暂停对局。`sf_observe` 返回该席位的观察，包含公开事件和该席位有权知道的私有信息。`sf_step` 提交一条发言或动作，然后环境推进到该席位的下一次决策或整局结束。投票、验人、狼刀和女巫技能使用旧控制器的动作格式与校验，自由发言直接传文本。每次提交都计入步数，旧控制器在非法选人后仍会给一次重试机会。

配置默认使用 `gpt-6-luna` 和 `medium` 控制一席，另外五席以及发言摘要使用 SEEH 的确定性脚本。任务按种子和席位轮换，默认是 `one_wolf` 阵容，也可把 `task.options.role_preset` 设为 `two_wolves`。胜负按受测席位所属阵营计算，阵营获胜得 1 分，落败或达到六天安全上限仍无胜方得 0 分。受测席位出局后，其余席位继续推进到本局结束。结果中的 `evaluation.details` 保存席位、角色、种子、获胜阵营及存活状态。

每次尝试的 `environment_events.jsonl` 按 `seq` 记录从开局到结束的环境事件，包括被测席位出局后的发言和投票。`match_event` 行保留原控制器的 `visibility`，出局对应的事件有 `tested_player_eliminated: true`。`match_finished` 行记录胜方、被测席位和存活状态。框架提前关闭对局时记录 `match_aborted`，此前已发生的事件仍可读取。这份文件用于赛后复盘，判断模型在某次决策前知道什么应读取当时的 Agent 轨迹。

```bash
sf tasks --config configs/werewolf.codex.toml
sf run --config configs/werewolf.codex.toml --task-id one_wolf-0002 \
  --output runs/werewolf-codex-demo
```

## 批量运行与结果

先看实际任务列表：

```bash
sf tasks --config configs/alfworld.codex.toml --limit 10
sf tasks --config configs/webshop.codex.toml --limit 10
```

跑完整测试集时显式使用 `--all`，默认只跑一局，避免把接口验证误当作完整评测：

```bash
sf run --config configs/alfworld.codex.toml --all --output runs/alfworld-full
sf run --config configs/webshop.codex.toml --all --output runs/webshop-full
```

可通过 `--workers N` 设置同时运行的任务数，默认 `1`；也可以在 TOML 的 `[run]` 中设置 `workers = 4`，命令行优先：

```bash
sf run --config configs/alfworld.codex.toml --all --workers 4 \
  --output runs/alfworld-test-luna-medium-w4
sf run --config configs/webshop.codex.toml --all --workers 4 \
  --output runs/webshop-test-luna-medium-w4
```

每个任务使用独立的环境、agent 对话和输出目录；Codex 在不同进程中运行，模型请求可以重叠。ALFWorld 的共享 PDDL 解析器和 WebShop 的短环境操作由锁保护，锁外的模型调用并行执行。WebShop 仅加载一份商品库和索引，各任务使用唯一会话前缀。worker 数只控制并行任务数，每个任务的动作预算和评分条件保持相同。

并行结束顺序可能不同，`summary.json` 始终按任务列表顺序保存，由主调度器统一写入；终端输出带任务和 worker 标识。并行请求仍然共享账号额度，不保证随 worker 数线性加速。

配置中的 `run.task_ids` 可以指定固定任务；也可通过 `--start N --episodes K` 选择连续子集。相同模型预算和任务列表下比较不同 agent。`max_steps` 控制环境动作；`agent.max_tool_calls` 控制工具调用；`agent.max_model_calls` 控制模型请求，Codex 以 token 使用更新通知观测请求数，内部网络重试不单独计数；`timeout_seconds` 控制单局 agent 时间。

```bash
sf run --config configs/alfworld.codex.toml --all --output runs/alfworld-full --resume
# 错误/超时局保留原尝试，在独立目录重试：
sf run --config configs/alfworld.codex.toml --all --output runs/alfworld-full --resume --retry-errors
```

断点续跑校验配置、任务集合、官方数据/索引、框架/插件源码和 Python 包版本，并记录 Codex 用户配置指纹；条件变化时拒绝混入旧结果。未明确指定模型时，也会沿用已完成任务记录中的实际模型，防止默认模型变化后混合成绩。并行派发前统一确定 Codex 的默认模型；一个输出目录只允许一个调度进程。

新版允许在 `--resume` 时仅更改 worker 数，例如从 4 调整到 2；每次调用的 worker 数和实际耗时记录在 `invocations.jsonl`，manifest 保留首次配置。并行功能引入了源码和清单版本变更，旧版实验目录仍保留原结果，新版实验应使用新的输出目录，避免混合不同实现版本。

按 Ctrl+C 时，调度器停止派发新任务，通知正在运行的任务停止并保存部分轨迹，关闭 Codex 进程。下次 `--resume` 会自动重新运行标记为 `interrupted` 的任务，保留旧尝试目录。同步 Chat API 后端需要等当前 HTTP 请求返回或超时后响应中断。

输出结构：

```text
runs/<name>/
  manifest.json                     # 配置、来源、数据哈希、任务列表、实现指纹、依赖版本
  summary.json                      # 成功率、Wilson 区间、均分、步数、错误与停止原因
  invocations.jsonl                 # 每次启动/续跑的 worker 数、耗时和复用任务数
  episodes/00000/
    result.json                     # 当前结果
    attempt_001/
      acp_trajectory.jsonl          # 对话、工具调用及其所属模型请求
      llm_trajectory.jsonl          # 末次请求、最终回复、请求索引及需补存的工具结果
      environment_events.jsonl      # 狼人杀完整对局事件，含可见性和终局
      codex.stderr.log              # Codex 诊断日志
      result.json                    # 执行状态与独立评分结果
      workspace/                     # B 类任务的输入文件和交付物
```

两份 agent 轨迹在单局结束时写入。ACP 工具调用的 `kind` 统一为 `execute`。ACP 的 `llm_call_index` 从 1 开始，与 LLM 轨迹中 `calls[].index` 对应。相同索引表示工具来自同一次模型请求。后一条工具记录的 `reasoning_since_previous_tool` 为 `true` 时表示发现了推理证据，`false` 表示该区间可确认没有单独的推理，`null` 表示首个工具或证据不足。跨模型请求也可能没有推理条目。

Codex 使用本地原生追踪关联模型响应与工具调用，包括同一个执行单元发出的多个工具。末次请求如果使用 WebSocket 增量，会沿响应 ID 展开请求上下文。最终回复在未产生时记为 `null`，缺失的请求上下文也记为 `null`。新版实验使用新的输出目录。

每条 ACP `tool_call` 的 `result_ref` 指向同一 attempt 的 LLM 记录。`source = "messages"` 时，`message_index` 和 `content_index` 是从 0 开始的消息和输出块索引。结果直接保存在字符串字段中时，`content_index` 为 `null`。无法在 `messages` 中唯一定位完整结果时，框架会在 `tool_results` 中补存原始结果，引用采用 `source = "tool_results"` 和 ACP 的 `tool_call_id`。工具未返回结果时，引用为 `null`。

经验提取器可以调用 `sf_framework.agents.trajectory.resolve_tool_result(acp_event, llm_record)` 取得结果对象。旧轨迹没有 `result_ref`，读取函数对这类事件返回 `None`。狼人杀终局观察的 `Public vote history` 包含每轮公开票型、票数和放逐对象。

失败和服务错误保留在结果与统计分母中，不会静默过滤。单局成功只说明该任务成功；不能据此推断完整基准成功率。

## 扩展任务和 agent

核心位于 `src/sf_framework/core.py`。A 类任务提供任务列表、环境构造方法和步数预算。环境实现 `Environment` 接口。

```python
from sf_framework.core import Observation, Transition

class MyEnvironment:
    def reset(self, task_id):
        # 用真实环境初始化指定任务；不要把答案放进观察。
        return Observation(text="...", goal="...", actions=["..."])

    def step(self, action):
        # 执行动作并用真实环境规则判定结果。
        return Transition(observation=..., reward=..., terminated=..., success=...)

    def close(self):
        ...

class MySuite:
    default_max_steps = 80

    def __init__(self, config):
        self.options = config.options

    def task_ids(self):
        return ["task-001", "task-002"]

    def make_environment(self):
        return MyEnvironment()

    def metadata(self):
        return {"name": "my-task", "version": "1", "score_range": [0, 1]}
```

将插件作为可导入模块安装，并在 TOML 中配置：

```toml
[task]
kind = "my_package.my_task:MySuite"
root = "/path/to/data"

[task.options]
difficulty = "hard"
```

任务元数据应包含数据版本/哈希，供断点续跑校验；分数不是 0–1 时设置真实 `score_range`，没有固定区间可省略。新 agent 实现 `run(session, output) -> AgentResult`，通过 `session.call()` 使用同一任务工具，然后设置 `agent.kind = "my_package.my_agent:MyAgent"`。agent 专属参数放入 `[agent.options]`。

这套边界使后续自进化策略可以替换 agent、读取历史训练轨迹、产生新的策略版本，同时保持任务定义、评分和冻结测试集不变。

## 验证与来源

最终真实验证结果、实际动作与原始文件链接见 [docs/verification.md](docs/verification.md)。

```bash
pytest -q
ruff check src tests
uv pip check --python "$(command -v python)"
```

测试覆盖预算、官方成功标志、禁止重置、任务划分、断点续跑、错误保留、原生 Codex 子进程协议，以及本地 HTTP 端点上的第三方模型工具循环。真实账号验证的结果在 `runs/verification/`；不同目录中保留了开发阶段的诊断尝试，正式结果应以各目录 manifest 和 result 为准。

参考官方代码和文档：

- [Codex App Server](https://developers.openai.com/codex/app-server)：原生 agent、动态工具、事件及使用量通知。动态工具目前属于实验接口，升级 CLI 后应重跑协议/真实任务验证。
- [Codex Authentication](https://developers.openai.com/codex/auth)：账号与 API 登录。
- [Codex Advanced Configuration](https://developers.openai.com/codex/config-advanced)：模型提供方、端点与按次配置。
- [ALFWorld 官方仓库](https://github.com/alfworld/alfworld)：文本游戏与任务筛选；数据使用官方 0.4.2 release 的编译游戏文件。
- [WebShop 官方环境与奖励](https://github.com/princeton-nlp/WebShop/tree/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site)。
- [WebShop 官方测试脚本](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/baseline_models/test.py)：500 个测试目标、100 步上限、分数换算。
