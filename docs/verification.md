# 本地验证记录

验证日期：2026-11-05（Asia/Shanghai）。

这是实现的端到端验证，每个基准仅运行一个测试任务；不能据此估计完整测试集成功率。

| 任务 | Agent / 模型 | 真实环境 | 步数 | 官方分数 | 成功 |
|---|---|---|---:|---:|---|
| ALFWorld | Codex / gpt-6.1-sol | ALFWorld 0.4.2 / TextWorld 1.7.0 | 5 | 1.0 | True |
| WebShop | Codex / gpt-6.1-sol | WebShop 完整商品库，1,181,430 件商品 | 6 | 1.0 | True |

两次运行均复用当前 ChatGPT 账号，Codex CLI 0.166.0。

条件：ALFWorld 使用 valid_unseen，合法动作列表模式，上限 50 步；WebShop 使用测试目标 0，text_rich、无图像特征，上限 100 步。WebShop 共加载 12,087 条人工目标，索引按官方规则跳过 60 个空文档。

框架源码指纹：`a7e4b9b5bb90fd92954e5fae829af0022c221b73f3806588e8783d5830d0e7d5`。

24 项自动测试通过，覆盖原生 Codex 子进程协议、本地 HTTP Chat Completions 工具循环、登录委托、预算、环境评分、任务筛选、续跑及模型固定。Ruff 检查与格式检查通过，pip check 无依赖冲突。

API 密钥和第三方模型服务的适配代码已测试；这些测试使用本地测试端点/模拟进程，没有使用真实第三方密钥调用远端模型。真实模型验证来自上述账号登录的 Codex 会话。

## ALFWorld

目标：examine the alarmclock with the desklamp.

实际执行动作：

1. `go to desk 1`
2. `go to desk 2`
3. `take alarmclock 1 from desk 2`
4. `go to desk 1`
5. `use desklamp 1`

观测到的模型请求数：6；token 使用量见原始结果（包含缓存输入）。

- [运行清单](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-ready/manifest.json>)
- [任务结果](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-ready/episodes/00000/result.json>)
- [原始轨迹](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-ready/episodes/00000/attempt_001/trajectory.jsonl>)

## WebShop

目标：Instruction: i need a long clip-in hair extension which is natural looking, and price lower than 30.00 dollars

实际执行动作：

1. `search[long clip-in hair extension natural looking under 30]`
2. `click[b08bzm24xr]`
3. `click[description]`
4. `click[< prev]`
5. `click[off black#1b]`
6. `click[buy now]`

观测到的模型请求数：7；token 使用量见原始结果（包含缓存输入）。

- [运行清单](</home/hb/projects/general self-evolving framework/runs/verification/webshop-ready/manifest.json>)
- [任务结果](</home/hb/projects/general self-evolving framework/runs/verification/webshop-ready/episodes/00000/result.json>)
- [原始轨迹](</home/hb/projects/general self-evolving framework/runs/verification/webshop-ready/episodes/00000/attempt_001/trajectory.jsonl>)


## 真实断点续跑核验

上述两个目录均已用相同配置执行 `--resume` 并正常退出。配置、环境与源码指纹校验通过；每个任务仍只有 `attempt_001`，未执行新的环境动作或模型请求。summary 中的模型请求数为原任务累计值。

## 切换 GPT-6 Luna 后的验证

2026-11-05：按用户要求，两个账号运行配置的默认模型改为 `gpt-6-luna`，结果目录分别使用 `alfworld-codex-luna` 和 `webshop-codex-luna`。

ALFWorld 官方测试任务已用 Luna 实际运行一局：实际模型 `gpt-6-luna`，推理强度 `xhigh`，5 步，官方分数 1.0，成功标志 True。WebShop 本次完成默认配置切换，未重新评测；上文 WebShop 实测记录属于 Sol。

[Luna 实际运行结果](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-luna/episodes/00000/result.json>)

## Luna Medium 验证

2026-11-05：两个账号运行配置均设定 `reasoning_effort = "medium"`。ALFWorld 已用这套配置实际运行一局：实际模型 `gpt-6-luna`，记录的推理强度 `medium`，6 步，官方分数 1.0，成功标志 True。

[Luna Medium 实际结果](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-luna-medium/episodes/00000/result.json>)

## 可配置 worker 并行验证

2026-11-05：增加命令行 `--workers N` 和配置 `[run] workers = N`，默认 1。清单版本为 2，允许同一新版实验续跑时调整 worker 数；旧版串行结果保持原样。

- 40 项测试通过，包括 2/4 worker 的任务调度重叠、独立 agent、确定性汇总、中断恢复、worker 数切换、原生协议默认模型固定、WebShop 会话隔离，以及安装的官方 ALFWorld 环境多线程检查。
- 真实模型验证：两个独立的 Codex `gpt-6-luna` / `medium` 会话并行执行两个官方 ALFWorld 测试任务，两题均成功，各 6 步，整批耗时 28.26 秒。两条任务运行时间区间重叠，Codex thread ID 不同。
- 实际续跑：将 workers 从 2 改为 4，复用 2 个已完成任务，新增任务数 0，没有追加尝试目录。
- 已校验原有全量 ALFWorld 实验的 136 个清单、汇总和单局结果文件哈希，全部未改变。

[并行运行清单](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-workers-2/manifest.json>) · [并行结果汇总](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-workers-2/summary.json>) · [各次运行记录](</home/hb/projects/general self-evolving framework/runs/verification/alfworld-workers-2/invocations.jsonl>)

- WebShop 额外环境集成检查：完整商品库 1,181,430 件商品，两个 worker 并发重置并执行官方 Lucene 搜索；目标描述、会话隔离与独立清理均通过，模型调用次数为 0。此检查不属于模型成功率评测。[检查记录](</home/hb/projects/general self-evolving framework/runs/verification/webshop-worker-environment.json>)
