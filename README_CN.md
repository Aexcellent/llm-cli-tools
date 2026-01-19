# LLM CLI 工具集

一个用于大语言模型（LLM）推理、评估和数据处理的综合工具包。

## 特性

- **统一推理与评判**：支持单轮和多轮推理与评估的单一命令行工具
- **模型评估**：灵活的评估框架，支持自定义指标和对比功能
- **数据处理**：用于合并、转换和准备训练数据（SFT、DPO）的工具
- **批量处理**：使用 ThreadPoolExecutor 实现高效的并行处理
- **格式支持**：自动检测和处理 JSON 和 JSONL 格式

## 安装

### 从源码安装

```bash
git clone https://github.com/aexcellent/llm-cli-tools.git
cd llm-cli-tools
pip install -e .
```

### 从 PyPI 安装（即将推出）

```bash
pip install llm-cli-tools
```

## 使用方法

### 通用大模型参数

以下参数适用于与大模型 API 交互的工具：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（如：qwen-plus, gpt-4, deepseek-chat） | 必需 |
| `--api-key` | API 密钥（未指定则从环境变量读取） | None |
| `--base-url` | API Base URL（未指定则使用默认值） | None |
| `--temperature` | 生成温度参数 | 0.6 |
| `--max-tokens` | 生成的最大 token 数 | 4096 |
| `--max-workers` | 并发线程数 | 10 |

### LLM 推理与评判 (`llm-cli`)

**模式：**
- `inference`: 单轮推理
- `inference-round`: 多轮推理
- `judge`: 单轮评判
- `judge-round`: 多轮评判

**通用参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式（inference, inference-round, judge, judge-round） | 必需 |
| `--input-path` | 输入文件路径（JSON 或 JSONL 格式） | 必需 |
| `--output-path` | 输出文件路径（未指定则自动生成） | None |
| `--preserve-fields` | 从输入数据中保留到结果中的字段列表（逗号分隔） | None |

**多轮模式参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rounds` | 每个样本获取结果的轮数 | 1 |

**评判模式参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt-file` | 自定义评判 Prompt 文件路径 | None |
| `--skip-no-output` | 跳过 output 为 None 的样本 | False |
| `--save-original` | 保存原始输入和输出到结果文件中 | False |

**示例：**

运行单轮推理：
```bash
llm-cli inference \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/output.jsonl \
  --api-key YOUR_API_KEY
```

运行多轮推理：
```bash
llm-cli inference-round \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/output.jsonl \
  --rounds 3 \
  --api-key YOUR_API_KEY
```

运行单轮评判：
```bash
llm-cli judge \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --api-key YOUR_API_KEY
```

运行多轮评判：
```bash
llm-cli judge-round \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --rounds 3 \
  --api-key YOUR_API_KEY
```

### 模型评估 (`llm-eval`)

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-path` | 输入测试文件（JSON 或 JSONL 格式） | test_data.json |
| `--output-path` | 结果输出文件路径 | outputs.jsonl |
| `--api-url` | API URL | http://localhost:8101/v1/chat/completions |
| `--timeout` | API 请求超时时间（秒） | 300 |
| `--json-mode` | 强制 JSON 输出 | False |
| `--result-key` | 模型输出中的结果字段名（支持嵌套字段如 'prediction,class'） | auditresult |
| `--expected-key` | 测试数据中的期望值字段名（支持嵌套字段如 'output,label'） | auditresult |
| `--eval-mode` | 评估模式：binary, multiclass, regression, 或 exact match | binary |
| `--only-infer` | 仅运行推理不进行评估 | False |
| `--only-eval` | 仅评估已有结果 | False |
| `--resume` | 跳过已存在的 trace_id | False |
| `--limit` | 限制测试用例数量 | None |
| `--verbose` | 显示详细日志 | False |

**示例：**
```bash
llm-eval \
  --input-path data/input.jsonl \
  --output-path results/evaluation.jsonl \
  --model gpt-4 \
  --api-key YOUR_API_KEY \
  --eval-mode binary \
  --result-key auditresult \
  --expected-key auditresult
```

### 合并 JSONL 文件 (`llm-merge`)

**参数：**
| 参数 | 说明 |
|------|------|
| `input_files` | 输入文件路径（多个文件，可以是 JSON 或 JSONL 格式） |
| `--output-path` | 输出文件路径（根据扩展名自动选择格式） |
| `--dedupe` | 根据指定键去重（例如：--dedupe id） |
| `--dedupe-all` | 根据内容完全去重 |
| `--verbose` | 显示详细统计信息 |

**示例：**
```bash
llm-merge \
  file1.jsonl file2.jsonl file3.json \
  --output-path merged.jsonl \
  --dedupe id \
  --verbose
```

### 清理失败数据 (`llm-clean`)

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_file` | 输入文件路径（JSON 或 JSONL 格式） | 必需 |
| `--check-fields` | 要检查的字段列表（逗号分隔），值为 None、False 或 "null" 时删除 | output |
| `--output-path` | 输出文件路径（不指定则使用原文件名添加 _cleaned 后缀） | None |
| `--overwrite` | 覆盖原文件（会自动创建备份） | False |
| `--verbose` | 显示详细信息 | False |

**示例：**
```bash
# 使用默认检查字段（output）
llm-clean input.jsonl --output-path cleaned.jsonl

# 检查多个字段
llm-clean input.jsonl --check-fields output,score --output-path cleaned.jsonl --verbose
```

### 转换为 SFT 数据 (`llm-convert`)

**参数：**
| 参数 | 说明 |
|------|------|
| `input_files` | 输入文件路径（多个文件，可以是 JSON 或 JSONL 格式） |
| `--output-path` | 输出文件路径（可多次使用以指定多个输出文件） |
| `--verbose` | 显示详细处理信息 |

**示例：**
```bash
llm-convert \
  input.jsonl \
  --output-path data/sft.jsonl \
  --verbose
```

### 构建 DPO 数据 (`llm-dpo`)

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `score_file` | 分数文件路径（JSON 或 JSONL 格式） | 必需 |
| `ref_file` | 参考文件路径（包含 messages 和 output） | 必需 |
| `--output-path` | 输出文件路径 | 必需 |
| `--min-margin` | 最小分差阈值 | 20.0 |
| `--min-chosen-score` | 正样本最低分数阈值 | 60.0 |
| `--save-filtered` | 保存被过滤的样本日志到指定文件 | None |
| `--id-key` | ID 字段名 | id |
| `--round-key` | 轮次字段名 | round |
| `--verbose` | 显示详细统计信息 | False |

**注意：** 对于每个样本 ID，得分最高的输出作为 `chosen`，其他满足分差阈值的输出都作为 `rejected`。这意味着同一样本可以生成多条 DPO 数据，它们的 `instruction`、`input` 和 `chosen` 相同，但 `rejected` 不同。

**示例：**
```bash
llm-dpo \
  score_data.jsonl \
  ref_data.jsonl \
  --output-path data/dpo.jsonl \
  --min-margin 15 \
  --min-chosen-score 70 \
  --verbose
```

### 对比模型指标 (`llm-compare`)

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--current-model-output` | 当前模型输出文件路径（JSONL 格式） | 必需 |
| `--evaluation-files` | 评估详情文件（包含 ground truth 和其他模型预测） | 必需 |
| `--difficulty-files` | 难度文件（用于映射和加权） | 必需 |
| `--result-key` | 模型输出中的结果字段名 | auditresult |
| `--eval-mode` | 评估模式：binary, multiclass, 或 regression | binary |
| `--trace-id-key` | trace ID 字段名 | trace_id |
| `--difficulty-key` | 难度字段名 | difficulty |
| `--evaluations-key` | 评估字段名 | evaluations |
| `--ground-truth-key` | 评估中的 ground truth 字段名 | ground_truth |
| `--predicted-key` | 评估中的预测值字段名 | predicted |
| `--output-path` | 结果输出文件路径（JSON 格式） | None |
| `--model-name` | 当前模型的自定义名称 | current_model |
| `--log-level` | 日志级别：DEBUG, INFO, WARNING, ERROR | INFO |

**示例：**
```bash
llm-compare \
  --current-model-output current_model.jsonl \
  --evaluation-files eval1.jsonl eval2.jsonl \
  --difficulty-files diff1.jsonl diff2.jsonl \
  --output-path comparison.jsonl \
  --eval-mode binary
```

## 项目结构

```
llm-cli-tools/
├── llm_cli_tools/
│   ├── cli/              # 命令行界面工具
│   │   └── llm_cli_unified.py
│   ├── eval/             # 评估和对比工具
│   │   ├── llm_eval.py
│   │   └── compare_models_metrics.py
│   ├── data/             # 数据处理工具
│   │   ├── merge_jsonl.py
│   │   ├── convert2sftdata.py
│   │   └── build_dpo.py
│   └── utils/            # 通用工具
│       ├── file_utils.py
│       ├── nested_utils.py
│       └── normalize.py
├── pyproject.toml
└── README.md
```

## 配置

### API 密钥

将 OpenAI API 密钥设置为环境变量：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

或通过命令行参数直接传递：
```bash
llm-cli inference --api-key YOUR_API_KEY ...
```

### 自定义评判提示

您可以通过修改源代码中的 `DEFAULT_JUDGE_PROMPT` 或传递自定义提示文件来自定义评判提示。

## 依赖要求

- Python 3.8+
- openai>=1.0.0
- requests>=2.31.0
- tqdm>=4.65.0

## 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 联系方式

- 作者：jiangdeyou
- 邮箱：jiangdeyou@inspur.com
- GitHub：https://github.com/aexcellent/llm-cli-tools

## 致谢

- 基于 [OpenAI API](https://openai.com/) 构建
- 灵感来源于各种 LLM 评估框架
