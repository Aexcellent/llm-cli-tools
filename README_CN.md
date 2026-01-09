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

### LLM 推理与评判 (`llm-cli`)

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
  --num-rounds 3 \
  --api-key YOUR_API_KEY
```

运行单轮评判：
```bash
llm-cli judge \
  --model gpt-4 \
  --data-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --api-key YOUR_API_KEY
```

运行多轮评判：
```bash
llm-cli judge-round \
  --model gpt-4 \
  --data-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --num-rounds 3 \
  --api-key YOUR_API_KEY
```

### 模型评估 (`llm-eval`)

评估模型输出与预期结果的对比：
```bash
llm-eval \
  --input-path results/output.jsonl \
  --expected-field expected \
  --output-field generated \
  --output-path results/evaluation.jsonl
```

### 合并 JSONL 文件 (`llm-merge`)

合并多个 JSON 或 JSONL 文件：
```bash
llm-merge \
  --input-files file1.jsonl file2.jsonl file3.json \
  --output-path merged.jsonl
```

### 转换为 SFT 数据 (`llm-convert`)

将数据转换为监督微调（SFT）格式：
```bash
llm-convert \
  --input-path data/input.jsonl \
  --output-path data/sft.jsonl \
  --input-field prompt \
  --output-field response
```

### 构建 DPO 数据 (`llm-dpo`)

构建直接偏好优化（DPO）训练数据：
```bash
llm-dpo \
  --input-path data/input.jsonl \
  --output-path data/dpo.jsonl \
  --chosen-field chosen \
  --rejected-field rejected
```

### 对比模型指标 (`llm-compare`)

对比多个模型输出的指标：
```bash
llm-compare \
  --input-files model1_results.jsonl model2_results.jsonl \
  --output-path comparison.jsonl
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
