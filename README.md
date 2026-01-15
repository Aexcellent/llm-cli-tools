# LLM CLI Tools

A comprehensive toolkit for Large Language Model (LLM) inference, evaluation, and data processing.

## Features

- **Unified Inference & Judgment**: Single command-line tool supporting both single-round and multi-round inference and evaluation
- **Model Evaluation**: Flexible evaluation framework with customizable metrics and comparison capabilities
- **Data Processing**: Utilities for merging, converting, and preparing training data (SFT, DPO)
- **Batch Processing**: Efficient parallel processing with ThreadPoolExecutor
- **Format Support**: Automatic detection and handling of both JSON and JSONL formats

## Installation

### From Source

```bash
git clone https://github.com/aexcellent/llm-cli-tools.git
cd llm-cli-tools
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install llm-cli-tools
```

## Usage

### Common LLM Parameters

The following parameters are available for tools that interact with LLM APIs:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name (e.g., qwen-plus, gpt-4, deepseek-chat) | Required |
| `--api-key` | API Key (reads from environment variable if not specified) | None |
| `--base-url` | API Base URL (uses default if not specified) | None |
| `--temperature` | Temperature parameter for generation | 0.6 |
| `--max-tokens` | Maximum number of tokens to generate | 4096 |
| `--max-workers` | Number of parallel threads | 10 |

### LLM Inference and Judgment (`llm-cli`)

**Modes:**
- `inference`: Single-round inference
- `inference-round`: Multi-round inference
- `judge`: Single-round judgment
- `judge-round`: Multi-round judgment

**Common Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Running mode (inference, inference-round, judge, judge-round) | Required |
| `--input-path` | Input file path (JSON or JSONL format) | Required |
| `--output-path` | Output file path (auto-generated if not specified) | None |
| `--preserve-fields` | Fields to preserve from input data (comma-separated) | None |

**Multi-round Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rounds` | Number of rounds to get results for each sample | 1 |

**Judgment Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--write-path` | Path to inference results file for judgment | Required for judge modes |
| `--prompt-file` | Custom judgment prompt file path | None |
| `--skip-no-output` | Skip samples with None output | False |
| `--save-original` | Save original input and output to results | False |

**Examples:**

Run single-round inference:
```bash
llm-cli inference \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/output.jsonl \
  --api-key YOUR_API_KEY
```

Run multi-round inference:
```bash
llm-cli inference-round \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/output.jsonl \
  --rounds 3 \
  --api-key YOUR_API_KEY
```

Run single-round judgment:
```bash
llm-cli judge \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/judgment.jsonl \
  --api-key YOUR_API_KEY
```

Run multi-round judgment:
```bash
llm-cli judge-round \
  --model gpt-4 \
  --input-path data/input.jsonl \
  --output-path results/judgment.jsonl \
  --rounds 3 \
  --api-key YOUR_API_KEY
```

### Model Evaluation (`llm-eval`)

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-path` | Input test file (JSON or JSONL format) | test_data.json |
| `--output-path` | Output file path for results | outputs.jsonl |
| `--api-url` | API URL | http://localhost:8101/v1/chat/completions |
| `--timeout` | API request timeout (seconds) | 300 |
| `--json-mode` | Force JSON output | False |
| `--result-key` | Result field name in model output (supports nested fields like 'prediction,class') | auditresult |
| `--expected-key` | Expected value field name in test data (supports nested fields like 'output,label') | auditresult |
| `--eval-mode` | Evaluation mode: binary, multiclass, regression, or exact match | binary |
| `--only-infer` | Only run inference without evaluation | False |
| `--only-eval` | Only evaluate existing results | False |
| `--resume` | Skip existing trace_id | False |
| `--limit` | Limit number of test cases | None |
| `--verbose` | Show detailed logs | False |

**Example:**
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

### Merge JSONL Files (`llm-merge`)

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `input_files` | Input file paths (multiple files, can be JSON or JSONL format) |
| `--output-path` | Output file path (format auto-detected by extension) |
| `--dedupe` | Deduplicate by specified key (e.g., --dedupe id) |
| `--dedupe-all` | Deduplicate by complete content |
| `--verbose` | Show detailed statistics |

**Example:**
```bash
llm-merge \
  file1.jsonl file2.jsonl file3.json \
  --output-path merged.jsonl \
  --dedupe id \
  --verbose
```

### Clean Failed Data (`llm-clean`)

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_file` | Input file path (JSON or JSONL format) | Required |
| `--check-fields` | Comma-separated list of fields to check (removes if value is None, False, or "null") | output |
| `--output-path` | Output file path (defaults to input filename with _cleaned suffix) | None |
| `--overwrite` | Overwrite original file (creates backup automatically) | False |
| `--verbose` | Show detailed information | False |

**Example:**
```bash
# Clean with default check field (output)
llm-clean input.jsonl --output-path cleaned.jsonl

# Check multiple fields
llm-clean input.jsonl --check-fields output,score --output-path cleaned.jsonl --verbose
```

### Convert to SFT Data (`llm-convert`)

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `input_files` | Input file paths (multiple files, can be JSON or JSONL format) |
| `--output-path` | Output file path (can be specified multiple times for multiple outputs) |
| `--verbose` | Show detailed processing information |

**Example:**
```bash
llm-convert \
  input.jsonl \
  --output-path data/sft.jsonl \
  --verbose
```

### Build DPO Data (`llm-dpo`)

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `score_file` | Score file path (JSON or JSONL format) | Required |
| `ref_file` | Reference file path containing messages and output | Required |
| `--output-path` | Output file path | Required |
| `--min-margin` | Minimum score difference threshold | 20.0 |
| `--min-chosen-score` | Minimum score for chosen samples | 60.0 |
| `--save-filtered` | Save filtered samples log to specified file | None |
| `--id-key` | ID field name | id |
| `--round-key` | Round field name | round |
| `--verbose` | Show detailed statistics | False |

**Example:**
```bash
llm-dpo \
  score_data.jsonl \
  ref_data.jsonl \
  --output-path data/dpo.jsonl \
  --min-margin 15 \
  --min-chosen-score 70 \
  --verbose
```

### Compare Model Metrics (`llm-compare`)

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--current-model-output` | Path to current model outputs JSONL file | Required |
| `--evaluation-files` | Evaluation detail files containing ground truth and predictions | Required |
| `--difficulty-files` | Difficulty files (used for mapping and weighting) | Required |
| `--result-key` | Field name for result key in model output | auditresult |
| `--eval-mode` | Evaluation mode: binary, multiclass, or regression | binary |
| `--trace-id-key` | Field name for trace ID | trace_id |
| `--difficulty-key` | Field name for difficulty | difficulty |
| `--evaluations-key` | Field name for evaluations | evaluations |
| `--ground-truth-key` | Field name for ground truth in evaluations | ground_truth |
| `--predicted-key` | Field name for predicted value in evaluations | predicted |
| `--output-path` | Output file to save results (JSON format) | None |
| `--model-name` | Custom name for the current model | current_model |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR | INFO |

**Example:**
```bash
llm-compare \
  --current-model-output current_model.jsonl \
  --evaluation-files eval1.jsonl eval2.jsonl \
  --difficulty-files diff1.jsonl diff2.jsonl \
  --output-path comparison.jsonl \
  --eval-mode binary
```

## Project Structure

```
llm-cli-tools/
├── llm_cli_tools/
│   ├── cli/              # Command-line interface tools
│   │   └── llm_cli_unified.py
│   ├── eval/             # Evaluation and comparison tools
│   │   ├── llm_eval.py
│   │   └── compare_models_metrics.py
│   ├── data/             # Data processing utilities
│   │   ├── merge_jsonl.py
│   │   ├── convert2sftdata.py
│   │   └── build_dpo.py
│   └── utils/            # Common utilities
│       ├── file_utils.py
│       ├── nested_utils.py
│       └── normalize.py
├── pyproject.toml
└── README.md
```

## Configuration

### API Keys

Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly via command-line arguments:
```bash
llm-cli inference --api-key YOUR_API_KEY ...
```

### Custom Judge Prompts

You can customize the judgment prompt by modifying the `DEFAULT_JUDGE_PROMPT` in the source code or passing a custom prompt file.

## Requirements

- Python 3.8+
- openai>=1.0.0
- requests>=2.31.0
- tqdm>=4.65.0

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Author: Deyou Jiang
- Email: jiangdeyou@inspur.com
- GitHub: https://github.com/aexcellent/llm-cli-tools

## Acknowledgments

- Built with [OpenAI API](https://openai.com/)
- Inspired by various LLM evaluation frameworks
