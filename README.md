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

### LLM Inference and Judgment (`llm-cli`)

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
  --num-rounds 3 \
  --api-key YOUR_API_KEY
```

Run single-round judgment:
```bash
llm-cli judge \
  --model gpt-4 \
  --data-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --api-key YOUR_API_KEY
```

Run multi-round judgment:
```bash
llm-cli judge-round \
  --model gpt-4 \
  --data-path data/input.jsonl \
  --write-path results/inference.jsonl \
  --output-path results/judgment.jsonl \
  --num-rounds 3 \
  --api-key YOUR_API_KEY
```

### Model Evaluation (`llm-eval`)

Evaluate model outputs against expected results:
```bash
llm-eval \
  --input-path results/output.jsonl \
  --expected-field expected \
  --output-field generated \
  --output-path results/evaluation.jsonl
```

### Merge JSONL Files (`llm-merge`)

Merge multiple JSON or JSONL files:
```bash
llm-merge \
  --input-files file1.jsonl file2.jsonl file3.json \
  --output-path merged.jsonl
```

### Convert to SFT Data (`llm-convert`)

Convert data to Supervised Fine-Tuning (SFT) format:
```bash
llm-convert \
  --input-path data/input.jsonl \
  --output-path data/sft.jsonl \
  --input-field prompt \
  --output-field response
```

### Build DPO Data (`llm-dpo`)

Build Direct Preference Optimization (DPO) training data:
```bash
llm-dpo \
  --input-path data/input.jsonl \
  --output-path data/dpo.jsonl \
  --chosen-field chosen \
  --rejected-field rejected
```

### Compare Model Metrics (`llm-compare`)

Compare metrics across multiple model outputs:
```bash
llm-compare \
  --input-files model1_results.jsonl model2_results.jsonl \
  --output-path comparison.jsonl
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
