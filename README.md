
# MedLA: A Multi-Agent Framework for Knowledge-Intensive Medical Reasoning (AAAI 2026 Oral)

Adama is a research and engineering framework for multi-agent collaborative reasoning on knowledge-intensive medical and biomedical question answering tasks. It supports datasets such as MedQA, PubMedQA, MMLU medical subsets, MedDdx differential diagnosis style data, BioASQ-like recall tasks, and internal knowledge-grounded cases. This repository accompanies the AAAI 2026 oral paper introducing a scalable agent architecture that integrates logical critique, iterative elimination, and arbitration to improve reliability on difficult clinical and biomedical questions.

Core idea: Multiple specialized medical reasoning agents independently produce answers + a LogicAgent generates structured logical assessments + a DecisionMakersAgent arbitrates conflicting candidates -> final robust answer. The framework contrasts single-pass baselines against multi-round collaborative refinement.

---
## Key Capabilities
- Configurable multi-agent debate: control agent count (`--num_agents`) and discussion depth (`--num_rounds`).
- Logic quality layer: LogicAgent synthesizes reasoning reports and flags inconsistent chains.
- Arbitration layer: DecisionMakersAgent consolidates remaining hypotheses when consensus fails.
- Baseline mode: `--baseline_bool 1` for single-agent direct answering.
- Multi-dataset unified interface (medqa / medxpert / mmlu / pubmedqa / bioasq / medddx ...).
- Integrated experiment tracking (Weights & Biases): accuracy, debate rounds, retries, arbitration frequency.
- Pluggable model backends: OpenAI-style, DeepSeek, vLLM, ZhipuAI, SiliconFlow, etc.
- Threaded execution & service configuration via environment variables.


---
## Repository Layout
```
adam_baseline/              # Classic single-agent baselines & analysis scripts
adama_main_new_prompt/       # Main multi-agent experimental pipeline (current active branch)
  main.py                    # Entry point (baseline + multi-agent)
  util/agent.py              # Agent classes & backend integrations
  util/base.py               # Data loading & invocation orchestration
  util/prompt.py             # Prompt templates (reasoning / elimination / logic)
  sweep/                     # YAML configs for parameter sweeps
  data/                      # Datasets & auxiliary knowledge stores
adama_main_publish/          # Slimmed release version
MedLA_review/                # Review / supplementary materials
```
Outputs (JSONL, logs, logic diagnostics) are written under `output/`, `logic_output/`, and `logic_extrac/`.

---
## Installation
```bash
git clone https://github.com/alexander2618/MedLA.git MedLA
cd MedLA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For full reproducibility: pip install -r requirements_environment.txt
```
Optional vLLM / local serving settings:
```bash
export ADAMA_PORT=8000          # vLLM HTTP server port
export ADAMA_NUM_THREADS=8      # Override --num_threads
```

---
## Data Preparation
Dataset folders reside in their respective `data/` subdirectories. To add a new dataset:
1. Place files under `data/<new_dataset>/`.
2. Extend `--dataset` choices in `get_args()` within `main.py`.
3. Add loading logic in `util/base.py` (parsing, field normalization).

---
## Usage Examples
Multi-agent collaborative run:
```bash
python main.py \
  --dataset medqa \
  --model deepseek \
  --llm_name deepseek-chat \
  --num_agents 8 \
  --num_rounds 3 \
  --few_shots 1 \
  --tag exp_medqa_v1
```
Single-agent baseline:
```bash
python main.py --dataset medqa --model deepseek --llm_name deepseek-chat --baseline_bool 1 --tag baseline_v1
```
vLLM backend:
```bash
export ADAMA_PORT=8000
python main.py --dataset mmlu --model vllm --llm_name llama3.3_4096 --tag vllm_test
```
Debug (small sample + reduced threads):
```bash
python main.py --dataset medqa --debug
```

---
## Main Arguments (main.py)
- --dataset: dataset identifier
- --model: backend selector (deepseek / openai / vllm / siliconflow / zhipuai ...)
- --llm_name: concrete model name
- --num_agents: number of debating medical agents
- --num_rounds: max debate iterations
- --few_shots: number of in-context exemplars
- --baseline_bool: 1 = baseline mode, 0 = multi-agent
- --num_samples: limit processed samples (-1 = all)
- --num_threads: parallelism for data / calls
- --temp: sampling temperature
- --tag: run label for logging
- --debug: enable lightweight run

Environment variables:
- ADAMA_PORT: vLLM HTTP port (required for --model vllm)
- ADAMA_NUM_THREADS: override thread count

---
## Logged Metrics (Weights & Biases)
- accuracy: cumulative accuracy
- talkTimes: debate rounds used per item
- retry_count / retry_pro: backend invocation / parse retries
- make_decision: arbitration triggered flag
- make_decision_num: proportion requiring arbitration
- num_right / numn_total: running correct / total items

---
## Architecture Overview
Process (multi-agent mode):
1. Instantiate LogicAgent and N MedAgent_Eliminate instances.
2. Each agent independently produces (answer, reasoning) using few-shot prompts.
3. LogicAgent evaluates chains -> structured logic_report.
4. If disagreement persists, run elimination / refinement cycles (agents_talk).
5. Remaining conflict -> DecisionMakersAgent consolidates and selects final answer.
6. Metrics persisted & logged.

Principal classes:
- LogicAgent: produces logical critique & coherence assessment
- MedAgent_Eliminate: medical reasoning + self-elimination
- DecisionMakersAgent: arbitration & synthesis
- Agent: generic baseline answering
- base (util/base.py): data orchestration & invoke pipeline

---
## Extensibility
- New backend: implement wrapper in `util/agent.py`, update argument choices.
- New dataset: folder + loader + choices update.
- New prompting strategy: add template in `util/prompt.py` and bind in agent initialization.

---
## FAQ
1. vLLM connection errors: ensure `ADAMA_PORT` matches running server.
2. No WandB logs: run `wandb login` and verify network access.
3. Zero accuracy: confirm dataset name and file presence.
4. Slow multi-agent runs: reduce `--num_agents` or enable `--debug` for profiling.

---
## Results & Analysis
Generated JSONL outputs and diagnostic files live in `output/`. Use `analysis.py` (in `adam_baseline/`) for accuracy curves, debate depth distributions, and comparative evaluation.

---
## Citation
If you use Adama in academic work, please cite:
```
@inproceedings{AdamaAAAI2026,
  title={MedLA: ALogic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models},
  author={Siqi Ma, Jiajie Huang, Fan Zhang, Jinlin Wu, Yue Shen, Guohui Fan, Zhu Zhang, Zelin Zang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  note={Oral Presentation}
}
```

---
## Licensing & Data
This framework interfaces with third-party APIs and medical datasets; verify and comply with each dataset's licensing and usage constraints. Ensure no protected health information (PHI) is introduced during custom data augmentation.

---
## Contributing
Contributions are welcome via Issues and Pull Requests:
- Describe motivation & impact
- Provide minimal reproducible example
- Maintain style & directory conventions

---
## Quick Checklist
- [ ] Install environment & dependencies
- [ ] Prepare datasets in `data/`
- [ ] Configure backend / service variables (if needed)
- [ ] Run baseline or multi-agent example
- [ ] Inspect WandB dashboard & `output/` artifacts

For deeper details, consult source modules in `util/` and associated sweep configs in `sweep/`.
