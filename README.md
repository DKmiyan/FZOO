# FZOO: Fast Zeroth-Order Optimizer for Fine‑Tuning Large Language Models towards Adam‑Scale Speed

In this work, we introduce FZOO, a Fast Zeroth-Order Optimizer towards Adam-Scale Speed. On the one hand, FZOO reduces the total forward passes needed for convergence by employing batched one-sided estimates that adapt step-sizes based on the standard deviation of batch losses. On the other hand, it accelerates per-batch computation through the use of Rademacher random vector (±1) perturbations coupled with CUDA's parallel processing capabilities. Extensive experiments on diverse models (including RoBERTa-large, the OPT family (350M-66B), Phi-2, and Llama3) across 11 varied downstream tasks validate FZOO's effectiveness. On average, FZOO outperforms MeZO by +3% in accuracy while requiring 3
 fewer forward passes. Notably, for the RoBERTa-large model, FZOO achieves average improvements of +5.6% in accuracy and 18
 reduction in forward passes compared to MeZO, achieving convergence speeds comparable to Adam. We also provide theoretical analysis proving FZOO’s formal equivalence to a normalized-SGD update rule and establishing its convergence guarantees. Beyond full-parameter tuning, FZOO plugs smoothly into PEFT techniques, unlocking even larger memory savings. Taken together, our results make single-GPU, high-speed, full-parameter fine-tuning realistic today and point toward future work on memory-efficient pre-training.

## Installation

```bash
conda create -n FZOO python==3.9.19
conda activate FZOO
pip install -r requirements.txt
```

This environment can support the **OPT**, **LLaMA**, **Phi** and other latest models.

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/FZOO/MeZO):

```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments.

We provide example script below for reproducing our experiments. All our examples sample 1,000 
training examples, 500 validation examples, and 1,000 testing examples. 

```bash
# FZOO (full-parameter fine-tune OPT-13B on CB dataset)
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b METHOD=FZOO FORWARD_N=8 TASK=WSC MODE=ft LR=5e-5 EPS=1e-4 bash fzoo.sh
# MeZO (full-parameter fine-tune OPT-13B on CB dataset)
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b METHOD=MeZO TASK=WSC MODE=ft LR=1e-6 EPS=1e-3 bash fzoo.sh
```
## License
Released under the MIT License. See [LICENSE](./LICENSE) for details.

