# CNM-BERT

**A Drop-In Structural Embedding for Chinese Characters via Ideographic Description Sequences**

Reference implementation of the **Compositional Network Model (CNM)** — a lightweight
augmentation that injects *discrete* sub-character structure into Transformer encoders
without modifying the backbone, vocabulary, or output space.

> **Status:** Manuscript submitted to **ACL Rolling Review, 2027**.
> Liqian Yan and Thomas Sing-wing Wu (equal contribution).
> Affiliations: Shanghai Starriver Bilingual School · LinkScape.
> This repository is under active development, and results below are those reported in the
> submitted manuscript.

---

## Motivation

Token-based encoders like BERT treat Chinese characters as atomic identifiers, discarding
their recursive orthographic structure. Models must therefore learn character semantics
indirectly through contextual co-occurrence — and that indirection falls hardest on the
**long tail**: rare characters receive few gradient updates, and standard embedding tables
cannot share parameters across orthographically related characters.

This limitation is **architectural, not scale-bound**. Adding parameters or pre-training
data cannot recover information the input interface has already discarded: an
out-of-vocabulary (`[UNK]`) character stays opaque regardless of model size.

CNM addresses this at the embedding interface. It canonicalizes **Ideographic Description
Sequences (IDS)** into rooted parse trees, encodes each character with a recursive
**Tree-MLP**, and fuses the resulting structural embedding into the standard BERT
embedding layer. Because the structure is symbolic and computed once per *unique character
per batch*, the overhead is roughly **5% training time**.

```
Input: character sequence
        │
        ├── Character Embedding ──────┐
        │                             ├── Fusion Layer ── Transformer Backbone ── MLM Head
        └── Tree-MLP Encoder ─────────┘   (LayerNorm)     (12/24 layers,          (same 21,128
            (IDS parse tree)                               identical to BERT)      vocabulary)
```

The Transformer backbone, vocabulary, and output head remain **identical to vanilla BERT**,
so CNM-BERT is a drop-in replacement in standard fine-tuning pipelines.

---

## Results

All baselines were **re-run from official HuggingFace checkpoints under a single identical
protocol** (five seeds, same grid, same hardware); reproduced baseline numbers fall within
±0.1 of published values on most tasks.

### Primary: structural probing on CCD

The Chinese Character Dataset (CCD; Wu et al., 2025) is an external diagnostic benchmark
for sub-character understanding. On the **OOV slice** — characters mapping to `[UNK]` under
the model's tokenizer — token-only baselines collapse to near chance:

| Model | Structure F1 (OOV) | Radical F1 (OOV) |
|---|---|---|
| Token-only baselines | ≤ 14.0 | ≤ 7.4 |
| SubChar-Wubi | 65.2 | 45.0 |
| ChineseBERT (strongest visual baseline) | 66.2 | 48.4 |
| **CNM-BERT (ours)** | **76.0** | **56.1** |

That is **+9.8 Structure** and **+7.7 Radical** over the strongest baseline (ChineseBERT),
and +10.8 / +11.1 over SubChar-Wubi.

Critically, the gain *grows* with distributional shift — the qualitative signature of an
architectural prior rather than a scaling artifact:

| Split | Structure gain over strongest baseline |
|---|---|
| IID | +1.3 |
| Long-tail | +4.0 |
| OOV | +9.8 |

On visual stroke metrics ChineseBERT remains best, consistent with its pixel-based design.
CNM-BERT and ChineseBERT capture genuinely complementary signals.

### Secondary: general NLU

CNM-BERT achieves the best CLUE average at both scales, establishing that structural
injection is a strict refinement of the token interface — specialized OOV robustness
**without** downstream regression.

| Scale | CLUE avg | vs. strongest external baseline | vs. matched-recipe MacBERT / BERT |
|---|---|---|---|
| base | **71.03** | +0.18 | +0.50 / +2.69 |
| large | **73.56** | +0.19 | — |

Margins over the strongest external baseline are small but consistent across five
fine-tuning seeds and both scales. Also evaluated on MRC (CMRC 2018, DRCD, C³) and NER
(MSRA, OntoNotes 4.0, Weibo).

---

## Installation

Requires Python 3.10–3.11 and PyTorch ≥ 2.2.

```bash
git clone https://github.com/ericyan534-dev/cnm-bert.git
cd cnm-bert

# conda (recommended — matches the training environment)
conda env create -f environment.yml
conda activate cnm

# or pip
pip install -e ".[dev]"
```

## Quickstart

### 1. Build the IDS structural vocabulary

IDS decompositions come from the [BabelStone IDS database](https://babelstone.co.uk/CJK/IDS.TXT),
covering 97,680 CJK Unified Ideographs.

```bash
python scripts/download_ids.py --output-dir data/ids
python scripts/prepare_ids.py \
    --ids-file data/ids/ids_parsed.json \
    --output-dir data/ids \
    --max-depth 6
```

Produces `data/ids/cnm_vocab.json`: a ~5K-component, 12-operator vocabulary with trees of
maximum depth 6. The ~3% of characters without a valid IDS entry fall back to a learnable
embedding.

### 2. Prepare the pre-training corpus

```bash
python scripts/prepare_corpus.py --wiki-dump <path-to-dump> --output-dir data/corpus
```

### 3. Pre-train

```bash
python scripts/pretrain.py \
    --train_file data/corpus \
    --cnm_vocab_path data/ids/cnm_vocab.json \
    --pretrained_bert bert-base-chinese \
    --output_dir outputs/pretrain \
    --mlm_probability 0.15 \
    --aux_loss_weight 0.1 \
    --fp16
```

For multi-GPU training, `scripts/train_8gpu.sh` wraps this in a `torchrun` + tmux session
(8 GPUs, per-device batch 32, gradient accumulation 8).

### 4. Fine-tune and evaluate

```bash
python scripts/finetune.py \
    --model_path outputs/pretrain \
    --task_name tnews \
    --output_dir outputs/finetune-tnews

python scripts/evaluate.py --model_path outputs/finetune-tnews --all_tasks
```

### Tests

```bash
pytest            # unit tests for the model and Tree-MLP
ruff check .      # lint
mypy src/         # type check
```

---

## Method

**Structural representation.** Each Han character is represented by an IDS parse tree whose
internal nodes are Ideographic Description Characters (binary/ternary layout operators) and
whose leaves are Unicode component codepoints. Because IDS decompositions are not unique,
CNM applies a deterministic **canonicalization** — discarding private-use candidates,
resolving intermediate aliases to atomic components, and tie-breaking by tree depth,
operator frequency, node count, and a stable operator hash.

**Tree-MLP encoder.** Structural embeddings are computed bottom-up: leaves embed their
component, internal nodes apply an operator-conditioned MLP with a bounded residual path,
and the root hidden state becomes the character's structural embedding.

**Fusion.** The token embedding and structural embedding are concatenated and linearly
projected back to hidden size, then added to position and segment embeddings as in BERT.
The fusion layer is initialized with Identity+Zero weights to preserve pre-trained
representations at initialization.

**Objective.** Whole-Word-Masked MLM at 15% corruption (80/10/10), with the same corruption
pattern applied to aligned structural indices to prevent gold-structure leakage, plus an
auxiliary component-prediction loss (λ = 0.1).

**Efficiency.** Structural embeddings are computed only for the set of unique Han characters
in the current batch and gathered back to token positions, reducing per-step structural
computation from O(BT) tree evaluations to O(|V_batch|).

## Training configuration

| | base | large |
|---|---|---|
| Initialization | `bert-base-chinese` | `hfl/chinese-roberta-wwm-ext-large` |
| Layers / hidden | 12 / 768 | 24 / 1024 |
| Effective batch | 256 | 512 |
| Hardware / time | 8×A100 80GB, ~7 days | 8×A100 80GB, ~14 days |

Common: 1M steps, LAMB optimizer, peak LR 1×10⁻⁴, 10K warmup steps, FP16. Structural
components use d_s = 256, hidden 512, max tree depth 6. Pre-training corpus is Chinese
Wikipedia plus filtered CommonCrawl (≈4B tokens), with Jieba segmentation for WWM.

## Repository layout

```
src/cnm/
  data/        IDS parsing, canonicalization, tokenizer, tree construction, vocabulary
  model/       CNM configuration, embeddings, Tree-MLP, modeling
  training/    training arguments, losses, whole-word masking, trainer
  evaluation/  CLUE evaluation
scripts/       download_ids, prepare_ids, prepare_corpus, pretrain, finetune, evaluate
configs/       pretrain.yaml, finetune.yaml, model configs
tests/         model and Tree-MLP unit tests
```


## Citation

If you use this work, please cite the manuscript:

```bibtex
@misc{yan2027cnmbert,
  title  = {{CNM-BERT}: A Drop-In Structural Embedding for Chinese Characters
            via Ideographic Description Sequences},
  author = {Yan, Liqian and Wu, Thomas Sing-wing},
  year   = {2027},
  note   = {Manuscript submitted to ACL Rolling Review}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).

IDS data is derived from the [BabelStone IDS database](https://babelstone.co.uk/CJK/IDS.TXT)
and remains subject to its own terms. Evaluation benchmarks (CCD, CLUE, CMRC, DRCD, C³,
MSRA, OntoNotes, Weibo) are the property of their respective authors.
