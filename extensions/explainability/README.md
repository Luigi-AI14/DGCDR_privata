# Exact Channel Attribution for DGCDR

Contribution 1 of the explainability framework: decompose a DGCDR
recommendation into the disentangled channels the model already uses, exactly,
and use that decomposition — not the LLM — as the source of truth for the
explanation.

## Why this is not a post-hoc explainer

DGCDR fuses the disentangled preferences additively when `fuse_mode='attention'`
(`DGCDR.fuse_and_update`):

```
attention_mode='all'   e = e_gnn + a_c * e^c + a_s * e^s
attention_mode='part'  e =         a_c * e^c + a_s * e^s
```

Since the final embedding is a sum of independent channels and the score is a
dot product, the score splits **exactly**:

```
score(u, i) = <sum_k U_k[u], sum_l I_l[i]> = sum_{k,l} <U_k[u], I_l[i]>
```

Each recommendation therefore comes with a (user-channel × item-channel) matrix
of signed contributions that provably sums back to the score the model actually
computed. There is no surrogate model, no gradient approximation, no sampling —
unlike LIME/SHAP-style attribution, where the explanation is a fitted
approximation of the model rather than the model itself.

`verify_decomposition` enforces this on every run: the channels are summed and
compared against `model.forward()`, and the pipeline refuses to emit
explanations if the reconstruction fails. On a trained checkpoint the residual
is float32 associativity noise (relative error ~1e-7).

## The transfer ratio

From the per-channel totals we derive, for each (user, item):

```
tau      = |C_shared| / (|C_shared| + |C_specific|)
tau_full = |C_shared| / (|C_base| + |C_shared| + |C_specific|)
```

`tau` measures how much of the *disentangled* preference behind a
recommendation is knowledge transferred from the source domain, as opposed to
taste native to the target domain. `tau_full` additionally discounts the part
of the score explained by the raw collaborative signal alone.

Contributions are signed (a channel can push a score down), so a plain share
would be unbounded near the denominator's zero; absolute-magnitude
normalisation is the standard convention for signed attributions and keeps tau
in [0, 1]. The signed totals and the full matrix are kept in the JSON output so
any other normalisation can be recomputed downstream.

## Usage

```bash
python explain_dgcdr.py -m saved/DGCDR-<timestamp>.pth --num_users 50 --topk 10
```

With item metadata and LLM verbalisation (any OpenAI-compatible endpoint —
local Ollama/vLLM or hosted):

```bash
python explain_dgcdr.py -m saved/model.pth \
    --metadata item_metadata/meta_CDs_and_Vinyl.jsonl \
    --llm_model qwen2.5:7b --llm_base_url http://localhost:11434/v1
```

The LLM stage is optional and strictly downstream: it renders an attribution it
is handed, under an explicit numeric constraint, and never decides *why* an item
was recommended. Without `--llm_model` the run produces the numeric attribution
only.

Outputs go to `explainability_out/`: a JSON record (full contribution matrices,
for downstream analysis) and a readable Markdown report.

## Scope and limitations

- **Requires `fuse_mode='attention'`.** With `concat` the channels pass through
  an MLP and the score is not additively separable; the code raises rather than
  silently returning an approximation. This is the configuration used in the
  paper's reported settings, but it is a real restriction and should be stated
  as such.
- **Overlapping users only.** `DGCDR.disentangle_layer` disentangles only
  overlapping users; for everyone else the fused embedding is the plain GNN
  embedding and tau is undefined. `explain_dgcdr.py` samples from overlapping
  users accordingly.
- **Attribution is not causation.** The decomposition says exactly how the score
  was composed, not what would happen if a channel were removed and the ranking
  recomputed. That counterfactual is a separate question (Contribution 2 of the
  framework).
- Item-side channels exist only when `item_disentangle=True`; otherwise the item
  side is a single channel and the matrix degenerates to one column.

## Files

| file | role |
|---|---|
| `channels.py` | recovers the additive channels; verifies against `forward()` |
| `attribution.py` | contribution matrix, transfer ratio, per-user explanations |
| `metadata.py` | optional item catalogue (JSONL), degrades to raw ids |
| `verbalize.py` | constrained prompt builder + OpenAI-compatible client |
| `../../explain_dgcdr.py` | CLI entry point |

Nothing in this module modifies `dgcdr.py`: the propagation is re-composed from
the model's own primitives (`get_ego_embeddings`, `graph_layer`), so the
training path is untouched and the decomposition is validated against it rather
than assumed to match.
