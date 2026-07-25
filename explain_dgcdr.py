"""Exact channel attribution for DGCDR recommendations.

Decomposes each recommendation into the model's own disentangled channels
(domain-shared vs domain-specific), reports the transfer ratio tau, and
optionally verbalises the result with an LLM constrained by those numbers.

The decomposition is exact by construction, and the run verifies it: the
channels are summed back and checked against ``model.forward()`` before any
explanation is produced.

Examples:
    python explain_dgcdr.py -m saved/DGCDR-Jul-14-2026_12-22-57.pth
    python explain_dgcdr.py -m saved/model.pth --num_users 50 --topk 5 \
        --metadata item_metadata/meta_CDs_and_Vinyl.jsonl \
        --llm_model qwen2.5:7b
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.attribution import explain_user
from extensions.explainability.channels import (
    SHARED,
    SPECIFIC,
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.metadata import load_catalogue
from extensions.explainability.verbalize import LLMClient, build_prompt, source_history

DEFAULT_MODEL_PATH = "saved/DGCDR-Jul-14-2026_12-22-57.pth"


def get_latest_checkpoint(checkpoint_dir='saved'):
    pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not pth_files:
        return None
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return pth_files[0]


def build_ground_truth(test_data):
    """user id -> list of held-out target-domain item ids."""
    ground_truth = defaultdict(list)
    try:
        dataset = test_data.dataset
        uid_field, iid_field = dataset.uid_field, dataset.iid_field
        users = dataset.inter_feat[uid_field].numpy()
        items = dataset.inter_feat[iid_field].numpy()
        for u, i in zip(users, items):
            ground_truth[int(u)].append(int(i))
    except (AttributeError, KeyError) as exc:
        print(f"[warn] could not read test interactions ({exc}); "
              f"explanations will have no ground truth.")
    return ground_truth


def select_users(model, num_users, seed=42):
    """Sample overlapping users -- the only ones with a defined transfer ratio."""
    candidates = list(range(1, model.overlapped_num_users))  # skip [PAD]
    if num_users and num_users < len(candidates):
        random.Random(seed).shuffle(candidates)
        candidates = sorted(candidates[:num_users])
    return candidates


def summarise(explanations):
    """Aggregate statistics over tau -- the analysis this attribution enables."""
    taus = np.array([a.tau for e in explanations for a in e.attributions])
    if taus.size == 0:
        return {}

    by_rank = defaultdict(list)
    by_history = defaultdict(list)
    for explanation in explanations:
        bucket = len(explanation.history)
        bucket = '0-4' if bucket < 5 else '5-19' if bucket < 20 else '20+'
        for attribution in explanation.attributions:
            by_rank[attribution.rank].append(attribution.tau)
            by_history[bucket].append(attribution.tau)

    return {
        'n_users': len(explanations),
        'n_recommendations': int(taus.size),
        'tau_mean': float(taus.mean()),
        'tau_std': float(taus.std()),
        'tau_median': float(np.median(taus)),
        'tau_p10': float(np.percentile(taus, 10)),
        'tau_p90': float(np.percentile(taus, 90)),
        'share_transfer_driven': float((taus >= 0.6).mean()),
        'share_native_driven': float((taus <= 0.4).mean()),
        'tau_by_rank': {int(r): float(np.mean(v)) for r, v in sorted(by_rank.items())},
        'tau_by_target_history_length': {
            k: {'mean_tau': float(np.mean(v)), 'n': len(v)}
            for k, v in sorted(by_history.items())
        },
    }


def write_markdown(path, explanations, summary, verification, catalogue):
    lines = [
        "# DGCDR — Exact Channel Attribution",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Decomposition validity",
        "",
        "The additive channels are summed back and compared against the model's",
        "own forward pass. This is the guarantee that the attribution is exact",
        "rather than a post-hoc approximation.",
        "",
        f"- user embedding max abs error : {verification['user_embedding_max_abs_error']:.3e}",
        f"- item embedding max abs error : {verification['item_embedding_max_abs_error']:.3e}",
        f"- score max abs error          : {verification['score_max_abs_error']:.3e}",
        f"- score relative error         : {verification['score_relative_error']:.3e}",
        f"- **passed (atol={verification['atol']:.0e})**: {verification['passed']}",
        "",
        "## Transfer ratio statistics",
        "",
        f"- recommendations analysed: {summary.get('n_recommendations', 0)} "
        f"over {summary.get('n_users', 0)} overlapping users",
        f"- tau mean: {summary.get('tau_mean', 0):.4f} (std {summary.get('tau_std', 0):.4f})",
        f"- tau median: {summary.get('tau_median', 0):.4f} "
        f"[p10 {summary.get('tau_p10', 0):.4f}, p90 {summary.get('tau_p90', 0):.4f}]",
        f"- transfer-driven (tau>=0.6): {summary.get('share_transfer_driven', 0) * 100:.2f}%",
        f"- native-driven (tau<=0.4): {summary.get('share_native_driven', 0) * 100:.2f}%",
        "",
        "### Mean tau by target-domain history length",
        "",
        "| history length | mean tau | n |",
        "|---|---|---|",
    ]
    for bucket, stats in summary.get('tau_by_target_history_length', {}).items():
        lines.append(f"| {bucket} | {stats['mean_tau']:.4f} | {stats['n']} |")

    lines += ["", "## Per-user attributions", ""]
    if not catalogue.has_metadata():
        lines += ["> No item metadata loaded: items are shown as internal ids.", ""]

    for explanation in explanations:
        lines += [
            f"### User {explanation.user_id}"
            + (f" (`{explanation.user_token}`)" if explanation.user_token else ""),
            "",
            f"- mean transfer ratio: **{explanation.mean_tau:.3f}**",
            f"- target-domain history: {len(explanation.history)} items",
            f"- held-out items: {explanation.ground_truth}",
            "",
            "| rank | item | score | shared | specific | tau | driven by |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in explanation.attributions:
            lines.append(
                f"| {a.rank} | {catalogue.describe(a.item_id, max_len=60)} | "
                f"{a.score:+.4f} | {a.user_channel_totals.get(SHARED, 0):+.4f} | "
                f"{a.user_channel_totals.get(SPECIFIC, 0):+.4f} | {a.tau:.3f} | "
                f"{a.dominant_channel} |"
            )
        lines.append("")
        for a in explanation.attributions:
            if getattr(a, 'text', None):
                lines += [f"> **#{a.rank}** {a.text}", ""]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Exact channel attribution for DGCDR")
    parser.add_argument('--model_path', '-m', type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to a .pth checkpoint (falls back to the latest in saved/)")
    parser.add_argument('--output_dir', '-o', type=str, default='explainability_out')
    parser.add_argument('--topk', '-k', type=int, default=10,
                        help="Recommendations to attribute per user")
    parser.add_argument('--num_users', '-n', type=int, default=20,
                        help="Overlapping users to sample (0 = all)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_mask_history', action='store_true',
                        help="Do not exclude the user's training items from the ranking")
    parser.add_argument('--metadata', nargs='*', default=None,
                        help="JSONL item metadata files (optional)")
    parser.add_argument('--metadata_id_field', default='parent_asin')
    parser.add_argument('--llm_model', type=str, default=None,
                        help="Model name for verbalisation; omit to skip the LLM stage")
    parser.add_argument('--llm_base_url', type=str, default=None,
                        help="OpenAI-compatible endpoint (default: env or localhost Ollama)")
    parser.add_argument('--llm_max_users', type=int, default=10,
                        help="Verbalise only the first N users, to bound LLM cost")
    parser.add_argument('--llm_max_items', type=int, default=3,
                        help="Verbalise only the top N items per user")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path or not os.path.exists(model_path):
        latest = get_latest_checkpoint('saved')
        if latest:
            print(f"Configured path '{model_path}' not found. Using latest: {latest}")
            model_path = latest
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at '{model_path}' and none in 'saved/'.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data and model from {model_path}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
    model.eval()

    print("Decomposing target-domain embeddings into additive channels...")
    decomposition = decompose_target_domain(model)
    print(f"  user channels: {decomposition.user_channel_names}")
    print(f"  item channels: {decomposition.item_channel_names}")

    print("Verifying that the channels reconstruct the model exactly...")
    verification = verify_decomposition(model, decomposition)
    for key, value in verification.items():
        print(f"  {key}: {value}")
    if not verification['passed']:
        raise RuntimeError(
            "Channel decomposition does not reconstruct the model's own outputs. "
            "The attribution would not be faithful; refusing to emit explanations."
        )

    catalogue = load_catalogue(dataset, args.metadata, id_field=args.metadata_id_field)
    if args.metadata and not catalogue.has_metadata():
        print("[warn] metadata files given but no records loaded; check --metadata_id_field.")

    ground_truth = build_ground_truth(test_data)
    users = select_users(model, args.num_users, args.seed)
    target_dataset = dataset.target_domain_dataset

    print(f"Attributing top-{args.topk} recommendations for {len(users)} users...")
    explanations = []
    for user_id in users:
        try:
            user_token = target_dataset.id2token(target_dataset.uid_field, user_id)
        except (ValueError, IndexError, KeyError):
            user_token = None
        explanations.append(explain_user(
            model, decomposition, user_id,
            topk=args.topk,
            mask_history=not args.no_mask_history,
            user_token=user_token,
            ground_truth=ground_truth.get(user_id, []),
        ))

    summary = summarise(explanations)

    if args.llm_model:
        print(f"Verbalising with {args.llm_model}...")
        client = LLMClient(args.llm_model, base_url=args.llm_base_url)
        source_name = config['source_domain']['dataset']
        target_name = config['target_domain']['dataset']
        for explanation in explanations[:args.llm_max_users]:
            src_hist = source_history(model, explanation.user_id, limit=20)
            for attribution in explanation.attributions[:args.llm_max_items]:
                prompt = build_prompt(
                    attribution, catalogue, explanation.history, src_hist,
                    source_domain_name=source_name, target_domain_name=target_name)
                try:
                    attribution.text = client.generate(prompt)
                except RuntimeError as exc:
                    print(f"  [warn] {exc}")
                    attribution.text = None
                    break

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"attribution_{os.path.basename(model_path).replace('.pth', '')}_{timestamp}"

    json_path = os.path.join(args.output_dir, f"{stem}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_checkpoint': model_path,
            'source_domain': config['source_domain']['dataset'],
            'target_domain': config['target_domain']['dataset'],
            'fuse_mode': model.fuse_mode,
            'attention_mode': model.attention_mode,
            'verification': verification,
            'summary': summary,
            'explanations': [
                dict(e.to_dict(),
                     recommendations=[
                         dict(a.to_dict(), text=getattr(a, 'text', None))
                         for a in e.attributions])
                for e in explanations
            ],
        }, f, indent=2)

    md_path = os.path.join(args.output_dir, f"{stem}.md")
    write_markdown(md_path, explanations, summary, verification, catalogue)

    print(f"\nmean transfer ratio: {summary.get('tau_mean', 0):.4f}")
    print(f"JSON report: {json_path}")
    print(f"MD report:   {md_path}")


if __name__ == '__main__':
    main()
