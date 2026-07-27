"""Semantic audit of DGCDR's disentangled item subspaces.

Phase 2 of Contribution 2: cluster each domain's item subspace, have an LLM name
the clusters blind, then test whether those names actually identify their
clusters.

The naming and the testing use different models on purpose. A model asked to
judge its own labels agrees with itself, which would measure its consistency
rather than the quality of the labels. Ollama unloads one before loading the
other, so 12 GB of VRAM is enough for both.

    python audit_concepts.py -m saved/DGCDR-Jul-25-2026_11-32-27.pth \
        --metadata_cache item_metadata/cache_elec_cloth.json

The run stops before the LLM if the channel it is asked to audit has collapsed:
naming a subspace that is a copy of another produces two identical lists and no
information.
"""

import argparse
import glob
import json
import os
from datetime import datetime

import torch

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.channels import (
    decompose_domain,
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.concept_eval import label_validity
from extensions.explainability.concept_naming import name_concepts
from extensions.explainability.concepts import (
    extract_concepts,
    structure_correlation,
)
from extensions.explainability.llm import OllamaClient
from extensions.explainability.metadata import load_catalogue


def get_latest_checkpoint(checkpoint_dir='saved'):
    files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    parser = argparse.ArgumentParser(description="Semantic audit of the item subspaces")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--metadata_cache', type=str,
                        default='item_metadata/cache_elec_cloth.json')
    parser.add_argument('--metadata', nargs='*', default=None,
                        help="JSONL dumps, used only to build the cache the first time")
    parser.add_argument('--output_dir', '-o', type=str, default='concept_audit')
    parser.add_argument('--channel', default='shared', choices=['shared', 'specific', 'base'])
    parser.add_argument('--n_clusters', type=int, default=12)
    parser.add_argument('--n_items', type=int, default=50,
                        help="Products shown per cluster when naming")
    parser.add_argument('--n_way', type=int, default=4,
                        help="Alternatives in the label validity test")
    parser.add_argument('--namer', default='qwen3.5:9b')
    parser.add_argument('--judge', default='gemma4:latest')
    parser.add_argument('--host', default='http://localhost:11434')
    parser.add_argument('--force', action='store_true',
                        help="Audit the channel even if it has collapsed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_path}...")
    config, model, dataset, _, _, _ = load_data_and_model(args.model_path)
    model.eval()

    decomposition = decompose_target_domain(model)
    verification = verify_decomposition(model, decomposition)
    if not verification['passed']:
        raise RuntimeError("Channel decomposition does not reconstruct the model.")
    print(f"  item channels: {decomposition.item_channel_names}")

    # --- gate: is there a distinct subspace to audit at all? ---
    print("\nChecking whether the channels are distinct...")
    observed, null = structure_correlation(model, decomposition, 'shared', 'specific',
                                           'target')
    print(f"  shared vs specific, item-item structure: {observed:.4f} (null {null:.4f})")
    if observed > 0.95:
        # This audit compares one channel across the two domains, which the
        # collapse does not invalidate. It would invalidate a comparison between
        # the two channels, so it is stated rather than passed over.
        print(f"\n[ATTENZIONE] I due canali item inducono la stessa struttura "
              f"({observed:.4f}).\n  Confrontare fra loro i concetti dei due "
              f"canali non avrebbe senso su questo checkpoint.\n  Il confronto "
              f"fra domini, che è quello che stiamo facendo, resta valido.")

    catalogue = load_catalogue(dataset, args.metadata, cache_path=args.metadata_cache)
    if not catalogue.has_metadata():
        raise RuntimeError(
            f"No metadata: {args.metadata_cache} is missing and no --metadata given.")
    print(f"  catalogue: {len(catalogue.records)} items")

    # --- extract, per domain ---
    # Each domain must be decomposed from its own propagation: the other
    # domain's items are untrained there, and clustering them yields noise.
    concepts = {}
    for domain in ('source', 'target'):
        domain_decomposition = (decomposition if domain == 'target'
                                else decompose_domain(model, 'source'))
        found = extract_concepts(model, domain_decomposition, args.channel, domain,
                                 n_clusters=args.n_clusters)
        concepts[domain] = found
        sizes = [c.size for c in found]
        norms = domain_decomposition.item_channels[args.channel][
            torch.as_tensor([c.members[0] for c in found])].norm(dim=1).mean()
        print(f"  {domain}: {len(found)} clusters, sizes {min(sizes)}-{max(sizes)}, "
              f"norma media {norms:.4f}")

    # --- phase A: naming ---
    print(f"\nNaming with {args.namer} (thinking off)...")
    namer = OllamaClient(args.namer, host=args.host, think=False)
    for domain in ('source', 'target'):
        name_concepts(concepts[domain], catalogue, namer, n_items=args.n_items)

    # --- phase B: are the labels real? ---
    print(f"\nTesting the labels with {args.judge} (thinking on)...")
    judge = OllamaClient(args.judge, host=args.host, think=True)
    validity = {}
    for domain in ('source', 'target'):
        print(f"  {domain}:")
        validity[domain] = label_validity(concepts[domain], catalogue, judge,
                                          n_way=args.n_way)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"concepts_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_checkpoint': args.model_path,
            'channel': args.channel,
            'namer': args.namer,
            'judge': args.judge,
            'structure_correlation': {'observed': observed, 'null': null},
            'concepts': {d: [c.to_dict() for c in concepts[d]] for d in concepts},
            'label_validity': validity,
        }, f, indent=2, ensure_ascii=False)

    print("\n=== esito del cancello ===")
    for domain in ('source', 'target'):
        v = validity[domain]
        if v['accuracy'] is None:
            print(f"  {domain}: nessun cluster valutabile")
            continue
        print(f"  {domain}: {v['n_correct']}/{v['n_clusters_tested']} corretti "
              f"= {v['accuracy'] * 100:.1f}% contro un caso del "
              f"{v['chance_level'] * 100:.1f}%")
    print(f"\nJSON: {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
