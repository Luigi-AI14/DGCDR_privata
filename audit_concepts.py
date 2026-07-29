"""Semantic audit of DGCDR's disentangled item subspaces.

Contribution 2. For each domain the item subspace is clustered, an LLM names the
clusters blind, a second LLM checks the names identify their clusters, and then
the two domains' concepts are matched against the correspondence the model
itself encodes.

The whole thing runs twice: on the `shared` channel and on the `base` channel.
Base is the raw LightGCN embedding, with no disentanglement at all. If the two
score the same, the disentanglement bought nothing semantic and any
correspondence was already in the collaborative structure.

Naming and judging use different models on purpose: a model grading its own
labels measures its own consistency. Ollama swaps them, so 12 GB is enough.

    python audit_concepts.py -m saved/DGCDR-Jul-25-2026_11-32-27.pth

One caveat the numbers cannot show. DGCDR's alignment loss (cl_sim_weight) acts
on the *user* common features; there is no cross-domain alignment term on items.
Any item-level correspondence is therefore indirect, inherited through the
shared user space.
"""

import argparse
import json
import os
from datetime import datetime

import torch

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.channels import (
    decompose_domain,
    verify_decomposition,
)
from extensions.explainability.concept_eval import (
    cross_domain_matching,
    graded_matching,
    label_validity,
)
from extensions.explainability.concept_naming import name_concepts
from extensions.explainability.concepts import (
    concept_centroids,
    extract_concepts,
    geometric_pairing,
    structure_correlation,
)
from extensions.explainability.llm import OllamaClient
from extensions.explainability.metadata import load_catalogue


def audit_channel(model, decompositions, channel, catalogue, namer, judge, args):
    """Extract, name, validate and match one channel. Returns a report dict."""
    print(f"\n{'=' * 62}\nCANALE: {channel}\n{'=' * 62}")

    concepts, centroids = {}, {}
    for domain in ('source', 'target'):
        found = extract_concepts(model, decompositions[domain], channel, domain,
                                 n_clusters=args.n_clusters, seed=args.seed)
        concepts[domain] = found
        centroids[domain] = concept_centroids(decompositions[domain], found, channel)
        sizes = [c.size for c in found]
        print(f"  {domain}: {len(found)} cluster, dimensioni {min(sizes)}-{max(sizes)}")

    print(f"\nNaming con {args.namer}...")
    for domain in ('source', 'target'):
        name_concepts(concepts[domain], catalogue, namer, n_items=args.n_items)

    print(f"\nValidita' delle etichette, con {args.judge}...")
    validity = {}
    for domain in ('source', 'target'):
        print(f"  {domain}:")
        validity[domain] = label_validity(concepts[domain], catalogue, judge,
                                          n_way=args.n_way)

    print("\nCorrispondenza cross-domain...")
    pairing = geometric_pairing(centroids['source'], centroids['target'])
    matching = cross_domain_matching(concepts['source'], concepts['target'],
                                     pairing, judge, n_way=args.n_way)

    print("
Corrispondenza, misura graduata...")
    graded = graded_matching(concepts['source'], concepts['target'], pairing, judge)

    return {
        'channel': channel,
        'graded_matching': graded,
        'concepts': {d: [c.to_dict() for c in concepts[d]] for d in concepts},
        'geometric_pairing': pairing,
        'label_validity': validity,
        'cross_domain_matching': matching,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic audit of the item subspaces")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--metadata_cache', type=str,
                        default='item_metadata/cache_elec_cloth.json')
    parser.add_argument('--metadata', nargs='*', default=None)
    parser.add_argument('--output_dir', '-o', type=str, default='concept_audit')
    parser.add_argument('--channels', nargs='*', default=['shared', 'base'],
                        help="Channels to audit; 'base' is the no-disentanglement control")
    parser.add_argument('--n_clusters', type=int, default=12)
    parser.add_argument('--n_items', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--namer', default='qwen3.5:9b')
    parser.add_argument('--judge', default='gemma4:latest')
    parser.add_argument('--host', default='http://localhost:11434')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_path}...")
    config, model, dataset, _, _, _ = load_data_and_model(args.model_path)
    model.eval()

    decompositions = {d: decompose_domain(model, d) for d in ('source', 'target')}
    for domain, decomposition in decompositions.items():
        verification = verify_decomposition(model, decomposition)
        if not verification['passed']:
            raise RuntimeError(f"Decomposition of the {domain} domain does not "
                               f"reconstruct the model.")
    print(f"  decomposizione verificata su entrambi i domini")

    observed, null = structure_correlation(model, decompositions['target'],
                                           'shared', 'specific', 'target')
    print(f"  struttura shared vs specific (item, target): {observed:.4f} "
          f"(null {null:.4f})")
    if observed > 0.95:
        print("  [ATTENZIONE] i due canali item inducono la stessa struttura: "
              "confrontarli fra loro non avrebbe senso.\n  Il confronto fra "
              "domini, che e' quello che facciamo, resta valido.")

    catalogue = load_catalogue(dataset, args.metadata, cache_path=args.metadata_cache)
    if not catalogue.has_metadata():
        raise RuntimeError(f"Nessun metadato: manca {args.metadata_cache}.")
    print(f"  catalogo: {len(catalogue.records)} item")

    namer = OllamaClient(args.namer, host=args.host, think=False)
    judge = OllamaClient(args.judge, host=args.host, think=True)

    reports = [audit_channel(model, decompositions, channel, catalogue,
                             namer, judge, args) for channel in args.channels]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"audit_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_checkpoint': args.model_path,
            'namer': args.namer,
            'judge': args.judge,
            'structure_correlation': {'observed': observed, 'null': null},
            'channels': reports,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 62}\nRIEPILOGO\n{'=' * 62}")
    print(f"{'canale':<10} {'validita src':>13} {'validita tgt':>13} {'corrispondenza':>16}")
    for report in reports:
        v = report['label_validity']
        m = report['cross_domain_matching']
        g = report['graded_matching']
        fmt = lambda x: f"{x * 100:.1f}%" if x is not None else "n/d"
        print(f"{report['channel']:<10} {fmt(v['source']['accuracy']):>13} "
              f"{fmt(v['target']['accuracy']):>13} {fmt(m['agreement']):>16}")
        if g.get('n_pairs'):
            print(f"{'':<10} graduata: {g['mean_geometric']:.2f} vs "
                  f"{g['mean_control']:.2f} controllo, differenza "
                  f"{g['mean_difference']:+.2f} (p = {g['p_paired']:.4f}, "
                  f"n = {g['n_pairs']})")
    print(f"\ncaso: {100.0 / args.n_way:.1f}%")
    print(f"JSON: {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
