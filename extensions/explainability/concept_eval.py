"""Does a label actually identify its cluster?

A label that reads well is not a label that is true. This is the gate of the
semantic audit: the LLM is shown a label and several sets of products, one from
the cluster the label came from and the rest from other clusters, and has to
pick the right one. Accuracy above chance means the label captured the latent
direction; accuracy at chance means it captured nothing and every later
measurement would be noise.

The products used here are the held-out ones, never shown during naming. A label
tested on the items that produced it would only prove the LLM can recognise its
own summary.
"""

import random

VALIDITY_SYSTEM = """You match a description to a group of products.

You are given one description and several groups of products, labelled A, B, C...
Exactly one group matches the description.

Answer with the letter of that group and nothing else."""


def build_validity_prompt(label, candidate_sets, catalogue, n_show=8):
    lines = [f'Description: "{label}"', "", "Which group does it describe?", ""]
    for position, items in enumerate(candidate_sets):
        letter = chr(ord('A') + position)
        lines.append(f"Group {letter}:")
        for item_id in items[:n_show]:
            lines.append(f"  - {catalogue.describe(item_id, max_len=90)}")
        lines.append("")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def label_validity(concepts, catalogue, client, n_way=4, seed=42, verbose=True):
    """Run the N-way test over every labelled cluster.

    Returns a dict with the accuracy, the chance level, and the per-cluster
    outcomes. Clusters without a label, or without enough held-out items, are
    skipped and counted separately rather than scored as failures.
    """
    labelled = [c for c in concepts if c.label and len(c.held_out) >= 3]
    rng = random.Random(seed)

    results, correct, unreadable = [], 0, 0
    for concept in labelled:
        others = [c for c in labelled if c.cluster_id != concept.cluster_id]
        if len(others) < n_way - 1:
            continue
        distractors = rng.sample(others, n_way - 1)

        candidate_sets = [concept.held_out] + [d.held_out for d in distractors]
        order = list(range(n_way))
        rng.shuffle(order)
        shuffled = [candidate_sets[i] for i in order]
        truth = order.index(0)

        prompt = build_validity_prompt(concept.label, shuffled, catalogue)
        choice = client.choose(prompt, shuffled, system=VALIDITY_SYSTEM)

        if choice is None:
            unreadable += 1
            hit = None
        else:
            hit = (choice == truth)
            correct += int(hit)

        results.append({
            'cluster_id': concept.cluster_id,
            'channel': concept.channel,
            'domain': concept.domain,
            'label': concept.label,
            'correct': hit,
        })
        if verbose:
            mark = '?' if hit is None else ('ok' if hit else 'X')
            print(f"  [{mark}] {concept.label}", flush=True)

    scored = len(results) - unreadable
    return {
        'n_clusters_tested': len(results),
        'n_unreadable_answers': unreadable,
        'n_correct': correct,
        'accuracy': correct / scored if scored else None,
        'chance_level': 1.0 / n_way,
        'n_way': n_way,
        'per_cluster': results,
    }
