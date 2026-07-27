"""Blind naming of the concept clusters.

The LLM sees titles and categories and nothing else. Not the domain, not the
channel, not the position in the ranking, not the item ids. If it could tell
what it is looking at, its answer would describe our setup rather than the
model's representation.

Allowing the answer "NO COMMON THEME" is deliberate. A cluster with no readable
theme is a finding about the subspace, and a model pressured into inventing a
label for it would hide exactly what we are trying to measure.
"""

import random

NAMING_SYSTEM = """You label groups of products.

You are shown the titles and categories of products that were grouped together.
Your job is to state the single theme they share.

Rules:
1. Answer with a short label, at most eight words.
2. Do not list the products and do not explain your reasoning.
3. If the products share no recognisable theme, answer exactly: NO COMMON THEME
4. Output the label alone, nothing else."""


def build_naming_prompt(concept, catalogue, n_items=50, seed=42):
    """Render a cluster as an anonymous list of products."""
    items = list(concept.representatives)
    # Representatives are ordered by closeness to the centroid; shuffling stops
    # the model from reading that order as a signal.
    pool = items + [i for i in concept.members if i not in items]
    pool = pool[:n_items]
    random.Random(seed + concept.cluster_id).shuffle(pool)

    lines = ["Products in this group:", ""]
    for item_id in pool:
        title = catalogue.describe(item_id, max_len=110)
        lines.append(f"- {title}")
    lines += ["", "What theme do these products share?"]
    return "\n".join(lines)


def name_concepts(concepts, catalogue, client, n_items=50, verbose=True):
    """Label every cluster in place. Returns the number that got a real label."""
    named = 0
    for index, concept in enumerate(concepts, start=1):
        prompt = build_naming_prompt(concept, catalogue, n_items)
        label = client.ask(prompt, system=NAMING_SYSTEM, max_tokens=60)

        label = label.strip().strip('"').strip()
        if label.upper().startswith('NO COMMON THEME'):
            concept.label = None
        else:
            concept.label = label
            named += 1

        if verbose:
            shown = concept.label or '(no common theme)'
            print(f"  [{index}/{len(concepts)}] {concept.domain}/{concept.channel} "
                  f"cluster {concept.cluster_id} (n={concept.size}): {shown}", flush=True)
    return named
