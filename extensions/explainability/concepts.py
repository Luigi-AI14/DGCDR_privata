"""Extraction of concept clusters from the disentangled item subspaces.

Contribution 2 asks whether the channel DGCDR calls "domain-shared" encodes
concepts that genuinely cross domains. Answering that needs something nameable
to point at, which is what this module produces: clusters of items that sit
close together inside one channel's subspace.

Clustering rather than principal directions: a PCA axis is faithful to the
geometry but rarely corresponds to anything a person can name, while a cluster
of items can be shown to a reader (or an LLM) and judged. The PCA variant is
kept available as an ablation.
"""

import numpy as np
import torch
import torch.nn.functional as F


class Concept:
    """A cluster of items inside one channel's subspace.

    Attributes:
        cluster_id (int): index within its extraction run.
        channel (str): 'shared', 'specific' or 'base'.
        domain (str): 'source' or 'target'.
        members (list[int]): every item id in the cluster.
        representatives (list[int]): the items closest to the centroid, shown
            to the LLM for naming.
        held_out (list[int]): members kept aside, never shown during naming,
            used to test whether the label actually identifies the cluster.
        size (int): number of members.
        cohesion (float): mean cosine similarity to the centroid. A cluster
            with low cohesion is a bag of unrelated items and any label given
            to it is fiction.
    """

    def __init__(self, cluster_id, channel, domain, members, representatives,
                 held_out, cohesion):
        self.cluster_id = cluster_id
        self.channel = channel
        self.domain = domain
        self.members = members
        self.representatives = representatives
        self.held_out = held_out
        self.size = len(members)
        self.cohesion = cohesion
        self.label = None

    def to_dict(self):
        return {
            'cluster_id': self.cluster_id,
            'channel': self.channel,
            'domain': self.domain,
            'size': self.size,
            'cohesion': self.cohesion,
            'representatives': self.representatives,
            'held_out': self.held_out,
            'label': self.label,
        }


def domain_item_ids(model, domain):
    """Item ids that actually belong to a domain.

    The merged space is ``[PAD] + overlap + target-only + source-only``, and
    both domains carry embeddings for every row. The rows outside a domain were
    zero-filled at init and never trained, so clustering over all of them would
    mostly be clustering zeros.
    """
    n_overlap = model.overlapped_num_items
    if domain == 'target':
        return list(range(1, model.target_num_items))
    return list(range(1, n_overlap)) + list(range(model.target_num_items,
                                                  model.total_num_items))


def _kmeans(X, n_clusters, seed=42, iters=50):
    """k-means on the unit sphere (cosine geometry), no sklearn dependency."""
    generator = torch.Generator().manual_seed(seed)
    centroids = X[torch.randperm(X.size(0), generator=generator)[:n_clusters]].clone()

    assignment = torch.zeros(X.size(0), dtype=torch.long)
    for _ in range(iters):
        similarity = X @ centroids.t()
        new_assignment = similarity.argmax(dim=1)
        if torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for k in range(n_clusters):
            mask = assignment == k
            if mask.any():
                centroids[k] = F.normalize(X[mask].mean(dim=0), dim=0)
    return assignment, centroids


def extract_concepts(model, decomposition, channel, domain, n_clusters=12,
                     n_representatives=15, n_held_out=10, seed=42,
                     min_cluster_size=20):
    """Cluster one channel's item subspace into candidate concepts.

    Returns:
        list[Concept], largest first. Clusters below ``min_cluster_size`` are
        dropped: too small to name reliably and too small to test.
    """
    if channel not in decomposition.item_channels:
        raise ValueError(
            f"channel '{channel}' not available; the checkpoint provides "
            f"{decomposition.item_channel_names}. Item channels beyond 'base' "
            f"require item_disentangle=True.")

    item_ids = domain_item_ids(model, domain)
    embeddings = decomposition.item_channels[channel][
        torch.as_tensor(item_ids, dtype=torch.long)].cpu()

    # Cosine geometry: cluster on directions, not magnitudes, so that popular
    # items (which have larger norms) do not form clusters of their own.
    X = F.normalize(embeddings, dim=1)
    assignment, centroids = _kmeans(X, n_clusters, seed)

    generator = torch.Generator().manual_seed(seed)
    concepts = []
    for k in range(n_clusters):
        mask = assignment == k
        if int(mask.sum()) < min_cluster_size:
            continue

        member_positions = torch.nonzero(mask, as_tuple=True)[0]
        similarity = X[member_positions] @ centroids[k]
        order = torch.argsort(similarity, descending=True)

        ranked = [item_ids[member_positions[i]] for i in order.tolist()]
        representatives = ranked[:n_representatives]

        # Held-out items are drawn from the rest of the cluster, not from its
        # core: a label that only describes the most typical members is not
        # much of a label.
        remainder = ranked[n_representatives:]
        if len(remainder) > n_held_out:
            picks = torch.randperm(len(remainder), generator=generator)[:n_held_out]
            held_out = [remainder[i] for i in picks.tolist()]
        else:
            held_out = remainder

        concepts.append(Concept(
            cluster_id=k,
            channel=channel,
            domain=domain,
            members=ranked,
            representatives=representatives,
            held_out=held_out,
            cohesion=float(similarity.mean()),
        ))

    concepts.sort(key=lambda c: c.size, reverse=True)
    return concepts


def principal_directions(model, decomposition, channel, domain, n_directions=12,
                         n_representatives=15):
    """PCA ablation: concepts as the extremes of the principal axes.

    Reported alongside the clustering to show the findings are not an artefact
    of how the subspace was carved up.
    """
    item_ids = domain_item_ids(model, domain)
    embeddings = decomposition.item_channels[channel][
        torch.as_tensor(item_ids, dtype=torch.long)].cpu()
    X = F.normalize(embeddings, dim=1)
    X = X - X.mean(dim=0, keepdim=True)

    _, _, components = torch.pca_lowrank(X, q=min(n_directions, X.size(1)))

    directions = []
    for axis in range(components.size(1)):
        projection = X @ components[:, axis]
        order = torch.argsort(projection, descending=True)
        directions.append({
            'axis': axis,
            'positive': [item_ids[i] for i in order[:n_representatives].tolist()],
            'negative': [item_ids[i] for i in order[-n_representatives:].tolist()],
            'explained_std': float(projection.std()),
        })
    return directions


def concept_centroids(decomposition, concepts, channel):
    """Unit-norm centroid of each cluster, in its own channel's space.

    Needed to ask which target cluster a source cluster sits closest to, i.e.
    what correspondence the model itself encodes.
    """
    centroids = []
    for concept in concepts:
        members = torch.as_tensor(concept.members, dtype=torch.long)
        X = F.normalize(decomposition.item_channels[channel][members].cpu(), dim=1)
        centroids.append(F.normalize(X.mean(dim=0), dim=0))
    return torch.stack(centroids) if centroids else torch.empty(0)


def geometric_pairing(source_centroids, target_centroids):
    """For each source cluster, the index of the closest target cluster."""
    if source_centroids.numel() == 0 or target_centroids.numel() == 0:
        return []
    similarity = source_centroids @ target_centroids.t()
    return similarity.argmax(dim=1).tolist()


def structure_correlation(model, decomposition, channel_a, channel_b, domain,
                          n_sample=2000, seed=0):
    """Do two channels organise the items in the same way?

    Compares the item-item similarity matrix each channel induces. Unlike
    clustering this is deterministic and has no free parameters, and unlike the
    cosine between channels it sees structure rather than alignment: two
    perfectly orthogonal channels can still rank every pair of items
    identically, which is exactly what DGCDR's item side does.

    Returns (observed, null). The null comes from shuffling one matrix, so a
    value near it means the two channels share no structure at all.
    """
    item_ids = domain_item_ids(model, domain)
    generator = torch.Generator().manual_seed(seed)
    picked = torch.as_tensor(item_ids)[
        torch.randperm(len(item_ids), generator=generator)[:n_sample]]

    def similarity(channel):
        X = F.normalize(decomposition.item_channels[channel][picked].cpu(), dim=1)
        S = (X @ X.t()).numpy()
        return S[~np.eye(S.shape[0], dtype=bool)]

    a, b = similarity(channel_a), similarity(channel_b)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    rng = np.random.default_rng(seed)
    null = float(np.mean([spearman(a, rng.permutation(b)) for _ in range(3)]))
    return spearman(a, b), null


def cluster_category_purity(concepts, catalogue, seed=42, n_null=20):
    """M0 -- are the clusters semantically coherent at all, without any LLM?

    Uses the catalogue's own category hierarchy as ground truth: a coherent
    cluster should concentrate on few categories. The null is a random cluster
    of the same size drawn from the same domain, which controls for how skewed
    the category distribution is to begin with.

    This is the first gate of Contribution 2. If clusters do not beat the null,
    the subspaces carry no readable semantic structure and naming them would be
    an exercise in pattern-matching noise.
    """
    rng = np.random.default_rng(seed)
    pool = sorted({item for concept in concepts for item in concept.members})

    def purity(item_ids):
        categories = [catalogue.category(i) for i in item_ids]
        categories = [c for c in categories if c]
        if not categories:
            return None
        counts = {}
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
        return max(counts.values()) / len(categories)

    results = []
    for concept in concepts:
        observed = purity(concept.members)
        if observed is None:
            continue
        nulls = []
        for _ in range(n_null):
            sample = rng.choice(pool, size=min(concept.size, len(pool)), replace=False)
            value = purity(sample.tolist())
            if value is not None:
                nulls.append(value)
        null_mean = float(np.mean(nulls)) if nulls else 0.0
        results.append({
            'cluster_id': concept.cluster_id,
            'size': concept.size,
            'cohesion': concept.cohesion,
            'purity': observed,
            'null_purity': null_mean,
            'lift': observed - null_mean,
        })
    return results
