"""Optional item-side metadata for turning internal ids into readable text.

The attribution itself is purely numeric and works without metadata; this
module only supplies the surface strings the verbalisation stage needs.
Everything degrades gracefully to bare ids when a catalogue is missing, so an
attribution run never fails because of absent metadata.
"""

import json
import os


class ItemCatalogue:
    """Maps internal target-domain item ids to human-readable descriptions."""

    def __init__(self, dataset, records=None):
        self.dataset = dataset
        self.records = records or {}
        self._id2token = self._build_id_map()

    def _build_id_map(self):
        """Map merged item ids to dataset tokens.

        RecBole-CDR lays the merged item space out as
        ``[PAD] + overlap + target-only + source-only``, but each domain keeps
        its own compacted token array: the target's index happens to equal the
        merged id, while the source's array skips the target-only block and is
        therefore offset by it. Indexing the source array with a merged id --
        the obvious thing to do, and what ``DGCDR.__init__`` does when loading
        text embeddings -- silently yields the wrong item.
        """
        target = self.dataset.target_domain_dataset
        source = self.dataset.source_domain_dataset
        target_tokens = target.field2id_token[target.iid_field]
        source_tokens = source.field2id_token[source.iid_field]

        n_overlap = self.dataset.num_overlap_item
        n_target_only = len(target_tokens) - n_overlap

        id2token = {i: token for i, token in enumerate(target_tokens)}
        for local_id, token in enumerate(source_tokens):
            merged_id = local_id if local_id < n_overlap else local_id + n_target_only
            id2token.setdefault(merged_id, token)
        return id2token

    def token(self, item_id):
        """Internal item id -> dataset token (e.g. the Amazon parent_asin)."""
        return self._id2token.get(int(item_id))

    def record(self, item_id):
        token = self.token(item_id)
        return self.records.get(token) if token else None

    def describe(self, item_id, max_len=160):
        """Short readable label for an item; falls back to the raw id."""
        record = self.record(item_id)
        if not record:
            token = self.token(item_id)
            return f"item#{item_id}" if token is None else f"item#{item_id} ({token})"

        title = (record.get('title') or '').strip()
        category = self.category(item_id)
        label = title if title else f"item#{item_id}"
        if len(label) > max_len:
            label = label[:max_len - 1].rstrip() + '…'
        return f"{label} [{category}]" if category else label

    def category(self, item_id):
        """The readable category of an item.

        Amazon's ``categories`` is a hierarchy whose first level repeats the
        domain ("Musical Instruments" for every instrument), so it says nothing
        within a domain. The second level ("Guitars", "Drums & Percussion") is
        the one that actually separates items, and is what a profile needs.
        """
        record = self.record(item_id)
        if not record:
            return ''
        hierarchy = record.get('categories') or []
        if len(hierarchy) > 1:
            return hierarchy[1]
        if hierarchy:
            return hierarchy[0]
        return (record.get('category') or '').strip()

    def profile(self, item_ids, top_n=4, examples=3, max_len=60):
        """Summarise a set of items into its main categories and a few titles.

        Meant to make a user's taste legible at a glance instead of listing
        dozens of product names.
        """
        counts = {}
        titles = []
        for item_id in item_ids:
            category = self.category(item_id)
            if category:
                counts[category] = counts.get(category, 0) + 1
            record = self.record(item_id)
            title = (record or {}).get('title', '').strip()
            if title and len(titles) < examples:
                titles.append(title if len(title) <= max_len
                              else title[:max_len - 1].rstrip() + '…')
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return ranked, titles

    def has_metadata(self):
        return bool(self.records)


def _first_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _first_string(value[0])
    return ''


def load_jsonl_catalogue(dataset, path, id_field='parent_asin',
                         title_field='title', category_field='main_category'):
    """Load an item catalogue from a JSONL metadata dump.

    Matches the format of ``item_metadata/meta_*.jsonl`` but the field names are
    configurable so other datasets (Douban, Yelp) can reuse it.  A missing file
    is not an error: it yields an empty catalogue.
    """
    records = {}
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = item.get(id_field)
                if key is None:
                    continue
                categories = item.get('categories')
                records[str(key)] = {
                    'title': _first_string(item.get(title_field)),
                    'category': _first_string(item.get(category_field)),
                    'categories': categories if isinstance(categories, list) else [],
                }
    return ItemCatalogue(dataset, records)


def load_catalogue(dataset, paths, **kwargs):
    """Load and merge several JSONL metadata files into one catalogue."""
    merged = {}
    for path in paths or []:
        catalogue = load_jsonl_catalogue(dataset, path, **kwargs)
        merged.update(catalogue.records)
    return ItemCatalogue(dataset, merged)
