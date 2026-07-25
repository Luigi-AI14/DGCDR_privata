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
        self._token_cache = {}

    def token(self, item_id):
        """Internal item id -> dataset token (e.g. the Amazon parent_asin)."""
        if item_id in self._token_cache:
            return self._token_cache[item_id]

        token = None
        target = self.dataset.target_domain_dataset
        try:
            token = target.id2token(target.iid_field, item_id)
        except (ValueError, IndexError, KeyError):
            source = self.dataset.source_domain_dataset
            try:
                token = source.id2token(source.iid_field, item_id)
            except (ValueError, IndexError, KeyError):
                token = None

        self._token_cache[item_id] = token
        return token

    def describe(self, item_id, max_len=160):
        """Short readable label for an item; falls back to the raw id."""
        token = self.token(item_id)
        record = self.records.get(token) if token else None
        if not record:
            return f"item#{item_id}" if token is None else f"item#{item_id} ({token})"

        title = (record.get('title') or '').strip()
        category = (record.get('category') or '').strip()
        label = title if title else f"item#{item_id}"
        if len(label) > max_len:
            label = label[:max_len - 1].rstrip() + '…'
        return f"{label} [{category}]" if category else label

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
                records[str(key)] = {
                    'title': _first_string(item.get(title_field)),
                    'category': _first_string(item.get(category_field)),
                }
    return ItemCatalogue(dataset, records)


def load_catalogue(dataset, paths, **kwargs):
    """Load and merge several JSONL metadata files into one catalogue."""
    merged = {}
    for path in paths or []:
        catalogue = load_jsonl_catalogue(dataset, path, **kwargs)
        merged.update(catalogue.records)
    return ItemCatalogue(dataset, merged)
