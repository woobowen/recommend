"""Output shape constraints; class names and the taxonomy remain model decisions."""


def taxonomy_schema(maximum):
    fields = {name: {'type': 'string', 'minLength': 1, 'maxLength': limit}
              for name, limit in [('coarse_name', 80), ('name', 80), ('definition', 160),
                                  ('include', 120), ('exclude', 120)]}
    entry = {'type': 'object', 'properties': fields, 'required': list(fields),
             'additionalProperties': False}
    return {'type': 'object', 'properties': {'categories': {
        'type': 'array', 'minItems': 1, 'maxItems': max(1, maximum), 'items': entry}},
        'required': ['categories'], 'additionalProperties': False}


def classification_schema(item_ids, category_ids):
    entry = {'type': 'object', 'properties': {
        'item_id': {'type': 'integer', 'enum': sorted(item_ids)},
        'category_id': {'type': 'string', 'enum': sorted(category_ids)},
        'needs_review': {'type': 'boolean'}},
        'required': ['item_id', 'category_id', 'needs_review'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'items': {
        'type': 'array', 'minItems': len(item_ids), 'maxItems': len(item_ids), 'items': entry}},
        'required': ['items'], 'additionalProperties': False}
