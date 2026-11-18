"""Single-GPU production token intervention using the unchanged Shuffle protocol.

The current formal_evaluate loop owns model loading and four-condition evidence.
The retired in-memory source-patch acceptance harness remains in Git history.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shuffle'))
from core import transform


def transform_inputs(ids, mask, groups, all_sid_ids, n, row_idx):
    """Production condition intervention; Normal bypasses token reconstruction."""
    if n is None:
        return list(ids), {'n': None, 'seed':42, 'protocol':'GRID-LETTER-SHUFFLE-V1',
            'row_idx':row_idx, 'evidence':{'groups':[], 'path':'native collator input; no intervention'}}
    changed,evidence = transform(ids,mask,groups,all_sid_ids,n,row_idx)
    return changed, {'n':n, 'seed':42, 'protocol':'GRID-LETTER-SHUFFLE-V1',
                     'row_idx':row_idx, 'original_input_token_ids':list(ids),
                     'attention_mask':list(mask), 'evidence':evidence}
