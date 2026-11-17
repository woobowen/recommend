"""CPU-only prompt sizing using the actual local tokenizers."""
from pathlib import Path
from transformers import AutoTokenizer

from common import roots, read_jsonl, sha, write_json


def summary(lengths):
    ordered = sorted(lengths)
    return {'total': sum(lengths), 'min': ordered[0], 'median': ordered[len(ordered)//2],
            'p99': ordered[int(len(ordered)*.99)], 'max': ordered[-1],
            'above_128_tokens': sum(n > 128 for n in lengths)}


def main():
    cfg, root, prep, out = roots()
    llama = AutoTokenizer.from_pretrained(root/cfg['model'], local_files_only=True)
    qwen_path = root/'models/Qwen2.5-7B-Instruct'
    qwen = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    report = {'note': 'Llama lengths exclude chat wrappers and repeated taxonomy; Qwen lengths match upstream add_special_tokens=True.',
              'datasets': {}}
    for dataset in cfg['datasets']:
        path = prep/dataset/'classifier_items.jsonl'
        rows = read_jsonl(path)
        llama_lengths = [len(llama.encode(r['text'], add_special_tokens=False)) for r in rows]
        qwen_lengths = [len(qwen.encode(r['text'], add_special_tokens=True)) for r in rows]
        report['datasets'][dataset] = {'items': len(rows), 'classifier_items_sha256': sha(path),
            'llama_text_tokens': summary(llama_lengths), 'qwen_upstream_tokens': summary(qwen_lengths),
            'qwen_truncated_item_ids': [r['item_id'] for r, length in zip(rows, qwen_lengths) if length > 128],
            'blank_texts': sum(not r['text'].strip() for r in rows)}
    write_json(prep/'text_token_audit.json', report)
    print({d: {k: v for k, v in data.items() if k != 'qwen_truncated_item_ids'}
           for d, data in report['datasets'].items()}, flush=True)


if __name__ == '__main__':
    main()
