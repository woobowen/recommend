"""Read public model metadata and small text/tokenizer metadata only; never fetch weights."""
import datetime
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
events = []

def read(url, target, method='GET'):
    request = urllib.request.Request(url, method=method, headers={'User-Agent': 'LC-Rec-preparation/metadata-audit'})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read(2 * 1024 * 1024 + 1) if method == 'GET' else b''
            if len(data) > 2 * 1024 * 1024:
                raise ValueError('metadata response unexpectedly exceeds 2MiB')
            result = {'url': url, 'method': method, 'status': response.status,
                      'bytes': len(data), 'headers': {k:v for k,v in response.headers.items() if k.lower() in ['content-length', 'content-type', 'etag', 'x-linked-size', 'x-linked-etag', 'x-repo-commit']}}
        if method == 'GET':
            (ROOT / target).write_bytes(data)
            result['sha256'] = hashlib.sha256(data).hexdigest()
        events.append(result)
        return data
    except Exception as exc:
        events.append({'url': url, 'method': method, 'error_type': type(exc).__name__, 'error': str(exc)})
        return None

api = read('https://huggingface.co/api/models/huggyllama/llama-7b?blobs=true', 'hf_model_api.json')
if api:
    metadata = json.loads(api)
    revision = metadata['sha']
    (ROOT / 'hf_revision.txt').write_text(revision + '\n')
    for filename in ['config.json', 'generation_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'pytorch_model.bin.index.json', 'model.safetensors.index.json', 'README.md', 'LICENSE']:
        read(f'https://huggingface.co/huggyllama/llama-7b/resolve/{revision}/{filename}', 'hf_' + filename.replace('/', '_'))
    summary = {'model_id': metadata.get('modelId'), 'revision': revision,
               'private': metadata.get('private'), 'gated': metadata.get('gated'),
               'lastModified': metadata.get('lastModified'), 'cardData': metadata.get('cardData'),
               'config': metadata.get('config'), 'safetensors': metadata.get('safetensors'),
               'siblings': metadata.get('siblings')}
    (ROOT / 'hf_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
(ROOT / 'fetch_events.json').write_text(json.dumps({'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'events': events}, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'successes': sum('status' in x for x in events), 'errors': [x for x in events if 'error' in x], 'revision': json.loads(api)['sha'] if api else None}, indent=2))
