"""Check epoch / accumulation arithmetic, not the runtime training loop."""
import json


def schedule(microbatches_per_epoch, epochs, gas):
    if microbatches_per_epoch <= 0 or epochs <= 0 or gas <= 0:
        raise ValueError('positive integer parameters required')
    return [{'epoch': epoch, 'consumed': epoch * microbatches_per_epoch,
             'updates': epoch * microbatches_per_epoch // gas,
             'pending_microbatches': epoch * microbatches_per_epoch % gas}
            for epoch in range(1, epochs+1)]


if __name__ == '__main__':
    rows = schedule(8214, 4, 4)
    assert [row['updates'] for row in rows] == [2053, 4107, 6160, 8214]
    assert [row['pending_microbatches'] for row in rows] == [2, 0, 2, 0]
    assert rows[-1]['consumed'] == 32856
    print(json.dumps({'status':'PASS', 'schedule':rows,
                      'scope':'arithmetic only; no optimizer, checkpoint or data run'}, indent=2))
