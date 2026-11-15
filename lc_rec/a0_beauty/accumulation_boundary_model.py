#!/usr/bin/env python3
"""Pure-CPU state-machine check; not a Trainer or DeepSpeed acceptance test."""
import copy
import json
import pickle
import random

GAS, EPOCHS, BATCHES = 4, 4, 10


def run(state, stop_after=None):
    random.setstate(state['rng'])
    while state['consumed'] < EPOCHS * BATCHES:
        position = state['consumed']
        epoch, batch = divmod(position, BATCHES)
        value = random.randrange(1, 1000000)
        state['order'].append((epoch, batch, value))
        state['gradient'] += value
        state['consumed'] += 1
        if state['consumed'] % GAS == 0:
            state['parameter'] += state['gradient']
            state['gradient'] = 0
            state['optimizer_steps'] += 1
            state['scheduler_steps'] += 1
        if state['consumed'] % BATCHES == 0:
            state['validations'].append((epoch + 1, state['consumed'], state['parameter']))
        state['rng'] = random.getstate()
        if state['consumed'] == stop_after:
            return state
    return state


def main():
    random.seed(42)
    initial = {'consumed': 0, 'gradient': 0, 'parameter': 0, 'optimizer_steps': 0,
               'scheduler_steps': 0, 'order': [], 'validations': [], 'rng': random.getstate()}
    reference = run(copy.deepcopy(initial))
    assert reference['consumed'] == 40 and reference['optimizer_steps'] == 10
    assert reference['scheduler_steps'] == 10 and reference['gradient'] == 0
    assert [v[1] for v in reference['validations']] == [10, 20, 30, 40]
    for boundary in (10, 20, 30):
        partial = run(copy.deepcopy(initial), stop_after=boundary)
        assert (partial['gradient'] != 0) == (boundary % GAS != 0)
        restored = pickle.loads(pickle.dumps(partial))
        assert run(restored) == reference
    # Also model a recovery point taken just before each partial epoch end:
    # replay the two consumed microbatches and the validation on resume. This
    # is a possible design to investigate, not proof that DS replay works.
    for full_boundary in (10, 30):
        safe = full_boundary - (full_boundary % GAS)
        checkpoint = run(copy.deepcopy(initial), stop_after=safe)
        assert checkpoint['gradient'] == 0
        restored = pickle.loads(pickle.dumps(checkpoint))
        assert run(restored) == reference
    print(json.dumps({'status': 'PASS', 'microbatches_per_epoch': BATCHES,
                      'epochs': EPOCHS, 'gas': GAS, 'optimizer_steps': 10,
                      'checkpoint_boundaries': [10, 20, 30], 'safe_boundary_replays': [8, 28],
                      'matches': ['parameters', 'pending_gradient', 'optimizer_count',
                                  'scheduler_count', 'rng', 'sample_order', 'validation_positions'],
                      'scope': 'CPU synthetic state machine only; no real model, Trainer, or DeepSpeed'}, indent=2))


if __name__ == '__main__':
    main()
