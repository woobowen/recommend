"""Serialize the historical two-rank Beauty sample stream on one GPU.

Each pair is the original rank-0 eight-row batch followed by its rank-1 batch.
The native Accelerate 0.21 BatchSamplerShard still determines tail completion.
Eight serial microbatches therefore contain the same 64 indexed examples as
four historical two-rank microsteps. This is sample/weight preservation, not a
claim that single- and dual-GPU floating-point trajectories are identical.
"""
from collections import Counter
import hashlib
import itertools
import json
import math


def execution_schedule(n=131413, *, preflight=False):
    if type(n) is not int or n <= 0:
        raise ValueError('positive training row count required')
    padded_rows=math.ceil(n/16)*16
    per_epoch=20 if preflight else padded_rows//8
    boundaries=[((epoch+1)*per_epoch)//8 for epoch in range(4)]
    if 4*per_epoch % 8:
        raise ValueError('four epochs must end on an optimizer boundary')
    return {'world_size':1,'logical_historical_ranks':2,'micro_batch_size':8,
            'gradient_accumulation_steps':8,'global_batch':64,'epochs':4,
            'author_rows':n,'full_epoch_serial_rows':padded_rows,
            'tail_repeat_rows':padded_rows-n,'full_epoch_microbatches':padded_rows//8,
            'per_epoch_microbatches':per_epoch,'epoch_steps':boundaries,
            'max_steps':boundaries[-1],'preflight_only':bool(preflight),
            'short_epoch_global_rows':160 if preflight else None,
            'loss_weighting':'Each unchanged eight-example native microbatch loss contributes 1/8; this equals its historical 1/4 accumulation times1/2 rank-average weight. Native response-token reduction inside each microbatch remains unchanged; no global-token-mean or floating-trajectory equivalence is claimed.',
            'numerical_equivalence_to_two_gpus_claimed':False}


def _native_shard_iterators(n, generator_state):
    import torch
    from torch.utils.data import RandomSampler, BatchSampler
    from accelerate.data_loader import BatchSamplerShard
    generators=[];iterators=[]
    for rank in (0,1):
        gen=torch.Generator();gen.set_state(generator_state.cpu())
        native=RandomSampler(range(n),replacement=False,generator=gen)
        batches=BatchSampler(native,batch_size=8,drop_last=False)
        shards=BatchSamplerShard(batches,num_processes=2,process_index=rank,
                                split_batches=False,even_batches=True)
        generators.append(gen);iterators.append(iter(shards))
    return generators,iterators


def _serial_batches(n, state, update_generator=None):
    generators,shards=_native_shard_iterators(n,state)
    for _ in range(math.ceil(n/16)):
        for logical_rank in (0,1):
            batch=next(shards[logical_rank])
            if len(batch)!=8:
                raise AssertionError('native historical shard must contain eight rows')
            if update_generator is not None:
                update_generator.set_state(generators[0].get_state())
            yield batch
    for iterator in shards:
        try:next(iterator)
        except StopIteration:pass
        else:raise AssertionError('unexpected extra historical shard batch')
    if update_generator is not None:
        update_generator.set_state(generators[0].get_state())


def reconstruct_serial_epoch(n, generator_state, microbatches=None):
    """CPU reconstruction on cloned RNG; never mutates the live loader state."""
    batches=_serial_batches(n,generator_state)
    if microbatches is not None:
        if type(microbatches) is not int or microbatches < 0:
            raise ValueError('nonnegative microbatch limit required')
        batches=itertools.islice(batches,microbatches)
    return [row for batch in batches for row in batch]


def build_serial_sampler(data_source, generator=None):
    """Keep native HF/Accelerate RandomSampler generator preparation intact."""
    import torch
    from torch.utils.data import RandomSampler
    if generator is None:
        # Accelerate0.21 creates this default generator for the old two-rank
        # sampler. Its single-process path omits it, so bind it explicitly.
        generator=torch.Generator()
    class SerialHistoricalSampler(RandomSampler):
        def __init__(self):
            super().__init__(data_source,replacement=False,generator=generator)
        def __len__(self):
            return math.ceil(len(self.data_source)/16)*16
        def __iter__(self):
            if self.generator is None:
                raise RuntimeError('native Accelerate synchronized sampler generator required')
            for batch in _serial_batches(len(self.data_source),self.generator.get_state(),self.generator):
                yield from batch
    return SerialHistoricalSampler()


def audit_serial_epoch(rows,n,*,preflight=False):
    expected=160 if preflight else math.ceil(n/16)*16
    if len(rows)!=expected or any(type(x) is not int or not 0<=x<n for x in rows):
        raise ValueError('serial global stream row count/range')
    counts=Counter(rows)
    repeats=sum(v-1 for v in counts.values())
    if preflight:
        if repeats or len(counts)!=160:
            raise ValueError('short serial epoch must cover160 distinct native sampled rows')
    elif len(counts)!=n or repeats!=expected-n:
        raise ValueError('serial global stream missing rows or wrong tail repeat policy')
    batches=[rows[i:i+8] for i in range(0,len(rows),8)]
    updates=[rows[i:i+64] for i in range(0,len(rows),64)]
    return {'scope':'first20 serial microbatches' if preflight else 'full serialized historical two-rank epoch',
            'rows':len(rows),'distinct_rows':len(counts),'repeat_rows':repeats,
            'duplicates':{str(k):v-1 for k,v in sorted(counts.items()) if v>1},
            'historical_logical_rank_rows':[list(itertools.chain.from_iterable(batches[i::2])) for i in (0,1)],
            'global_row_order_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
            'global64_groups_sha256':hashlib.sha256(json.dumps(updates,separators=(',',':')).encode()).hexdigest(),
            'tail_rows':rows[-16:],'loss_weighting':execution_schedule(n,preflight=preflight)['loss_weighting'],
            'numerical_equivalence_claimed':False}
