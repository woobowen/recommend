"""Full-user, resumable three-condition inference on one selected adapter."""
from pathlib import Path
import argparse
import collections
import json
import math
import re
import time
from common import MODEL,AUTHOR,event,write,sha,runtime
from shuffle_grid_protocol import PROTOCOL_VERSION,transform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['Beauty','Instruments'],required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    try:
        torch=runtime()
        from transformers import LlamaForCausalLM,LlamaTokenizer
        from peft import PeftModel
        from data import SeqRecDataset
        # Import the exact author evaluator under a unique module name.
        import importlib.util
        spec=importlib.util.spec_from_file_location('author_letter_metrics',AUTHOR/'LETTER-LC-Rec/evaluate.py')
        metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)
        trained=(json.loads((args.training/'capacity_result.json').read_text())['trainer_integration']
                 if args.preflight else json.loads((args.training/'training_result.json').read_text()))
        assert trained['complete'] and (args.preflight or trained['full_training'])
        adapter=Path(trained['best_checkpoint'])
        checkpoint_sha=sha(adapter/'adapter_model.safetensors')
        assert checkpoint_sha==trained['best_adapter_sha256']
        tokenizer=LlamaTokenizer.from_pretrained(adapter,local_files_only=True,padding_side='left')
        tokenizer.pad_token_id=0
        cfg=argparse.Namespace(dataset=args.dataset,data_path=str(AUTHOR/'data'),max_his_len=20,
            his_sep=', ',index_file='.index.json',add_prefix=False,only_train_response=True,
            sample_valid=False,valid_prompt_id=0,valid_prompt_sample_num=2)
        test=SeqRecDataset(cfg,mode='test',prompt_sample_num=1,sample_num=-1);test.prompt_id=0
        users=list(test.inters.keys());histories=list(test.inters.values())
        assert len(users)==(22363 if args.dataset=='Beauty' else 24772)
        if args.preflight:users=users[:4];histories=histories[:4]
        sid_ids={tokenizer.convert_tokens_to_ids(t) for t in test.get_new_tokens()}
        allowed=test.get_prefix_allowed_tokens_fn(tokenizer);all_items=test.get_all_items()
        counts=collections.Counter(tuple(code) for code in test.indices.values())
        ambiguous=sum(counts[tuple(test.indices[str(h[-1])])]>1 for h in histories)
        model=LlamaForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map='balanced',
            max_memory={0:'8GiB',1:'8GiB'},attn_implementation='sdpa',local_files_only=True)
        assert all(str(v) not in ['cpu','disk'] for v in model.hf_device_map.values())
        model.resize_token_embeddings(len(tokenizer));model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        model.eval();device=model.get_input_embeddings().weight.device
        event(args.out,'evaluation_started',dataset=args.dataset,total_users=len(users),
            adapter=str(adapter),checkpoint_sha256=checkpoint_sha,beam=20,
            precision='unquantized_fp16',shuffle_protocol=PROTOCOL_VERSION,shuffle_seed=2026,
            ambiguous_sid_test_targets=ambiguous)
        main_keys=['hit@5','hit@10','ndcg@5','ndcg@10'];all_modes={}
        for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
            folder=args.out/mode;folder.mkdir(exist_ok=True)
            manifest={'dataset':args.dataset,'mode':mode,'checkpoint_sha256':checkpoint_sha,
                'beam':20,'test_users':len(users),'precision':'unquantized_fp16',
                'preflight':args.preflight,
                'protocol':PROTOCOL_VERSION,'seed':2026,'prompt_id':0,
                'indices_sha256':sha(AUTHOR/'data'/args.dataset/f'{args.dataset}.index.json'),
                'interactions_sha256':sha(AUTHOR/'data'/args.dataset/f'{args.dataset}.inter.json'),
                'evaluator_source_sha256':sha(Path(__file__))}
            if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())==manifest
            else:write(folder/'manifest.json',manifest)
            sums={k:0.0 for k in main_keys};done=changed=groups=valid_predictions=0
            parts=sorted(folder.glob('part-*.json'))
            for file in parts:
                for row in json.loads(file.read_text()):
                    assert row['row']==done and row['user_id']==users[done]
                    for key in main_keys:sums[key]+=row['metrics'][key]
                    changed+=row['changed_items'];groups+=row['history_items'];valid_predictions+=row['valid_candidates'];done+=1
            started_done=done;start=time.monotonic();buffer=[];part_number=len(parts)
            event(args.out,'mode_started',mode=mode,users_done=done,total_users=len(users))
            for row_idx in range(done,len(users)):
                sample=test[row_idx]
                original=tokenizer.encode(sample['input_ids'])
                expected=[[tokenizer.convert_tokens_to_ids(t) for t in test.indices[str(i)]]
                          for i in histories[row_idx][:-1][-20:]]
                ids,audit=transform(original,[1]*len(original),expected,sid_ids,n,int(users[row_idx]),args.dataset.lower(),2026)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    generation=model.generate(input_ids=torch.tensor([ids],device=device),
                        attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device=device),
                        num_beams=20,num_return_sequences=20,max_new_tokens=5,do_sample=False,
                        prefix_allowed_tokens_fn=allowed,output_scores=True,return_dict_in_generate=True,
                        early_stopping=True,pad_token_id=0,eos_token_id=tokenizer.eos_token_id,use_cache=True)
                scores=generation.sequences_scores.float().cpu().tolist()
                assert len(scores)==20 and all(math.isfinite(s) for s in scores)
                predictions=tokenizer.batch_decode(generation.sequences,skip_special_tokens=True)
                hit_rows=metrics_module.get_topk_results(predictions,scores,[sample['labels']],20,all_items=None)
                metrics=metrics_module.get_metrics_results(hit_rows,main_keys)
                assert all(0<=float(value)<=1.000001 for value in metrics.values())
                candidates=[tokenizer.decode(s[len(ids):],skip_special_tokens=True).strip().replace(' ','')
                            for s in generation.sequences]
                assert all(re.fullmatch(r'<a_\d+><b_\d+><c_\d+><d_\d+>',s) for s in candidates)
                changed_items=sum(g['nonidentity'] for g in audit['groups'])
                valid_count=sum(s in all_items for s in candidates)
                row={'row':row_idx,'user_id':users[row_idx],'target':sample['labels'],'metrics':metrics,
                    'candidates':candidates,'scores':scores,'history_items':len(expected),
                    'changed_items':changed_items,'valid_candidates':valid_count,
                    'permutations':[g['permutation'] for g in audit['groups']],
                    'original_input_sha256':audit['original_sha256'],'input_sha256':audit['shuffled_sha256']}
                buffer.append(row);done+=1;changed+=changed_items;groups+=len(expected);valid_predictions+=valid_count
                for key in main_keys:sums[key]+=metrics[key]
                if len(buffer)==100 or done==len(users):
                    write(folder/f'part-{part_number:05d}.json',buffer);part_number+=1;buffer=[]
                if done%20==0 or done==len(users):
                    event(args.out,'test_progress',mode=mode,users_done=done,total_users=len(users),
                        metrics={key:sums[key]/done for key in main_keys},
                        seconds_per_user=(time.monotonic()-start)/max(1,done-started_done),
                        durable_users=done-len(buffer))
                del generation
            assert done==len(users)
            outcome={**manifest,'complete':True,'full_test_set':not args.preflight,'evaluated_users':done,
                'metrics':{'recall@5':sums['hit@5']/done,'recall@10':sums['hit@10']/done,
                           'ndcg@5':sums['ndcg@5']/done,'ndcg@10':sums['ndcg@10']/done},
                'changed_history_items':changed,'total_history_items':groups,
                'valid_candidate_fraction':valid_predictions/(20*done),'ambiguous_sid_test_targets':ambiguous,
                'metric_identity':'full SID match; official collisions retained'}
            write(args.out/f'test_{mode}.json',outcome);all_modes[mode]=outcome
            event(args.out,'mode_complete',mode=mode,evaluated_users=done,metrics=outcome['metrics'])
        final={'complete':True,'dataset':args.dataset,'checkpoint_sha256':checkpoint_sha,
               'test_users':len(users),'results':all_modes}
        write(args.out/'evaluation_result.json',final)
        rows=['| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |', '|---|---:|---:|---:|---:|']
        for mode,r in all_modes.items():rows.append('| '+mode+' | '+' | '.join(f'{r["metrics"][k]:.6f}' for k in ['recall@5','recall@10','ndcg@5','ndcg@10'])+' |')
        (args.out/'metrics.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
        event(args.out,'evaluation_complete',dataset=args.dataset,total_users=len(users))
    except Exception as exc:
        import traceback
        event(args.out,'failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
