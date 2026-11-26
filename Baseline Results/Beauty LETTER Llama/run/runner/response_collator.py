"""Keep author batches unchanged except rows whose entire answer was truncated."""
import torch


class ResponseSafeCollator:
    def __init__(self,author_collator,tokenizer):
        self.author=author_collator;self.tokenizer=tokenizer;self.repaired_rows=0

    def __call__(self,rows):
        batch=self.author(rows)
        empty=(batch['labels']!=-100).sum(dim=1)==0
        for i in empty.nonzero(as_tuple=False).flatten().tolist():
            prompt=self.tokenizer.encode(rows[i]['input_ids'],truncation=False)
            full=self.tokenizer.encode(rows[i]['labels']+self.tokenizer.eos_token,truncation=False)
            response=full[len(prompt):]
            if not response:raise ValueError('Missing response even before truncation')
            maximum=self.tokenizer.model_max_length
            reserve=min(len(response),128)
            budget=maximum-reserve
            prefix=full[:len(prompt)]
            # Preserve the instruction opening and the final Response marker.
            kept=prefix[:budget-16]+prefix[-16:] if len(prefix)>budget else prefix
            answer=response[:maximum-len(kept)]
            ids=kept+answer
            labels=[-100]*len(kept)+[t if t!=self.tokenizer.pad_token_id else -100 for t in answer]
            assert len(ids)<=batch['input_ids'].shape[1] and any(t!=-100 for t in labels)
            batch['input_ids'][i].fill_(self.tokenizer.pad_token_id)
            batch['attention_mask'][i].zero_();batch['labels'][i].fill_(-100)
            batch['input_ids'][i,:len(ids)]=torch.tensor(ids)
            batch['attention_mask'][i,:len(ids)]=1
            batch['labels'][i,:len(ids)]=torch.tensor(labels)
            self.repaired_rows+=1
        return batch
