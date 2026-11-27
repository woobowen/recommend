"""Identify history from rendered seqrec instructions and retained SID tokens."""

import re
from functools import lru_cache

import torch


PROMPT_START = ("Below is an instruction that describes a task. Write a response that "
                "appropriately completes the request.\n\n### Instruction:\n")
RESPONSE = "\n\n### Response:"
CODE = re.compile(r"<[^<>]+>")


def history_patterns(templates):
    patterns = []
    for template in templates:
        left, right = template["instruction"].split("{inters}")
        if template["response"] != "{item}":
            raise ValueError("unexpected historical seqrec response template")
        patterns.append(re.compile(re.escape(PROMPT_START + left) + r"(?P<history>.+)"
                                   + re.escape(right + RESPONSE)))
    return patterns


class HistoryCollator:
    """Original LETTER collation, augmented with token-aligned history bits."""

    def __init__(self, collator, tokenizer, sid_tokens):
        from prompt import all_prompt

        self.collator = collator
        self.tokenizer = tokenizer
        self.patterns = history_patterns(all_prompt["seqrec"])
        self.sid = {text: tokenizer.convert_tokens_to_ids(text) for text in sid_tokens}
        if len(set(self.sid.values())) != len(self.sid):
            raise ValueError("SID token ID collision")
        if any(tokenizer.convert_ids_to_tokens(i) != text for text, i in self.sid.items()):
            raise ValueError("SID tokens must already be registered with the native tokenizer")
        self.sid_ids = set(self.sid.values())
        self.history_ids = lru_cache(maxsize=4096)(self._history_ids)

    def _codes(self, text):
        pieces = CODE.findall(text)
        if "".join(pieces) != text or len(pieces) != 4:
            raise ValueError("expected a four-token LETTER SID")
        return [self.sid[x] for x in pieces]

    def _history_ids(self, prompt):
        spans = [match.group("history") for p in self.patterns if (match := p.fullmatch(prompt))]
        if len(spans) != 1:
            raise ValueError("expected exactly one recognized history span in a seqrec instruction")
        return [token_id for item in spans[0].split(", ") for token_id in self._codes(item)]

    def __call__(self, examples):
        result = self.collator(examples)
        history = torch.zeros_like(result["attention_mask"], dtype=torch.bool)
        for i, row in enumerate(examples):
            prompt, full = row["input_ids"], row["labels"]
            if not full.startswith(prompt):
                raise ValueError("response does not follow the exact instruction")
            expected_history = self.history_ids(prompt)
            expected_codes = expected_history + self._codes(full[len(prompt):])
            observed = [(j, code) for j, (code, valid) in enumerate(zip(
                result["input_ids"][i].tolist(), result["attention_mask"][i].tolist()))
                        if valid and code in self.sid_ids]
            n = len(observed)
            offset = len(expected_codes) - n if self.tokenizer.truncation_side == "left" else 0
            if (n > len(expected_codes)
                    or [code for _, code in observed] != expected_codes[offset:offset + n]):
                raise ValueError("retained SID IDs disagree with the rendered history and canonical answer")
            positions = [j for ordinal, (j, _) in enumerate(observed)
                         if offset + ordinal < len(expected_history)]
            history[i, positions] = True
        result["history_sid_mask"] = history
        return result
