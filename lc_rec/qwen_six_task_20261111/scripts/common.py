"""LC-Rec six-task semantics and explicit Qwen3 Base interface adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


TASKS = ["seqrec", "item2index", "index2item", "fusionseqrec", "itemsearch", "preferenceobtain"]
EXPECTED_ROWS = [131837, 9922, 9922, 20000, 24772, 24772]
MODELS = {
    "q06": ("Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "qwen3"),
    "q17": ("Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", "qwen3"),
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write(path, value):
    def convert(container):
        if isinstance(container, (set, frozenset)):
            return sorted(container)
        raise TypeError("Unsupported evidence JSON type: " + type(container).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n")
    temporary.replace(path)


def tokenizer_for(path, tokens):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        padding_side="right", model_max_length=2048,
    )
    assert len(tokens) == 930 and len(set(tokens)) == 930, "Expected original LC-Rec 930 SID tokens"
    assert list(tokens) == sorted(tokens), "Preserve official BaseDataset.get_new_tokens ordering"
    original_size = len(tokenizer)
    added = tokenizer.add_tokens(tokens)
    assert added == 930, (added, "Do not reuse LETTER's SID tokenizer")
    encoded = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
    assert all(len(token_ids) == 1 for token_ids in encoded)
    token_ids = [token_ids[0] for token_ids in encoded]
    assert len(set(token_ids)) == 930
    assert token_ids == list(range(original_size, original_size + 930))
    assert not set(token_ids).intersection(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.eos_token_id is not None and tokenizer.pad_token_id is not None
    return tokenizer, {
        "class": type(tokenizer).__name__,
        "original_tokenizer_size": original_size,
        "expanded_tokenizer_size": len(tokenizer),
        "added_sid_tokens": added,
        "sid_tokens": tokens,
        "sid_ids": token_ids,
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "model_max_length": tokenizer.model_max_length,
        "response_separator_ids": tokenizer.encode("Response:", add_special_tokens=False),
        "encodes_automatic_bos": tokenizer.encode("Response:") != tokenizer.encode("Response:", add_special_tokens=False),
        "padding_mask_adapter": "Use attention_mask, preserving real EOS when Qwen PAD equals EOS",
        "chat_template_applied": False,
    }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        tokenizer = self.tokenizer
        assert tokenizer.padding_side == "right"
        assert all(row["labels"].startswith(row["input_ids"]) for row in rows)
        inputs = tokenizer(
            [row["labels"] + tokenizer.eos_token for row in rows],
            padding="longest", truncation=True, max_length=tokenizer.model_max_length,
            return_tensors="pt", return_attention_mask=True,
        )
        prefixes = tokenizer(
            [row["input_ids"] for row in rows], padding=False,
            truncation=True, max_length=tokenizer.model_max_length,
        )["input_ids"]
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for position, prefix in enumerate(prefixes):
            labels[position, :len(prefix)] = -100
        inputs["labels"] = labels
        return inputs


def loss(logits, labels):
    import torch.nn.functional as functional

    return functional.cross_entropy(
        logits.float()[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction="mean",
    )


def model_class_and_config(path, family="qwen3"):
    from transformers import AutoConfig, Qwen3ForCausalLM

    assert family == "qwen3", "Only the two authorized Qwen3 Base models are supported"
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    assert config.model_type == "qwen3"
    assert config.tie_word_embeddings is True
    return Qwen3ForCausalLM, config


def load_model(path, family="qwen3", quantized=False, device=None):
    import torch

    model_class, config = model_class_and_config(path, family)
    options = {
        "config": config, "local_files_only": True, "trust_remote_code": False,
        "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa", "output_loading_info": True,
    }
    if quantized:
        from transformers import BitsAndBytesConfig

        options.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": device})
    model, information = model_class.from_pretrained(path, **options)
    for field in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
        assert not information.get(field), (field, information)
    if device is not None and not quantized:
        model.to(device)
    return model, information


def sharing(model):
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    return {
        "same_parameter_object": embedding is head,
        "same_storage": embedding.data_ptr() == head.data_ptr(),
        "input_shape": list(embedding.shape), "output_shape": list(head.shape),
        "input_dtype": str(embedding.dtype), "output_dtype": str(head.dtype),
        "input_requires_grad": embedding.requires_grad, "output_requires_grad": head.requires_grad,
    }


class Allowed:
    def __init__(self, tokenizer, indices):
        self.tok = tokenizer
        self.eos = tokenizer.eos_token_id
        self.allowed = {position: set() for position in range(4)}
        self.sep = None
        self.prefix = None
        self.calls = {}
        self.events = []
        for parts in indices.values():
            assert len(parts) == 4
            for position, token in enumerate(parts):
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                assert len(token_ids) == 1
                self.allowed[position].add(token_ids[0])
        self.allowed[4] = {self.eos}
        assert sum(len(self.allowed[position]) for position in range(4)) == 930
        assert all(self.allowed[position] for position in range(4))

    def bind(self, prompt_ids, text):
        assert text.endswith("### Response:")
        encoded = self.tok(text, add_special_tokens=True, return_offsets_mapping=True)
        assert encoded["input_ids"] == list(prompt_ids)
        beginning = len(text) - len("Response:")
        self.sep = [
            token_id for token_id, offsets in zip(encoded["input_ids"], encoded["offset_mapping"])
            if offsets[1] > beginning
        ]
        assert self.sep and list(prompt_ids)[-len(self.sep):] == self.sep
        self.prefix = list(prompt_ids)

    def __call__(self, batch_id, sentence):
        assert self.prefix is not None, "Bind the actual original LC-Rec prompt first"
        sentence_ids = sentence.tolist()
        assert sentence_ids[:len(self.prefix)] == self.prefix
        beginning = len(self.prefix)
        assert sentence_ids[beginning - len(self.sep):beginning] == self.sep
        position = len(sentence_ids) - beginning
        assert position in self.allowed, ("Unexpected generation position", position)
        self.calls[position] = self.calls.get(position, 0) + 1
        if len(self.events) < 32:
            self.events.append({"position": position, "batch_id": batch_id, "allowed_count": len(self.allowed[position])})
        return sorted(self.allowed[position])
