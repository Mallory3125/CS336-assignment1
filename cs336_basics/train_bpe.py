import yaml,os
from bpe import train_bpe_tokenizer

FILE_NAME = "/root/autodl-tmp/cs336-assign1/TinyStoriesV2-GPT4-valid.txt"
# FILE_NAME = "/root/autodl-tmp/cs336-assign1/owt_valid.txt"

def save_tokenizer_yaml(vocab, merges, fname):
    "Save vocab and merges to a YAML file with UTF-8 decoding for readability."
    # Convert bytes → string for readability
    vocab_serializable = {
        k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }
    merges_serializable = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]
    
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=False
        )

def load_tokenizer_yaml(fname):
    "Load vocab and merges from a YAML file, converting strings back to bytes."
    with open(fname, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    vocab_loaded = {
        int(k): v.encode("utf-8") if isinstance(v, str) else v
        for k, v in data["vocab"].items()
    }
    merges_loaded = [
        (a.encode("utf-8"), b.encode("utf-8")) for a, b in data["merges"]
    ]
    return vocab_loaded, merges_loaded


vocab, merges = train_bpe_tokenizer(
    input_path=FILE_NAME,
    vocab_size=10_000,
    special_tokens=['<|endoftext|>'],
    num_workers=os.cpu_count() 
)

save_tokenizer_yaml(vocab,merges,'tokenizer_tinystory_valid.yaml')