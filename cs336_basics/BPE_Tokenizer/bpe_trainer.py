from collections import Counter
import regex as re

def _init_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_vocab_id = 256
    for special in special_tokens:
        vocab[next_vocab_id] = special.encode('utf-8')
        next_vocab_id += 1
    return vocab, next_vocab_id

def _pre_tokentize(input_path: str, special_tokens: list[str]):
    # GPT-2的预分词正则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_PAT = re.compile(PAT)

    # 把special_tokens转义成可以作为正则pattern的形式
    escaped_special_tokens = [re.escape(token) for token in special_tokens]

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            pass
            # re.split(pattern=escaped_special_tokens, string=line)




def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab, next_vocab_id = _init_vocab(special_tokens)
    word_freqs = _pre_tokentize(input_path, special_tokens)



if __name__ == "__main__":
    train_bpe(input_path='', vocab_size=0, special_tokens=['<|endoftext|>'])
