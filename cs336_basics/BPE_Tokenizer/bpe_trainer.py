from collections import Counter
import regex as re

def _init_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_vocab_id = 256
    for special in special_tokens:
        vocab[next_vocab_id] = special.encode('utf-8')
        next_vocab_id += 1
    return vocab, next_vocab_id

# 先去除special_tokens，切分输入流，然后再进行pretokenization
def _pre_tokenize(input_path: str, special_tokens: list[str]):
    # 全局词频分词状态表-原始文本的化身，只保留预分词切分后的结果与其出现的频率
    # 也就是一个元组内是一个预分词的token，然后元组内的元素表明了实际的合并情况
    # eg: ('t', 'o', 'k', 'e', 'n'): 10
    word_freqs = Counter()

    # 把special_tokens转义成可以作为正则pattern的形式  A | B | C
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    rm_special_tokens_PAT = "|".join(escaped_special_tokens)

    # GPT-2的预分词正则
    pre_tokenize_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenize_PAT = re.compile(pre_tokenize_PAT)

    with open(input_path, 'r', encoding='utf-8') as f:
        # 逐行流式读取文件内容
        for line in f:
            # 先去除special_tokens并分割
            splited_chunks = re.split(rm_special_tokens_PAT, line)
            for chunk in splited_chunks:
                if chunk:
                    for pretoken in re.finditer(pre_tokenize_PAT, chunk):
                        # group()把Match对象转成str，然后再转utf-8的字节串，tuple打散成0-255的整数(字节)序列
                        word_freqs[tuple(pretoken.group().encode('utf-8'))] += 1

    return word_freqs

def _merge(word_freqs, top_pair):
    pass

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab, next_vocab_id = _init_vocab(special_tokens)
    word_freqs = _pre_tokenize(input_path, special_tokens)
    pass



if __name__ == "__main__":
    train_bpe(input_path=r'data/TinyStoriesV2-GPT4-valid.txt', vocab_size=0, special_tokens=['<|endoftext|>'])
