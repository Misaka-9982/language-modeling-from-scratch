from collections import Counter
import regex as re

def _init_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for special in special_tokens:
        vocab[len(vocab)] = special.encode('utf-8')
    return vocab


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


# 只在循环开始前调用一次，全表扫描word_freqs，初始化pair_freqs  后续在pair_freqs上增量更新
def _get_stats(word_freqs):
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        for i, byte in enumerate(word):
            if byte == word[-1]:
                continue
            pair_freqs[(word[i], word[i + 1])] += freq

    return pair_freqs


# 合并以及pair_freqs的增量更新逻辑
def _merge(word_freqs, pair_freqs, vocab:dict[int,bytes], merges:list):
    top_pair = max(pair_freqs, key=lambda k: (pair_freqs[k], k))
    new_id = len(vocab)
    vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]  # 生成新token
    merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))
    new_word_freqs = Counter()

    # 内部函数，减少pair的频率，频率为空时从字典中删除
    def _decrease_pair(word, freq):
        for j in range(len(word) - 1):
            pair_freqs[(word[j], word[j + 1])] -= freq
            if pair_freqs[(word[j], word[j + 1])] <= 0:
                del pair_freqs[(word[j], word[j + 1])]
    
    def _increase_pair(new_word, freq):
        for j in range(len(new_word) - 1):
            pair_freqs[(new_word[j], new_word[j + 1])] += freq


    # 执行合并，更新word_freqs
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):  # while不能直接写成i < len(word) - 1，否则会漏掉单独最后一个的情况
            if i < len(word) - 1 and word[i] == top_pair[0] and word[i + 1] == top_pair[1]:  # 第一个条件防止i+1越界
                # new_word.append(word[i] + word[i + 1])  # bug: 字符级str合并的逻辑在字节级上不能照搬，会变成整数求和
                # 合并 生成新token
                new_word.append(new_id)  # 字节级实现这里要填入合并后分配的新ID，而不是求和
                # print(pair_freqs[top_pair])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if len(new_word) == len(word):  # 如果没有发生合并就直接跳过
            new_word_freqs[word] += freq
            continue
        # 更新pair_freqs的逻辑
        # 减旧词中的pair频率，增加新词中的pair频率
        _decrease_pair(word, freq)
        _increase_pair(new_word, freq)
        new_word_freqs[tuple(new_word)] += freq
    return new_word_freqs, pair_freqs, vocab, merges
    


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = _init_vocab(special_tokens)
    word_freqs = _pre_tokenize(input_path, special_tokens)
    # 这里定义局部增量更新的pair_freqs，不用重复统计不受影响的pair
    pair_freqs = _get_stats(word_freqs)
    merges = []

    # for 训练循环
    for _ in range(vocab_size - len(vocab)):
        word_freqs, pair_freqs, vocab, merges = _merge(word_freqs, pair_freqs, vocab, merges)

    return vocab, merges



if __name__ == "__main__":
    train_bpe(input_path=r'data/TinyStoriesV2-GPT4-valid.txt', vocab_size=280, special_tokens=['<|endoftext|>'])
