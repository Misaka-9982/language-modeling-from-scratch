from collections import Counter

def get_stat(text: str, vocab: dict):
    # pre_tokenization
    pre_tokens_list = text.split()
    tuples_list = []
    for token in pre_tokens_list:
        tuples_list.append(tuple(token))  # tuple(str)返回的是逐char拆开的，一个字符一个元组内元素
    # python没有元组生成式!!下面的写法append的是一个迭代器对象
        # tuples_list.append((char for char in token))
    
    frequency_dict_origin = Counter(tuples_list)
    frequency_dict = Counter()
    # scan_begin
    for pre_token in frequency_dict_origin:
        for i, char in enumerate(pre_token):
            if i > 0:
                # 这里用元组对来作为字典的键，可以保存是哪两个元素合并得来的，用于最后生成merges合并对信息；
                # 如果用两个串直接拼起来，会丢失合并前信息，比如low是lo+w还是l+ow，导致无法生成正确的merges
                frequency_dict[(pre_token[i - 1], pre_token[i])] += frequency_dict_origin[pre_token]     

    return frequency_dict
    
def merge(vocab, pair):
    pass

def init_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    reversed_vocab = {v: k for k, v in vocab.items()}
    return vocab, reversed_vocab

def BPE(text:str):
    merges = []
    vocab, reversed_vocab = init_vocab()
    frequency_dict = get_stat(text, vocab)
    top_pair = max(frequency_dict.keys(), key=lambda x: (frequency_dict[x], x))  # max会按照每个元素进入key的返回结果来比大小
    merge(vocab, top_pair)





if __name__ == "__main__":
    BPE('''low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest''')

    