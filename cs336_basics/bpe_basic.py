from collections import Counter

def init_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    # 训练阶段不需要反转的字典，在推理的时候，编码和解码才需要两个方向的vocab
    # reversed_vocab = {v: k for k, v in vocab.items()}
    return vocab

def pre_tokenization(text:str):
    pre_tokens_list = text.split()  # 这里简化版本用默认所有空白分隔符切分
    word_list = []
    for token in pre_tokens_list:
        word_list.append(tuple(token))  # tuple(str)返回的是逐char拆开的，一个字符一个元组内元素
    # python没有元组生成式!!下面的写法append的是一个迭代器对象
        # tuples_list.append((char for char in token))
    return word_list

def get_stat(word_list: dict):
    pair_dict = Counter()
    for word, freq in word_list.items():
        for i, char in enumerate(word):
            if i > 0:
                # 这里用元组对来作为字典的键，可以保存是哪两个元素合并得来的，用于最后生成merges合并对信息；
                # 如果用两个串直接拼起来，会丢失合并前信息，比如low是lo+w还是l+ow，导致无法生成正确的merges
                pair_dict[(word[i - 1], word[i])] += freq     

    return pair_dict
    
def merge(word_freqs:dict, top_pair:tuple):
    p0, p1 = top_pair
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        # 因为合并后索引会发生改变，涉及指针跳跃操作，下面的合并操作使用while来灵活控制指针
        i = 0
        while i < len(word):
            # 处理伴随合并、跳跃扫描的时候，永远优先让指针指向“即将被处理的第一个元素”，然后向前探测(用i+1，而不是i-1)
            if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                new_word.append(p0 + p1)  # 拼合字符串
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    
    return new_word_freqs
        # while这里向后看的逻辑(看i-1)会导致+=2的时候跳过单独的结尾元素
        # if i > 0 and word[i - 1] == top_pair[0] and word[i] == top_pair[1]:
        #     new_word[-1] = word[i - 1] + word[i]  # 拼合字符串
        #     i += 2
        # else:
        #     new_word.append(word[i])
        #     i += 1


def BPE(text:str):
    merges = []
    vocab = init_vocab()
    next_vocab_id = len(vocab)

    word_list = pre_tokenization(text)
    # 建立核心数据结构word_freqs,记录了当前的合并进度和每个词的词频(词频value不变，合并或者说分词状态key在变)
    word_freqs = Counter(word_list)
    # 统计当前合并状态下，每个pair的频率
    pair_dict = get_stat(word_freqs)
    # 获取最高频pair，同频率按字典序
    top_pair = max(pair_dict.keys(), key=lambda x: (pair_dict[x], x))  # max会按照每个元素进入key的返回结果来比大小
    # 在word_freqs中执行合并
    word_freqs = merge(word_freqs, top_pair)
    merges.append(top_pair)
    vocab[next_vocab_id] = top_pair[0] + top_pair[1]






if __name__ == "__main__":
    BPE('''low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest''')

    