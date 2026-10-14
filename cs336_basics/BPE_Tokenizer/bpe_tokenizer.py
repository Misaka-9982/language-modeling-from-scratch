import pickle
from collections.abc import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reversed_vocab = {token: id for id, token in vocab.items()}
        self.merges = merges
        self.merges_rank = {pair: idx for idx, pair in enumerate(merges)}  # merges优先级字典，pair: priority
        self.special_tokens = special_tokens

        # GPT-2的预分词正则
        pre_tokenize_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pre_tokenize_PAT = re.compile(pre_tokenize_PAT)
        
        # special_token正则  追加special_token到词表
        if self.special_tokens:
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_special_tokens.sort(reverse=True, key=lambda t: len(t))  # 让长的正则优先匹配，保证<|endoftext|><|endoftext|>连着，且单独的和二连均作为sp传入时，正确被切成一个token，而不是两个
            self.rm_special_tokens_PAT = f"({'|'.join(escaped_special_tokens)})"  # 加上圆括号变成捕获组，re.split的时候保留special_toknes

            for special in self.special_tokens:
                special_bytes = special.encode('utf-8')
                # 防止重复追加
                if special_bytes not in self.reversed_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = special_bytes
                    self.reversed_vocab[special_bytes] = new_id
        else:
            self.rm_special_tokens_PAT = None


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f_vocab, open(merges_filepath, "rb") as f_merges:
            vocab = pickle.load(f_vocab)
            merges = pickle.load(f_merges)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def encode(self, text: str) -> list[int]:
        # 内部函数，获取当前pretoken中优先级最高的pair
        def get_top_pair(pretoken: list[bytes]):
            pairs = []
            for i in range(len(pretoken) - 1):
                pairs.append((pretoken[i], pretoken[i + 1]))

            # 找优先级最高的
            top_pair = None
            top_rank = float('inf')
            for pair in pairs:
                rank = self.merges_rank.get(pair)
                if rank != None and rank < top_rank:  # 这里不能用if rank，rank最高是0，会直接跳过
                    top_rank = rank
                    top_pair = pair
            return top_pair
        
        def merge(token_list, top_pair):
            i = 0
            new_token_list = []
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == top_pair[0] and token_list[i + 1] == top_pair[1]:
                    new_token_list.append(top_pair[0] + top_pair[1])
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            return new_token_list

        # special_token处理
        if self.rm_special_tokens_PAT:
            splited_text = re.split(self.rm_special_tokens_PAT, text)
        else:
            splited_text = [text]

        final_ids = []
        for chunk in splited_text:
            # 跳过空串
            if not chunk:
                continue

            # special_tokens直接查ID，不经过正则切分
            if self.special_tokens and chunk in self.special_tokens:
                final_ids.append(self.reversed_vocab[chunk.encode('utf-8')])

            # 普通文本处理
            else:
                for pretoken in re.finditer(self.pre_tokenize_PAT, chunk):
                    raw_bytes = pretoken.group().encode('utf-8')
                    token_list = [bytes([b]) for b in raw_bytes]
                    while True:
                        top_pair = get_top_pair(token_list)
                        if not top_pair:
                            break
                        token_list = merge(token_list, top_pair)
                    
                    # 合并完成后转ID
                    for merged_token in token_list:
                        final_ids.append(self.reversed_vocab[merged_token])
        
        return final_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue

            chunk_ids = self.encode(chunk)

            for token_id in chunk_ids:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids]
        bytes_tokens = b"".join(bytes_list)
        return bytes_tokens.decode("utf-8", errors="replace")
    

    def print_vocab_longest(self, n: int):
        longest_tokens = sorted(self.vocab.values(), key=len, reverse=True)[:n]
        
        for token in longest_tokens:
            decoded = token.decode('utf-8', errors='replace')
            print(f"长度 {len(token):<2} | Token: {decoded!r}")


    def print_vocab_range(self, a: int, b: int):
        tokens = sorted(self.vocab.values(), key=len, reverse=False)[a:b]
        
        for token in tokens:
            decoded = token.decode('utf-8', errors='replace')
            print(f"长度 {len(token):<2} | Token: {decoded!r}")


    def print_vocab_by_length(self, length: int, top_k: int):
        # 1. 过滤出字节长度恰好等于 length 的 token
        matched_tokens = [token for token in self.vocab.values() if len(token) == length]
        
        # 2. 取前 top_k 个（若需按字典序或其他规则排序，可在此加上 sorted）
        selected_tokens = matched_tokens[:top_k]
        
        # 3. 打印输出
        for token in selected_tokens:
            decoded = token.decode('utf-8', errors='replace')
            print(f"长度 {len(token):<2} | Token: {decoded!r}")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(vocab_filepath='data/tokenizer_vocab_10k_TinyStoriesV2-GPT4-train.pkl', 
                                     merges_filepath='data/tokenizer_merges_10k_TinyStoriesV2-GPT4-train.pkl', 
                                     special_tokens=[])
    # encoded_text = tokenizer.encode('Hello World tokenizer! <|endoftext|> 你好，分词器！')
    # print(tokenizer.decode(encoded_text))
    # tokenizer.print_vocab(10)
    ids = tokenizer.encode('Today is a good day!')
    print(ids)
    tokenizer.print_vocab_by_length(length=6, top_k=100)
