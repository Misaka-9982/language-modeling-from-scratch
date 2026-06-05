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

        final_id_seq = []
        for pretoken in re.finditer(self.pre_tokenize_PAT, text):
            raw_bytes = pretoken.group().encode('utf-8')
            token_list = [bytes([b]) for b in raw_bytes]
            while True:
                top_pair = get_top_pair(token_list)
                if not top_pair:
                    break
                token_list = merge(token_list, top_pair)
            
            # 合并完成后转ID
            for merged_token in token_list:
                final_id_seq.append(self.reversed_vocab[merged_token])
        
        return final_id_seq


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass


    def decode(self, ids: list[int]) -> str:
        pass
    

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(vocab_filepath='cs336_basics/BPE_Tokenizer/tokenizer_vocab_1k_TinyStoriesV2-GPT4-valid.pkl', merges_filepath='cs336_basics/BPE_Tokenizer/tokenizer_merges_1k_TinyStoriesV2-GPT4-valid.pkl')
    tokenizer.encode('Hello World tokenizer! 你好，分词器！')
    pass