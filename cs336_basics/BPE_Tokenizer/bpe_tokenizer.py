import pickle
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f_vocab, open(merges_filepath, "rb") as f_merges:
            vocab = pickle.load(f_vocab)
            merges = pickle.load(f_merges)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
    

if __name__ == "__main__":
    t = Tokenizer.from_files(vocab_filepath='cs336_basics/BPE_Tokenizer/tokenizer_vocab_1k_TinyStoriesV2-GPT4-valid.pkl', merges_filepath='cs336_basics/BPE_Tokenizer/tokenizer_merges_1k_TinyStoriesV2-GPT4-valid.pkl')
    pass