from typing import Iterable, Iterator

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

    def from_file(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs using the BPE tokenizer.
        """
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass