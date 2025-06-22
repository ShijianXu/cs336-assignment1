import pickle
import regex as re
from typing import Iterable, Iterator
from typing import List

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_dict = {token: idx for idx, token in self.vocab.items()}
        
    @classmethod
    def from_file(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
    
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        pre_tokens = self.pre_tokenize(text)

        # print(f"Pre-tokens: {pre_tokens}")

        # Merge consecutive special tokens if their concatenation is a special token
        merged_tokens = []
        i = 0
        while i < len(pre_tokens):
            token = pre_tokens[i]
            if token in self.special_tokens:
                j = i + 1
                merged = token
                # Keep adding while the concatenation is a valid special token
                while j < len(pre_tokens) and pre_tokens[j] in self.special_tokens:
                    candidate = merged + pre_tokens[j]
                    if candidate in self.special_tokens:
                        merged = candidate
                        j += 1
                    else:
                        break
                merged_tokens.append(merged)
                i = j
            else:
                merged_tokens.append(token)
                i += 1

        byte_tokens: List[List[bytes]] = []
        for token in merged_tokens:
            if token in self.special_tokens:
                byte_tokens.append([token])  # add the raw special token into the list
            else:
                byte_tokens.append([bytes([b]) for b in token.encode("utf-8")])

        # print(f"Byte tokens before merging: {byte_tokens}")
        encoded_ids = self.apply_merges(byte_tokens)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: List[int]) -> str:
        # Convert IDs back to tokens
        tokens = [self.vocab[idx] for idx in ids]

        # Join tokens into a single string
        byte_str = b"".join(tokens)

        # Decode bytes to string
        decoded_text = byte_str.decode("utf-8", errors="replace")
                
        return decoded_text

    def apply_merges(self, byte_tokens: List[List[bytes]]) -> List[int]:
        # Apply BPE merges to the byte tokens in the order specified in the merges
        for a, b in self.merges:
            for token_list in byte_tokens:
                if len(token_list) == 1 and token_list[0] in self.special_tokens:
                    # Skip special tokens
                    continue
                i = 0
                write = 0 # write pointer for in-place modification
                while i < len(token_list):
                    if i < len(token_list) - 1 and token_list[i] == a and token_list[i + 1] == b:
                        # Merge the tokens a and b
                        token_list[write] = a + b
                        i += 2
                    else:
                        # Keep the token as is
                        token_list[write] = token_list[i]
                        i += 1
                    write += 1
                
                # Resize the list to the new length after merging
                del token_list[write:]

        # print(f"Byte tokens after merging: {byte_tokens}")

        # Convert merged tokens to IDs
        encoded_ids = []
        for token_list in byte_tokens:
            for token in token_list:
                # Encode special token first, then check in vocab
                if token in self.special_tokens:
                    token = token.encode("utf-8")  # Convert special tokens to bytes

                if token in self.vocab_dict:
                    encoded_ids.append(self.vocab_dict[token])
                else:
                    raise ValueError(f"Token {token} not found in vocabulary.")

        return encoded_ids

    def pre_tokenize(self, text: str) -> list[str]:
        if not self.special_tokens:
            # If no special tokens, just split with regex
            return [match.group(0) for match in re.finditer(PAT, text)]

        # Escape and compile special token regex
        tok_re = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        result = []
        for part in re.split(tok_re, text):
            if part in self.special_tokens:
                result.append(part)
            elif part:
                result.extend(match.group(0) for match in re.finditer(PAT, part))

        return result


if __name__ == "__main__":
    
    vocab_path = "../data/vocab_tinystories.pkl"
    merges_path = "../data/merges_tinystories.pkl"
    special_tokens=["<|endoftext|>"]

    tokenizer = BPE_Tokenizer.from_file(vocab_path, merges_path, special_tokens)

    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")