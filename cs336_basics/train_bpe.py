import os
import regex as re
import multiprocessing
import json
import pickle
from typing import BinaryIO
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from tqdm import tqdm

import cProfile
import pstats

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L12
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_PROCESSES = 8


class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(args):
    filename, start, end = args
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # split the chunk by the special token
    docs = re.split("|".join(re.escape(tok) for tok in SPECIAL_TOKENS), chunk)
    
    pre_tokens = []
    for doc in docs:
        # Find all pre-tokens in the document
        for match in re.finditer(PAT, doc):
            pre_tokens.append(match.group(0))

    return pre_tokens  # Return the list of pre-tokens for this chunk


def parallel_pretokenize(filename, num_processes, boundaries):
    # Create list of (filename, start, end) tuples
    chunks = [(filename, boundaries[i], boundaries[i+1]) for i in range(len(boundaries) - 1)]

    # Use multiprocessing to process each chunk
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(pretokenize_chunk, chunks)

    # Combine all results into a single list of pre-tokens
    all_pre_tokens = []
    for pre_tokens in results:
        all_pre_tokens.extend(pre_tokens)

    return all_pre_tokens  # Return the combined list of pre-tokens


def update_byte_tokens(
        tokens: List[List[bytes]],
        pair_counts: Counter,
        pair_occurrences: Dict[Tuple[bytes, bytes], set],
        merge_pair: Tuple[bytes, bytes]
    ) -> List[List[bytes]]:
    """
    Update the tokens and statistics after merging a pair of bytes.
    
    Args:
        tokens: List of tokens, each token being a list of bytes
        pair_counts: Counter of byte pair frequencies
        pair_occurrences: Dictionary mapping pairs to their locations
        merge_pair: The pair (byte1, byte2) being merged
    
    Returns:
        The updated tokens (though they're modified in-place)
    """
    # Get all occurrences of the pair we're merging
    occurrences = pair_occurrences.get(merge_pair, set())
    if not occurrences:
        return tokens
    
    merged_byte = merge_pair[0] + merge_pair[1]

    for token_idx in occurrences:
        token = tokens[token_idx]
        i = 0
        while i < len(token) - 1:
            if (token[i], token[i + 1]) == merge_pair:
                # Update pair_counts and pair_occurrences before modifying token
                if i > 0:
                    left_pair = (token[i - 1], token[i])
                    pair_counts[left_pair] -= 1
                    if pair_counts[left_pair] == 0:
                        del pair_counts[left_pair]
                        del pair_occurrences[left_pair]

                if i + 2 < len(token):
                    right_pair = (token[i + 1], token[i + 2])
                    pair_counts[right_pair] -= 1
                    if pair_counts[right_pair] == 0:
                        del pair_counts[right_pair]
                        del pair_occurrences[right_pair]

                # Replace the pair with the merged byte
                token[i:i + 2] = [merged_byte]  # In-place replacement
                # No need to increment i here, because the current i now has a new merged token
            else:
                i += 1

        i = 0
        while i < len(token):
            if i > 0 and token[i] == merged_byte:
                new_left_pair = (token[i - 1], token[i])
                pair_counts[new_left_pair] += 1
                pair_occurrences[new_left_pair].add(token_idx)
            if i < len(token) - 1 and token[i] == merged_byte:
                new_right_pair = (token[i], token[i + 1])
                pair_counts[new_right_pair] += 1
                pair_occurrences[new_right_pair].add(token_idx)
            i += 1

    # Remove the pair from pair_counts
    del pair_counts[merge_pair]
    del pair_occurrences[merge_pair]

    return tokens


def initialize_pair_counts(tokens: List[List[bytes]]):
    """
    Build initial pair_counts and pair_occurrences for a list of tokens.
    Each token is a list of bytes (as single-byte bytes or merged byte chunks).
    Returns:
        pair_counts: Counter mapping (byte1, byte2) -> count of occurrences.
        pair_occurrences: dict mapping (byte1, byte2) -> set of (token_idx)
    """
    pair_counts = Counter()
    pair_occurrences = defaultdict(set)

    for tidx, token in enumerate(tokens):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += 1
            pair_occurrences[pair].add(tidx)

    return pair_counts, pair_occurrences


def bpe_merge(pre_tokens, vocab_size, special_tokens=None):
    # Convert pre_tokens to byte tokens
    # byte_tokens = [list(token.encode("utf-8")) for token in pre_tokens]
    # byte_tokens = [[bytes([b]) for b in token] for token in byte_tokens]    # List[List[bytes]]
    byte_tokens: List[List[bytes]] = [
        [bytes([b]) for b in token.encode("utf-8")]
        for token in pre_tokens
    ]

    pair_counts, pair_occurrences = initialize_pair_counts(byte_tokens)

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    merges: List[Tuple[bytes, bytes]] = []

    # Handle special tokens
    if special_tokens is None:
        special_tokens = []
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_token_id] = token_bytes
            next_token_id += 1

    while len(vocab) < vocab_size and pair_counts:
        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count == 0:
            break

        # Create a new token from the best pair
        new_token = b"".join(best_pair)
        if new_token not in vocab.values():
            vocab[next_token_id] = new_token
            merges.append(best_pair)
            next_token_id += 1

        # Update byte_tokens with the new token
        byte_tokens = update_byte_tokens(byte_tokens, pair_counts, pair_occurrences, best_pair)        

    return vocab, merges


def train_bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] = None, # A list of strings to add to the vocabulary
)-> BPETokenizerParams:
    """
    Train a BPE tokenizer on the input text file.
    
    Args:
        input_path: Path to the input text file.
        vocab_size: Desired size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.
    
    Returns:
        A BPETokenizerParams object containing the vocabulary and merges.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8"))

        # parallel pretokenization
        pre_tokens = parallel_pretokenize(input_path, NUM_PROCESSES, boundaries)

    print("Pre-tokenization complete. Number of pre-tokens:", len(pre_tokens))

    # sequential bpe_merge
    vocab, merges = bpe_merge(pre_tokens, vocab_size, special_tokens)

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    filename = "../data/TinyStoriesV2-GPT4-train.txt"
    
    # filename = "../data/test.txt"
    vocab_path = "../data/vocab_tinystories.pkl"
    merges_path = "../data/merges_tinystories.pkl"
    
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    tokenizer_params = train_bpe_tokenizer(
        input_path=filename,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # serialize the tokenizer params
    with open(vocab_path, 'wb') as f:
        pickle.dump(tokenizer_params.vocab, f)
            
    with open(merges_path, 'wb') as f:
        pickle.dump(tokenizer_params.merges, f)



    # with open(vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)
        
    # with open(merges_path, 'rb') as f:
    #     merges = pickle.load(f)