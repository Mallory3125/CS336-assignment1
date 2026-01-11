import regex as re
import os
from typing import BinaryIO
from collections import Counter,defaultdict
from multiprocessing import Pool


#gpt-2 style regex-based pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_bpe_fast(
    frequency_table: dict[tuple[bytes], int], 
    vocab_size: int,
    num_tokens: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Highly optimized BPE merging with minimal allocations."""
    vocab = {idx: bytes([idx]) for idx in range(256)}
    next_token_id = 256
    merges = []
    max_merges = vocab_size - next_token_id - num_tokens
    
    # Use defaultdict instead of Counter
    pair_counts = defaultdict(int)
    for token_seq, freq in frequency_table.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += freq
    
    word_cnt = dict(frequency_table)
    
    for merge_iteration in range(max_merges):
        if not pair_counts:
            break
        
        most_frequent_pair = max(pair_counts.items(), 
                                key=lambda x: (x[1], x[0]))[0]
        token_a, token_b = most_frequent_pair
        
        merged_token = token_a + token_b
        vocab[next_token_id] = merged_token
        merges.append((token_a, token_b))
        
        new_word_cnt = {}
        
        for word_bytes, cnt in word_cnt.items():
            # Single-pass check
            has_merge = any(word_bytes[i] == token_a and word_bytes[i+1] == token_b 
                          for i in range(len(word_bytes) - 1))
            
            if not has_merge:
                new_word_cnt[word_bytes] = cnt
                continue
            
            # Decrement old pairs
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                pair_counts[pair] -= cnt
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            
            # Apply merge
            new_word = []
            i = 0
            while i < len(word_bytes):
                if (i < len(word_bytes) - 1 and 
                    word_bytes[i] == token_a and word_bytes[i + 1] == token_b):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word_bytes[i])
                    i += 1
            
            new_word = tuple(new_word)
            new_word_cnt[new_word] = cnt
            
            # Add new pairs
            for i in range(len(new_word) - 1):
                pair_counts[(new_word[i], new_word[i + 1])] += cnt
        
        word_cnt = new_word_cnt
        next_token_id += 1
    
    return vocab, merges


def merge_bpe_optimized(
    frequency_table: dict[tuple[bytes], int], 
    vocab_size: int,
    num_tokens: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Optimized BPE merging with incremental pair count updates.
    
    Args:
        frequency_table: Dictionary mapping token sequences to their frequencies
        vocab_size: Target vocabulary size
        num_tokens: Number of special tokens to reserve space for
    
    Returns:
        vocab: Dictionary mapping token IDs to byte sequences
        merges: List of merge operations as (token_a, token_b) pairs
    """
    # Initialize vocabulary with base bytes (0-255)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    next_token_id = 256
    
    merges = []
    max_merges = vocab_size - next_token_id - num_tokens
    
    # Build initial pair counts from frequency table
    pair_counts = Counter()
    for token_seq, freq in frequency_table.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += freq
    
    # Keep word count dictionary for incremental updates
    word_cnt = dict(frequency_table)
    
    # Perform merges
    for merge_iteration in range(max_merges):
        if not pair_counts:
            break
        
        # Find most frequent pair with lexicographic tie-breaking
        most_frequent_pair = max(pair_counts.items(), 
                                key=lambda x: (x[1], x[0]))[0]
        token_a, token_b = most_frequent_pair
        
        # Create new merged token
        merged_token = token_a + token_b
        vocab[next_token_id] = merged_token
        merges.append((token_a, token_b))
        
        # Incrementally update only affected sequences
        new_word_cnt = {}
        new_pair_counts = Counter(pair_counts)  # Copy current counts
        
        for word_bytes, cnt in word_cnt.items():
            # Get all pairs in this word
            old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
            
            # Skip words that don't contain the merged pair
            if most_frequent_pair not in old_pairs:
                new_word_cnt[word_bytes] = cnt
                continue
            
            # Apply merge to this word
            new_word = []
            i = 0
            while i < len(word_bytes):
                if (i < len(word_bytes) - 1 and 
                    word_bytes[i] == token_a and 
                    word_bytes[i + 1] == token_b):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word_bytes[i])
                    i += 1
            
            new_word = tuple(new_word)
            new_word_cnt[new_word] = cnt
            
            # Update pair counts: subtract old pairs
            for pair in old_pairs:
                new_pair_counts[pair] -= cnt
                if new_pair_counts[pair] <= 0:
                    del new_pair_counts[pair]
            
            # Update pair counts: add new pairs
            new_pairs = list(zip(new_word[:-1], new_word[1:]))
            for pair in new_pairs:
                new_pair_counts[pair] += cnt
        
        # Update state for next iteration
        word_cnt = new_word_cnt
        pair_counts = new_pair_counts
        next_token_id += 1
    
    return vocab, merges


def pre_tokenize(
    chunks: list[bytes],
    pat: str
) ->  dict[tuple[bytes], int]:
    # use the given regex to pretokenize and count, get the frequency table
    token_freq = Counter()
    
    for chunk in chunks:
        try:
            # decode bytes to str for regex matching
            text = chunk.decode('utf-8')
        except UnicodeDecodeError:
            # Skip chunks that can't be decoded
            continue
        
        for match in re.finditer(pat, text):
            token_text = match.group()
            # Convert token to bytes and then to tuple of individual bytes
            token_bytes = token_text.encode('utf-8')
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            token_freq[token_tuple] += 1
    
    return dict(token_freq)


def remove_special_token_in_chunk(
    file_path: str | os.PathLike,  
    start: int,
    end: int,
    special_tokens: list[bytes],          
) -> list[bytes]:
    """
    Remove special tokens from a file chunk and return the segments.
    """
    # Open file in this process
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)
    
    if not special_tokens:
        return [chunk_data] if chunk_data else []
    
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = b"|".join(escaped_tokens)
    segments = re.split(pattern, chunk_data)
    segments = [seg for seg in segments if seg]
    
    return segments


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def process_chunk_and_pretokenize(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[bytes],
    pat: str = PAT
) -> dict[tuple[bytes], int]:
    """Combined function for parallel processing"""
    # Remove special tokens
    segments = remove_special_token_in_chunk(file_path, start, end, special_tokens)
    
    # Pre-tokenize segments
    token_freq = pre_tokenize(segments, pat)
    
    return token_freq

def parallel_process_chunks(
    file_path: str | os.PathLike,
    num_process: int,
    special_tokens: list[bytes],
    pat: str = PAT
):
    """Simple parallel processing file"""
    # Find boundaries
    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_process, special_tokens[0])
    
    # Create argument tuples
    args = [
        (file_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]
    
    # Process chunks in parallel
    with Pool(processes=num_process) as pool:
        chunk_results = pool.starmap(process_chunk_and_pretokenize, args)
    
    # Merge frequency dictionaries from all chunks
    merged_freq = Counter()
    for chunk_freq in chunk_results:
        merged_freq.update(chunk_freq)
    
    return dict(merged_freq)


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Complete BPE training."""
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    
    # Step 1: Parallel pre-processing (special tokens removed here)
    token_frequencies = parallel_process_chunks(
        input_path,num_workers, special_tokens_bytes
    )
    
    # Step 2: Sequential merging (no special tokens involved)
    vocab, merges = merge_bpe_fast(
        token_frequencies,
        vocab_size,
        len(special_tokens_bytes)
    )
    
    # Step 3: Add special tokens to final vocabulary
    next_id = 256 + len(merges)
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode('utf-8')
        next_id += 1
    
    return vocab, merges



if __name__ == '__main__':
    filename = './data/TinyStoriesV2-GPT4-valid_short.txt'
    special_tokens = ['<|endoftext|>']
    vocab_size = 1024
    num_process = 4
    
    # with open(filename, 'rb') as f:
    #     boundaries = find_chunk_boundaries(f, num_process, special_tokens[0])

    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     chunk = remove_special_token_in_chunk(filename,start, end,special_tokens)  
    # result = parallel_process_chunks(filename, num_process, special_tokens,PAT)
    # for i, (key, value) in enumerate(result.items()):
    #     if i < 3:  
    #         print(f"{key}: {value}")
    vocab, merges = run_train_bpe(filename,vocab_size,special_tokens, num_process)
