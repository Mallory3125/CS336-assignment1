import regex as re
import os
from typing import BinaryIO
from collections import Counter,defaultdict
from multiprocessing import Pool


#gpt-2 style regex-based pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_bpe_optimized(
    frequency_table: dict[tuple[bytes], int], 
    vocab_size: int,
    num_tokens: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Optimized BPE merging with incremental pair count updates.
    """    
    # Initialize vocabulary with base bytes (0-255)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    next_token_id = 256
    
    merges = []
    max_merges = vocab_size - next_token_id - num_tokens
    
    # Build initial pair counts index
    pair_counts = Counter()
    for token_seq, freq in frequency_table.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += freq
    
    # Keep track of where each pair appears
    pair_positions = defaultdict(set)
    for seq_id, (token_seq, freq) in enumerate(frequency_table.items()):
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_positions[pair].add(seq_id)
    
    # Convert dict to list for mutable indexing
    sequence_list = list(frequency_table.items())
    
    for merge_iteration in range(max_merges):
        if not pair_counts:
            break
        
        # Find most frequent pair
        most_frequent_pair, count = pair_counts.most_common(1)[0]
        token_a, token_b = most_frequent_pair
        
        # Create new merged token
        merged_token = token_a + token_b
        vocab[next_token_id] = merged_token
        merges.append((token_a, token_b))
        
        # Update only affected sequences
        sequences_to_update = pair_positions[most_frequent_pair]
        new_pair_counts = Counter()
        
        for seq_id in sequences_to_update:
            token_seq, freq = sequence_list[seq_id]
            
            # Remove old pair counts for this sequence
            for i in range(len(token_seq) - 1):
                old_pair = (token_seq[i], token_seq[i + 1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            
            # Perform merge in this sequence
            new_seq = []
            i = 0
            while i < len(token_seq):
                if (i < len(token_seq) - 1 and 
                    token_seq[i] == token_a and 
                    token_seq[i + 1] == token_b):
                    new_seq.append(merged_token)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            
            # Update sequence
            sequence_list[seq_id] = (tuple(new_seq), freq)
            
            # Add new pair counts for this sequence
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i + 1])
                new_pair_counts[new_pair] += freq
        
        # Update global pair counts
        pair_counts.update(new_pair_counts)
        
        # Rebuild pair positions index for affected pairs
        pair_positions.clear()
        for seq_id, (token_seq, freq) in enumerate(sequence_list):
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_positions[pair].add(seq_id)
        
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


def run_train_bpe(
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
    vocab, merges = merge_bpe_optimized(
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


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: # type: ignore
    """
    Train a Byte Pair Encoding (BPE) tokenizer.
    
    Args:
        input_path: Path to training text file
        vocab_size: Maximum vocabulary size (including base bytes, merges, and special tokens)
        special_tokens: List of special token strings to add to vocabulary
        
    Returns:
        vocab: Mapping from token ID to token bytes
        merges: Ordered list of merge operations as (byte_pair_a, byte_pair_b) tuples
    """
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            print(start, end)

    pass

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
