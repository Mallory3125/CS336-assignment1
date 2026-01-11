# profile_bpe.py
import cProfile
import pstats
import io
from bpe import train_bpe_tokenizer

def profile_with_cprofile():
    """Profile with cProfile and print results."""
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run your training
    vocab, merges = train_bpe_tokenizer(
        './tests/fixtures/tinystories_sample.txt',
        vocab_size=10000,
        special_tokens=['<|endoftext|>'],
        num_workers=4
    )
    
    # Stop profiling
    profiler.disable()
    
    # Print results
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    
    print("\nTOP 20 FUNCTIONS BY CUMULATIVE TIME:")
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print("\nTOP 20 FUNCTIONS BY TOTAL TIME:")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Save for visualization
    profiler.dump_stats('bpe_profile.prof')
    print("\nProfile saved to 'bpe_profile.prof'")
    print("View with: snakeviz bpe_profile.prof")

if __name__ == '__main__':
    profile_with_cprofile()
