# profile_bpe.py
import cProfile
import pstats
import io,os
from bpe import train_bpe_tokenizer

FILE_NAME = "/root/autodl-tmp/cs336-assign1/TinyStoriesV2-GPT4-valid.txt"

def profile_with_cprofile():
    """Profile with cProfile and print results."""
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run your training
    vocab, merges = train_bpe_tokenizer(
        FILE_NAME,
        vocab_size=10000,
        special_tokens=['<|endoftext|>'],
        num_workers=os.cpu_count() 
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

# (base) root@autodl-container-b8cd119e52-2726b96e:/assign1/CS336-assignment1/cs336_basics# uv run profile_bpe.py 

# TOP 20 FUNCTIONS BY CUMULATIVE TIME:
#          667687606 function calls (667687270 primitive calls) in 626.360 seconds

#    Ordered by: cumulative time
#    List reduced from 509 to 20 due to restriction <20>

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.063    0.063  626.360  626.360 bpe.py:326(train_bpe_tokenizer)
#         1  275.898  275.898  620.622  620.622 bpe.py:12(merge_bpe_fast)
# 127740473  142.970    0.000  223.658    0.000 {built-in method builtins.any}
#      9790   55.529    0.006   81.642    0.008 {built-in method builtins.max}
# 311022907   80.720    0.000   80.720    0.000 bpe.py:48(<genexpr>)
# 128536959/128536919   39.251    0.000   39.251    0.000 {built-in method builtins.len}
#  99657168   26.112    0.000   26.112    0.000 bpe.py:37(<lambda>)
#         1    0.001    0.001    5.674    5.674 bpe.py:297(parallel_process_chunks)
#         1    0.000    0.000    5.265    5.265 pool.py:738(__exit__)
#         1    0.000    0.000    5.193    5.193 pool.py:654(terminate)
#        97    0.000    0.000    5.005    0.052 util.py:208(__call__)
#         1    0.000    0.000    5.004    5.004 pool.py:680(_terminate_pool)
#         1    0.000    0.000    4.917    4.917 pool.py:671(_help_stuff_finish)
#         1    0.075    0.075    4.917    4.917 {method 'acquire' of '_multiprocessing.SemLock' objects}
#       192    0.000    0.000    4.844    0.025 connection.py:202(send)
#       3/1    0.000    0.000    4.842    4.842 threading.py:1018(_bootstrap)
#       3/1    0.000    0.000    4.842    4.842 threading.py:1058(_bootstrap_inner)
#       3/1    0.000    0.000    4.842    4.842 threading.py:1001(run)
#         1    0.000    0.000    4.842    4.842 pool.py:527(_handle_tasks)
#       197    0.000    0.000    4.842    0.025 connection.py:406(_send_bytes)



# TOP 20 FUNCTIONS BY TOTAL TIME:
#          667687606 function calls (667687270 primitive calls) in 626.360 seconds

#    Ordered by: internal time
#    List reduced from 509 to 20 due to restriction <20>

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1  275.898  275.898  620.622  620.622 bpe.py:12(merge_bpe_fast)
# 127740473  142.970    0.000  223.658    0.000 {built-in method builtins.any}
# 311022907   80.720    0.000   80.720    0.000 bpe.py:48(<genexpr>)
#      9790   55.529    0.006   81.642    0.008 {built-in method builtins.max}
# 128536959/128536919   39.251    0.000   39.251    0.000 {built-in method builtins.len}
#  99657168   26.112    0.000   26.112    0.000 bpe.py:37(<lambda>)
#   198/194    3.529    0.018    3.529    0.018 {built-in method posix.read}
#       149    0.571    0.004    0.571    0.004 {method 'poll' of 'select.poll' objects}
#     11716    0.277    0.000    0.364    0.000 {built-in method posix.waitpid}
#    248738    0.230    0.000    0.230    0.000 {method 'get' of 'dict' objects}
#        99    0.185    0.002    0.185    0.002 {built-in method _pickle.loads}
#        97    0.177    0.002    0.408    0.004 __init__.py:669(update)
#        96    0.144    0.001    0.144    0.001 {built-in method posix.fork}
#    298399    0.133    0.000    0.133    0.000 {method 'append' of 'list' objects}
#      7230    0.095    0.000    0.123    0.000 selectors.py:351(register)
#       149    0.092    0.001    3.375    0.023 connection.py:1122(wait)
#      7104    0.091    0.000    4.382    0.001 process.py:224(exitcode)
#         1    0.075    0.075    4.917    4.917 {method 'acquire' of '_multiprocessing.SemLock' objects}
#         1    0.063    0.063  626.360  626.360 bpe.py:326(train_bpe_tokenizer)
#        33    0.056    0.002    0.057    0.002 <frozen importlib._bootstrap>:74(__new__)



# Profile saved to 'bpe_profile.prof'
# View with: snakeviz bpe_profile.prof