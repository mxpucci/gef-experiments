#!/usr/bin/env python3
import os
import sys
import struct
import array
import math
import multiprocessing
from functools import partial

def calculate_stats(data, n):
    if n == 0:
        return 0, 0
        
    # Calculate Gaps Statistics
    # First gap is abs(s_0)
    first = data[0]
    sum_abs_gaps = abs(first)
    positive_gaps_count = 1 if first >= 0 else 0
    
    # Use iterators to avoid indexing overhead in the loop
    # This is faster than for i in range(1, n): data[i]
    it1 = iter(data)
    it2 = iter(data)
    next(it2) # Advance to second element
    
    # Process in pure Python (slow for very large datasets without numpy)
    for x, y in zip(it1, it2):
        gap = y - x
        if gap < 0:
            sum_abs_gaps -= gap # equivalent to += abs(gap)
        else:
            sum_abs_gaps += gap
            positive_gaps_count += 1
            
    return sum_abs_gaps, positive_gaps_count

def process_file_path(file_path):
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            # Read n (8 bytes)
            n_bytes = f.read(8)
            if len(n_bytes) < 8:
                return None
            n = struct.unpack('<Q', n_bytes)[0]

            # Read p (8 bytes)
            p_bytes = f.read(8)
            if len(p_bytes) < 8:
                return None
            p = struct.unpack('<q', p_bytes)[0]

            # Read data
            data = array.array('q')
            try:
                data.fromfile(f, n)
            except EOFError:
                pass
            
            actual_n = len(data)
            if actual_n != n:
                n = actual_n
            
            if n == 0:
                print(f"File: {os.path.basename(file_path)}\n"
                      f"n: 0\n"
                      f"p: {p}\n"
                      f"w: 0.00\n"
                      f"Average absolute gap: 0.000000\n"
                      f"Sortedness: 0.00%\n"
                      f"{'-' * 20}")
                return {
                    'n': 0,
                    'p': p,
                    'w': 0.0,
                    'avg_abs_gap': 0.0,
                    'sortedness': 0.0
                }

            # Calculate w (Width of the range)
            min_val = min(data)
            max_val = max(data)
            
            val_range = max_val - min_val
            if val_range < 0:
                w = 0.0
            else:
                # ceil(log2(max - min + 1))
                w = math.ceil(math.log2(val_range + 1))
            
            sum_abs_gaps, positive_gaps_count = calculate_stats(data, n)
            
            avg_abs_gap = sum_abs_gaps / n
            sortedness = (positive_gaps_count / n) * 100.0

            # Construct output string to print atomically to avoid interleaving
            output = (f"File: {os.path.basename(file_path)}\n"
                      f"n: {n}\n"
                      f"p: {p}\n"
                      f"w: {w:.2f}\n"
                      f"Average absolute gap: {avg_abs_gap:.6f}\n"
                      f"Sortedness: {sortedness:.2f}%\n"
                      f"{'-' * 20}")
            print(output)
            
            return {
                'n': n,
                'p': p,
                'w': w,
                'avg_abs_gap': avg_abs_gap,
                'sortedness': sortedness
            }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dataset_stats.py <file_or_folder>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    results = []
    
    # Use multiprocessing to speed up processing of multiple files
    # Determine number of processes (use fewer if CPU count is high to avoid I/O saturation? 
    # But this is CPU bound loop usually)
    num_processes = max(1, multiprocessing.cpu_count())

    if os.path.isfile(input_path):
        res = process_file_path(input_path)
        if res: results.append(res)
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".bin")]
        files.sort()
        if not files:
            print(f"No .bin files found in {input_path}")
        
        # Parallel processing
        if len(files) > 0:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap to get results as they complete (order doesn't strictly matter for gathering results, 
                # but we sorted files for consistent processing order if we were doing serial.
                # Here outputs might interleave if we just printed, but we construct string and print atomically-ish.
                # Actually, standard print is not atomic across processes, but it's okay for stats script.
                
                # To keep output order consistent with file sort order, we can map and then print,
                # but that delays feedback.
                # Let's just collect results.
                
                # For better user experience (seeing progress), we let workers print.
                # We sorted files list, but workers might finish out of order.
                
                mapped_results = pool.map(process_file_path, files)
                results = [r for r in mapped_results if r is not None]
                
    else:
        print(f"Invalid path: {input_path}")
        
    if len(results) > 1:
        count = len(results)
        avg_n = sum(r['n'] for r in results) / count
        avg_p = sum(r['p'] for r in results) / count
        avg_w = sum(r['w'] for r in results) / count
        avg_gap = sum(r['avg_abs_gap'] for r in results) / count
        avg_sortedness = sum(r['sortedness'] for r in results) / count
        
        print(f"Average Statistics ({count} datasets):")
        print(f"n: {avg_n:.2f}")
        print(f"p: {avg_p:.2f}")
        print(f"w: {avg_w:.2f}")
        print(f"Average absolute gap: {avg_gap:.6f}")
        print(f"Sortedness: {avg_sortedness:.2f}%")

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows support/safety
    main()
