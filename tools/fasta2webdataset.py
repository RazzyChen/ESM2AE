#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import mmap
import multiprocessing as mp
import os
import queue
import shutil
import webdataset as wds
from Bio import SeqIO
from tqdm import tqdm

# ==============================================================================
#  FASTA Parsers (Producer part)
# ==============================================================================

def get_sequence_count(fasta_file: str) -> int:
    """Counts sequences using a fast mmap method."""
    print("Counting total sequences with mmap (fast)...")
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            count = mm.count(b">")
    print(f"Found {count} sequences.")
    return count

def read_fasta_in_chunks_mmap(fasta_file: str, chunk_size: int):
    """Generator that reads a FASTA file in chunks using mmap."""
    chunk = []
    idx = 0
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            positions = []
            pos = mm.find(b">")
            while pos != -1:
                positions.append(pos)
                pos = mm.find(b">", pos + 1)

            for i in range(len(positions)):
                start_pos = positions[i]
                end_pos = positions[i + 1] if i + 1 < len(positions) else len(mm)
                record_bytes = mm[start_pos:end_pos]
                header_end_pos = record_bytes.find(b'\n')
                header = record_bytes[1:header_end_pos].decode(errors='ignore')
                seq = record_bytes[header_end_pos+1:].replace(b'\n', b'')
                
                chunk.append((idx, header, seq))
                idx += 1
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk

def producer(
    task_queue: mp.Queue,
    fasta_file: str,
    chunk_size: int,
    num_workers: int,
):
    """Producer: Reads FASTA, creates chunks, and puts them in the task queue."""
    print("[Producer] Starting to read FASTA...")
    chunk_generator = read_fasta_in_chunks_mmap(fasta_file, chunk_size)

    for chunk in chunk_generator:
        task_queue.put(chunk)

    print("[Producer] All chunks queued. Sending termination signals.")
    for _ in range(num_workers):
        task_queue.put(None)

# ==============================================================================
#  Worker and Writer (Consumer part)
# ==============================================================================

def worker_process(task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker: Processes chunks of FASTA data."""
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(None) # Signal writer that this worker is done
            break

        records_to_write = []
        for idx, header, sequence in task:
            # Webdataset expects a dictionary of bytes
            records_to_write.append({
                "__key__": f"{idx:08d}",
                "fasta": sequence,
                "header": header.encode('utf-8')
            })
        result_queue.put(records_to_write)

def writer_process(
    result_queue: mp.Queue, output_dir: str, total_sequences: int, num_workers: int, shard_size: int
):
    """Writer: Writes processed data to webdataset shards."""
    print("[Writer] Process started.")
    workers_done = 0
    shard_count = 0
    
    # Pattern for shard names
    pattern = os.path.join(output_dir, f'shard-%06d.tar')
    
    with tqdm(total=total_sequences, desc="Writing to WebDataset", unit="seq") as pbar:
        with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
            while workers_done < num_workers:
                try:
                    results_chunk = result_queue.get(timeout=120)
                    if results_chunk is None:
                        workers_done += 1
                        continue

                    for record in results_chunk:
                        sink.write(record)
                    pbar.update(len(results_chunk))

                except queue.Empty:
                    print("[Writer] Warning: Queue empty for 120s. Potential worker issue.")
                    if pbar.n >= total_sequences:
                        break

    print(f"\n[Writer] Finished writing {pbar.n} sequences to shards.")

# ==============================================================================
#  Main Coordinator
# ==============================================================================

def fasta_to_webdataset(
    fasta_file: str, output_dir: str, num_processes: int, chunk_size: int, shard_size: int
):
    """Main coordinator for FASTA to WebDataset conversion."""
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2) # Leave cores for producer/writer

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing output directory: {output_dir}")
    os.makedirs(output_dir)

    print("--- FASTA to WebDataset Conversion ---")
    print(f"Worker processes: {num_processes}")
    print(f"Read chunk size: {chunk_size}")
    print(f"WebDataset shard size: {shard_size}")
    print("--------------------------------------")

    total_sequences = get_sequence_count(fasta_file)
    if total_sequences == 0:
        print("No sequences found. Exiting.")
        return

    with mp.Manager() as manager:
        task_queue = manager.Queue(maxsize=num_processes * 2)
        result_queue = manager.Queue(maxsize=num_processes * 2)

        producer_proc = mp.Process(
            target=producer, args=(task_queue, fasta_file, chunk_size, num_processes)
        )
        producer_proc.start()

        writer_proc = mp.Process(
            target=writer_process,
            args=(result_queue, output_dir, total_sequences, num_processes, shard_size),
        )
        writer_proc.start()

        worker_procs = []
        for _ in range(num_processes):
            p = mp.Process(target=worker_process, args=(task_queue, result_queue))
            p.start()
            worker_procs.append(p)

        producer_proc.join()
        print("[Main] Producer finished.")
        for p in worker_procs:
            p.join()
        print("[Main] All workers finished.")
        writer_proc.join()
        print("[Main] Writer finished.")

        print(f"\nSuccessfully converted {fasta_file} to WebDataset format in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to WebDataset format.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WebDataset shards")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes. Defaults to CPU count - 2.")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Number of sequences per read chunk")
    parser.add_argument("--shard_size", type=int, default=100000, help="Maximum number of sequences per .tar shard")

    args = parser.parse_args()

    fasta_to_webdataset(
        args.fasta_file, args.output_dir, args.processes, args.chunk_size, args.shard_size
    )
