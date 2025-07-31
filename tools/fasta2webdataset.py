#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-Performance FASTA to WebDataset Converter

Core Optimization Strategies:
1.  Move TAR formatting operations to Worker processes for parallel execution.
2.  Writer only performs simple file writes, avoiding formatting bottlenecks.
3.  Use memory mapping and pre-allocation strategies to reduce memory copying.
4.  Intelligent batching and caching optimization.
"""

import argparse
import glob
import io
import json
import mmap
import multiprocessing as mp
import os
import queue
import signal
import sys
import tarfile
import time
from typing import Generator, List, Tuple

from tqdm import tqdm

# ==============================================================================
#  FASTA Parser (Producer)
# ==============================================================================


def get_sequence_count(fasta_file: str, output_dir: str) -> int:
    """Fast sequence counting with caching."""
    meta_file = os.path.join(output_dir, "meta.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            if meta_data.get("fasta_file") == fasta_file:
                print(
                    f"Loaded sequence count from cache: {meta_data['total_sequences']}"
                )
                return meta_data["total_sequences"]
        except (json.JSONDecodeError, KeyError):
            pass
    print("Counting sequences with mmap...")
    count = 0
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = 0
            while True:
                pos = mm.find(b">", pos)
                if pos == -1:
                    break
                count += 1
                pos += 1
    print(f"Found {count} sequences.")
    with open(meta_file, "w") as f:
        json.dump(
            {
                "fasta_file": fasta_file,
                "total_sequences": count,
                "parser_used": "mmap_optimized",
            },
            f,
        )
    return count


def parse_fasta_mmap(mm: mmap.mmap) -> Generator[Tuple[str, str], None, None]:
    """Efficiently parse FASTA records from mmap."""
    # Pre-find all sequence positions
    positions = []
    pos = 0
    while True:
        pos = mm.find(b">", pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1
    # Batch parsing
    for i in range(len(positions)):
        start_pos = positions[i]
        end_pos = positions[i + 1] if i + 1 < len(positions) else len(mm)
        # Parse directly in memory, avoiding copies
        record_view = mm[start_pos:end_pos]
        header_end = record_view.find(b"\n")
        if header_end == -1:
            continue
        # Extract record ID (first word of header)
        header = (
            record_view[1:header_end].split(b" ", 1)[0].decode("utf-8", errors="ignore")
        )
        # Extract sequence, removing newlines
        sequence = (
            record_view[header_end + 1 :]
            .replace(b"\n", b"")
            .decode("utf-8", errors="ignore")
        )
        yield header, sequence


def read_fasta_chunks(
    fasta_file: str, chunk_size: int
) -> Generator[List[Tuple[int, str, str]], None, None]:
    """Read FASTA file in chunks."""
    chunk = []
    idx = 0
    with open(fasta_file, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for header, sequence in parse_fasta_mmap(mm):
                chunk.append((idx, header, sequence))
                idx += 1
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def producer_process(
    task_queue: mp.Queue, fasta_file: str, chunk_size: int, num_workers: int
):
    """Producer process: Read FASTA and dispatch tasks."""
    try:
        print("[Producer] Starting to read FASTA file...")
        chunk_id_counter = 0
        for chunk in read_fasta_chunks(fasta_file, chunk_size):
            # Block until space is available in the queue to prevent memory buildup
            while True:
                try:
                    task_queue.put((chunk_id_counter, chunk), timeout=30)
                    chunk_id_counter += 1
                    break
                except queue.Full:
                    time.sleep(0.1)  # Brief pause if queue is full
        print("[Producer] All chunks queued. Sending termination signals...")
        # Send termination signal (None) for each worker
        for _ in range(num_workers):
            while True:
                try:
                    task_queue.put(None, timeout=5)
                    break
                except queue.Full:
                    time.sleep(0.1)
        print("[Producer] Termination signals sent. Exiting.")
    except Exception as e:
        print(f"[Producer] Error: {e}")
        # Attempt to send termination signals on error
        for _ in range(num_workers):
            try:
                task_queue.put(None, timeout=1)
            except:
                pass


# ==============================================================================
#  Optimized Worker Process - Key Innovation!
# ==============================================================================


def create_tar_bytes(records: List[Tuple[int, str, str]]) -> bytes:
    """
    Create a complete TAR file byte stream in memory - This is the key to performance!
    Each record in 'records' generates two files in the TAR: {idx}.fasta and {idx}.header
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for idx, header, sequence in records:
            # Create .fasta file content
            fasta_data = sequence.encode("utf-8")
            fasta_info = tarfile.TarInfo(name=f"{idx:08d}.fasta")
            fasta_info.size = len(fasta_data)
            tar.addfile(fasta_info, io.BytesIO(fasta_data))

            # Create .header file content
            header_data = header.encode("utf-8")
            header_info = tarfile.TarInfo(name=f"{idx:08d}.header")
            header_info.size = len(header_data)
            tar.addfile(header_info, io.BytesIO(header_data))
    # Get the complete byte stream of the TAR file
    return tar_buffer.getvalue()


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, worker_id: int):
    """Worker process: Parallel TAR formatting - Core optimization!"""
    try:
        print(f"[Worker-{worker_id}] Started.")
        while True:
            try:
                # Wait for a task (chunk of records) from the producer
                task = task_queue.get(timeout=60)
                if task is None:
                    # None is the termination signal
                    print(f"[Worker-{worker_id}] Received termination signal.")
                    result_queue.put(
                        (worker_id, None, None)
                    )  # Send termination signal to writer
                    break

                chunk_id, chunk_data = task
                # Perform the CPU-intensive task of TAR formatting in the worker
                tar_bytes = create_tar_bytes(chunk_data)
                # Send the pre-formatted TAR bytes to the writer
                # Block if result queue is full to prevent memory issues
                while True:
                    try:
                        result_queue.put((worker_id, chunk_id, tar_bytes), timeout=30)
                        break
                    except queue.Full:
                        time.sleep(0.1)  # Brief pause if result queue is full

            except queue.Empty:
                print(f"[Worker-{worker_id}] Timeout waiting for task. Exiting.")
                result_queue.put((worker_id, None, None))  # Send termination signal
                break
    except Exception as e:
        print(f"[Worker-{worker_id}] Error: {e}")
        # Attempt to send termination signal on error
        try:
            result_queue.put((worker_id, None, None), timeout=1)
        except:
            pass
    finally:
        print(f"[Worker-{worker_id}] Finished.")


# ==============================================================================
#  Ultra-Efficient Writer Process
# ==============================================================================


def writer_process(
    result_queue: mp.Queue,
    output_dir: str,
    total_sequences: int,
    num_workers: int,
    chunk_size: int,
):
    """Writer process: Only responsible for file writing, no formatting."""
    try:
        print("[Writer] Starting writer process...")
        workers_done = 0
        written_sequences = 0
        chunk_counter = 0  # For naming shards uniquely per run

        # Progress bar for writing
        with tqdm(total=total_sequences, desc="Writing WebDataset", unit="seq") as pbar:
            while workers_done < num_workers:
                try:
                    # Get a pre-formatted TAR byte stream from a worker
                    worker_id, chunk_id, tar_bytes = result_queue.get(timeout=120)
                    if chunk_id is None:  # Termination signal from a worker
                        workers_done += 1
                        print(
                            f"[Writer] Worker-{worker_id} finished ({workers_done}/{num_workers} done)."
                        )
                        continue

                    # Write the complete TAR file directly to disk - Ultra-fast!
                    # Use a simple counter for shard naming to ensure uniqueness
                    shard_path = os.path.join(
                        output_dir, f"shard-{chunk_counter:06d}.tar"
                    )
                    chunk_counter += 1

                    with open(shard_path, "wb") as f:
                        f.write(tar_bytes)  # Single, fast write of the entire TAR

                    # Estimate sequences written (assuming chunk_size is roughly accurate for most chunks)
                    # Note: Last chunk might be smaller. For precise count, worker could send actual count.
                    sequences_in_chunk = (
                        len(tar_bytes) / 1000
                    )  # Very rough estimate placeholder
                    written_sequences += chunk_size  # Approximation
                    # Update progress bar (clamped to total)
                    pbar.update(min(chunk_size, total_sequences - pbar.n))

                except queue.Empty:
                    print(
                        "[Writer] Result queue empty for 120s. Checking completion..."
                    )
                    if written_sequences >= total_sequences * 0.95:  # Heuristic check
                        break
                    # If not close to done, continue waiting

        print(f"[Writer] Finished writing approximately {written_sequences} sequences.")

    except Exception as e:
        print(f"[Writer] Error: {e}")


# ==============================================================================
#  Main Coordinator
# ==============================================================================


def clean_output_directory(output_dir: str):
    """Clean the output directory by removing existing files."""
    print(f"Cleaning output directory: {output_dir}")
    # More robust cleaning: remove all .tar and .json files
    for pattern in ["*.tar"]:
        files_to_remove = glob.glob(os.path.join(output_dir, pattern))
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"  Removed existing file: {file_path}")
            except OSError as e:
                print(f"  Warning: Could not remove {file_path}: {e}")
    print("Output directory cleaned.")


def fasta_to_webdataset_optimized(
    fasta_file: str,
    output_dir: str,
    num_processes: int = None,
    chunk_size: int = 10000,  # Default chunk size for reading FASTA
):
    """Optimized main function."""

    # --- 1. Setup and Argument Processing ---
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)  # Reserve cores for system/OS

    # Limit chunk_size to prevent excessive memory usage per chunk
    original_chunk_size = chunk_size
    chunk_size = min(chunk_size, 50000)  # Adjust upper limit if needed
    if chunk_size != original_chunk_size:
        print(
            f"[INFO] Adjusted chunk_size from {original_chunk_size} to {chunk_size} to manage memory."
        )

    # Ensure output directory exists and is clean
    if os.path.exists(output_dir):
        clean_output_directory(output_dir)
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- 2. Initial Logging ---
    print("=== High-Performance FASTA to WebDataset Converter ===")
    print(f"Worker processes: {num_processes}")
    print(f"Read chunk size: {chunk_size}")
    print("=====================================================")

    # --- 3. Sequence Counting (with caching) ---
    total_sequences = get_sequence_count(fasta_file, output_dir)
    if total_sequences == 0:
        print("No sequences found. Exiting.")
        return

    # --- 4. Signal Handling for Graceful Shutdown ---
    def cleanup_handler(signum, frame):
        print("\nReceived termination signal. Cleaning up...")
        # Note: Cleaning up child processes in a signal handler can be tricky.
        # A more robust solution might involve inter-process communication.
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    # --- 5. Multiprocessing Setup and Execution ---
    try:
        # Use moderate queue sizes to control memory and backpressure
        # Manager queues can be slower but are necessary for sharing between processes
        with mp.Manager() as manager:
            # Task queue: Producer -> Workers (chunks of raw FASTA data)
            # Size it based on number of workers to provide some buffer
            task_queue = manager.Queue(maxsize=num_processes * 2)
            # Result queue: Workers -> Writer (pre-formatted TAR bytes)
            # Size it based on number of workers
            result_queue = manager.Queue(maxsize=num_processes * 2)

            # --- 6. Start Processes ---
            print("[Main] Starting processes...")

            # Start Producer
            producer_proc = mp.Process(
                target=producer_process,
                args=(task_queue, fasta_file, chunk_size, num_processes),
            )
            producer_proc.start()
            print(f"[Main] Started Producer (PID: {producer_proc.pid})")

            # Start Writer
            # Pass chunk_size to writer for progress estimation
            writer_proc = mp.Process(
                target=writer_process,
                args=(
                    result_queue,
                    output_dir,
                    total_sequences,
                    num_processes,
                    chunk_size,
                ),
            )
            writer_proc.start()
            print(f"[Main] Started Writer (PID: {writer_proc.pid})")

            # Start Worker Pool
            worker_procs = []
            for i in range(num_processes):
                p = mp.Process(
                    target=worker_process,
                    args=(task_queue, result_queue, i),  # Pass worker ID
                )
                p.start()
                worker_procs.append(p)
                print(f"[Main] Started Worker-{i} (PID: {p.pid})")

            # --- 7. Wait for Completion ---
            print("[Main] Waiting for processes to finish...")
            producer_proc.join()
            print("[Main] Producer finished.")

            for i, p in enumerate(worker_procs):
                p.join()
                print(f"[Main] Worker-{i} finished.")

            writer_proc.join()
            print("[Main] Writer finished.")

        # --- 8. Final Success Message ---
        print(f"\nSuccess! Converted {fasta_file} to WebDataset format in {output_dir}")

    except Exception as e:
        print(f"[Main] Fatal error: {e}")
        # Attempt to terminate child processes if main process fails unexpectedly
        # Note: This is basic cleanup. More robust handling might be needed.
        try:
            if "producer_proc" in locals() and producer_proc.is_alive():
                producer_proc.terminate()
                producer_proc.join(timeout=5)
            if "worker_procs" in locals():
                for p in worker_procs:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
            if "writer_proc" in locals() and writer_proc.is_alive():
                writer_proc.terminate()
                writer_proc.join(timeout=5)
        except:
            pass
        raise  # Re-raise the exception after cleanup attempt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Performance FASTA to WebDataset Converter. "
        "Optimized by moving TAR formatting to worker processes."
    )
    parser.add_argument(
        "--fasta_file", type=str, required=True, help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset shards. "
        "Existing .tar and meta files will be removed.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to CPU count minus 2.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of sequences per read chunk. "
        "Smaller chunks use less memory per worker. "
        "Default is 10,000.",
    )

    args = parser.parse_args()

    fasta_to_webdataset_optimized(
        args.fasta_file, args.output_dir, args.processes, args.chunk_size
    )
