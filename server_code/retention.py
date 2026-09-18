"""FIFO disk retention for all temporary camera images."""

import collections
import os
import threading
import time

from app_config import (
    CORRECTED_DIR, CROP_TEST_DIR, FLAGGED_DIR, IMAGE_RETENTION_SEC, SAVE_DIR,
)

IMAGE_DIRECTORIES = (SAVE_DIR, CORRECTED_DIR, FLAGGED_DIR, CROP_TEST_DIR)
_lock = threading.Lock()


def _ordered_files(directory):
    entries = []
    try:
        names = os.listdir(directory)
    except OSError:
        return collections.deque()
    for name in names:
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path):
                entries.append((os.path.getmtime(path), path))
        except OSError:
            continue
    entries.sort(key=lambda item: item[0])
    return collections.deque(entries)


def expire_fifo_images(now=None):
    """Remove expired files strictly from the oldest end of each directory."""
    cutoff = (time.time() if now is None else now) - IMAGE_RETENTION_SEC
    removed = 0
    with _lock:
        for directory in IMAGE_DIRECTORIES:
            queue = _ordered_files(directory)
            while queue and queue[0][0] < cutoff:
                _, oldest_path = queue.popleft()
                try:
                    os.remove(oldest_path)
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    print(f"[RETENTION] Could not remove {oldest_path}: {exc}")
                    break
    return removed


def _worker():
    while True:
        expire_fifo_images()
        time.sleep(30)


def start_retention_worker():
    thread = threading.Thread(target=_worker, daemon=True, name="fifo-retention")
    thread.start()
    return thread
