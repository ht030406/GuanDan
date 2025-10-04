# storage/replay_buffer.py
"""
Simple thread-safe replay/trajectory buffer suitable for on-policy PPO-style training.
"""

from typing import Dict, List, Any, Iterable, Iterator, Tuple
import threading
import numpy as np
import os
import json


class ReplayBuffer:
    def __init__(self):
        # store transitions as list of dicts
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def add(self, transition: Dict[str, Any]):
        """
        Add a single transition to the buffer.
        transition: dict with consistent keys across calls, e.g.
            {'obs': np.array(...), 'action': int, 'reward': float, 'done': bool, 'logp': float, 'value': float, 'mask': np.array(...)}
        """
        with self._lock:
            self._buffer.append(transition)

    def extend(self, transitions: Iterable[Dict[str, Any]]):
        """Add multiple transitions (e.g. one trajectory) atomically."""
        with self._lock:
            self._buffer.extend(transitions)

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def pop_all(self) -> List[Dict[str, Any]]:
        """Atomically retrieve and clear all stored transitions (returns list of dicts)."""
        with self._lock:
            data = self._buffer
            self._buffer = []
        return data

    def get_all(self, clear: bool = False) -> Dict[str, np.ndarray]:
        """
        Return all data as a dict of numpy arrays keyed by transition fields.
        If clear=True, the internal buffer is cleared atomically.
        Example return:
            {'obs': np.array([...]), 'action': np.array([...]), 'reward': np.array([...]), ...}
        Note: this assumes each transition has the same set of keys and compatible shapes for concatenation.
        """
        with self._lock:
            buf = list(self._buffer)
            if clear:
                self._buffer = []

        if not buf:
            return {}

        # gather keys and build arrays
        keys = list(buf[0].keys())
        out: Dict[str, List[Any]] = {k: [] for k in keys}
        for item in buf:
            for k in keys:
                out[k].append(item.get(k, None))

        # convert lists to numpy arrays where sensible
        result: Dict[str, np.ndarray] = {}
        for k, vlist in out.items():
            # try to convert to a numpy array; leave object dtype if inconsistent
            try:
                arr = np.array(vlist)
            except Exception:
                arr = np.array(vlist, dtype=object)
            result[k] = arr
        return result

    def minibatch_generator(self, batch_size: int, epochs: int = 1, shuffle: bool = True) -> Iterator[Dict[str, np.ndarray]]:
        """
        Yield minibatches for a number of epochs from the current buffer snapshot (without clearing).
        This is convenient for PPO where you iterate multiple epochs over the same data.

        - batch_size: number of samples per minibatch
        - epochs: number of passes over full dataset
        - shuffle: whether to shuffle before each epoch

        Yields:
            dict-of-arrays with same keys as transitions and first dim == batch_size (last batch may be smaller)
        """
        snapshot = self.get_all(clear=False)
        if not snapshot:
            return

        # determine dataset size
        # find first key to get length
        first_key = next(iter(snapshot.keys()))
        N = snapshot[first_key].shape[0]
        idxs = np.arange(N)

        for ep in range(epochs):
            if shuffle:
                np.random.shuffle(idxs)
            # iterate minibatches
            for start in range(0, N, batch_size):
                batch_idx = idxs[start:start + batch_size]
                batch = {}
                for k, arr in snapshot.items():
                    # fancy indexing; if object dtype, use list comprehension fallback
                    try:
                        batch[k] = arr[batch_idx]
                    except Exception:
                        batch[k] = np.array([arr[i] for i in batch_idx])
                yield batch

    def sample(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Randomly sample num_samples transitions (without replacement).
        Returns dict-of-arrays.
        """
        snapshot = self.get_all(clear=False)
        if not snapshot:
            return {}

        first_key = next(iter(snapshot.keys()))
        N = snapshot[first_key].shape[0]
        if num_samples >= N:
            return snapshot

        idxs = np.random.choice(N, size=num_samples, replace=False)
        sampled: Dict[str, np.ndarray] = {}
        for k, arr in snapshot.items():
            try:
                sampled[k] = arr[idxs]
            except Exception:
                sampled[k] = np.array([arr[i] for i in idxs])
        return sampled

    # ---------------- persistence helpers ----------------
    def save(self, path: str):
        """
        Save buffer to disk as numpy npz (each field saved as array).
        If buffer is empty, still writes an empty file with metadata.
        """
        data = self.get_all(clear=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # convert non-numeric arrays to object np.savez by serializing JSON if necessary
        meta = {}
        np_save_dict = {}
        for k, arr in data.items():
            # attempt to save numeric arrays directly
            try:
                np_save_dict[k] = np.asarray(arr)
            except Exception:
                # fallback: serialize to JSON strings per element
                np_save_dict[k] = np.asarray([json.dumps(x) for x in arr], dtype=object)
                meta[f"{k}_serialized"] = True
        np.savez_compressed(path, **np_save_dict)
        # optionally write meta file
        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, path: str):
        """
        Load buffer from a .npz file previously written by save().
        Loaded content replaces current buffer.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            # reconstruct list of transitions
            arrays = {k: data[k] for k in data.files}
        # find length from first array
        first_key = next(iter(arrays.keys()))
        N = arrays[first_key].shape[0]
        rebuilt = []
        for i in range(N):
            item = {}
            for k, arr in arrays.items():
                val = arr[i]
                # if val is bytes/object, allow it through
                item[k] = val.tolist() if hasattr(val, "tolist") else val
            rebuilt.append(item)
        with self._lock:
            self._buffer = rebuilt
