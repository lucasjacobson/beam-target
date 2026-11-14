"""Sharing the work within a case between threads (and, through tokens, between processes)."""

# Threads for the work within a case, and an optional semaphore shared between processes that
# limits how many such tasks run at once across them (see set_parallel)
_PARALLEL = {"threads": 1, "tokens": None}


def set_parallel(threads=1, tokens=None):
    """Use up to threads threads (in this process) for the work within each case that can be
    shared out: the spectra of evaluated continua at their beam energies (compiled code that
    runs in parallel) and blocks of output rows of their two-body levels.

    tokens, if given, is a semaphore shared between processes (multiprocessing.Semaphore)
    with one token per core: each task holds one while it runs, so that processes running
    several cases at once share the cores between them, and a case still running when others
    finish takes over their cores. Results do not depend on these settings."""
    _PARALLEL["threads"] = max(1, int(threads))
    _PARALLEL["tokens"] = tokens


def threads_set():
    """The number of threads set with set_parallel."""
    return _PARALLEL["threads"]


def in_parallel(function, items):
    """Return function(item) for each of items, in order, using the threads and the tokens
    set with set_parallel."""
    tokens = _PARALLEL["tokens"]

    def task(item):
        if tokens is None:
            return function(item)
        with tokens:
            return function(item)

    threads = min(_PARALLEL["threads"], len(items))
    if threads <= 1:
        return [task(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as executor:
        return list(executor.map(task, items))
