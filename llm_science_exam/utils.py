import contextlib
import ctypes
import datetime as dt
import gc

import torch


@contextlib.contextmanager
def timer(desc: str):
    start_time = dt.datetime.now()
    print(f"* Start: {desc}")
    yield
    print(f"  Finished by {dt.datetime.now() - start_time}", end="\n\n")


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
