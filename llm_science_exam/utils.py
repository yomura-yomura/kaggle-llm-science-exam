import contextlib
import datetime as dt


@contextlib.contextmanager
def timer(desc: str):
    start_time = dt.datetime.now()
    print(f"* Start: {desc}")
    yield
    print(f"  Finished by {dt.datetime.now() - start_time}", end="\n\n")
