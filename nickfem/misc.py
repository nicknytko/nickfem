import time

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t_start = time.monotonic()

    def __exit__(self, type, value, traceback):
        self.t_end = time.monotonic()
        print(f'{self.label}: {self.t_end - self.t_start:.3f}')
