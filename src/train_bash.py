import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"

from llmtuner import run_exp
import time

def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # time.sleep(1000000)
    main()
