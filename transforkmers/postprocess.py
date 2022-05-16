import numpy as np
import logging
import os

log = logging.getLogger("commands")


def softmax(x: np.array) -> np.array:
    row_max = np.max(x, axis=-1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - row_max)  # subtracts each row with its max value
    row_sum = np.sum(e_x, axis=-1, keepdims=True)  # returns sum of each row and keeps same dims
    return e_x / row_sum


def write_in_csv(filename, cols, header=None, sep=",") -> None:
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    rows = 0
    with open(filename, "w") as fd:
        if header is not None:
            fd.write(f"{sep.join(header)}\n")
        for cols_tuple in zip(*cols):
            fd.write(f"{sep.join(map(str, cols_tuple))}\n")
            rows += 1
    log.info(f"{filename} created. {rows} rows written.")
