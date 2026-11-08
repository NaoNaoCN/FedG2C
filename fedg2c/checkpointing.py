"""Checkpoint cadence uses completed (one-based) communication rounds."""

DEFAULT_CHECKPOINT_INTERVAL = 20


def should_save_checkpoint(round_id, rounds, interval=DEFAULT_CHECKPOINT_INTERVAL):
    if any(type(value) is not int for value in (round_id, rounds, interval)):
        raise ValueError('Checkpoint round counts must be integers')
    if rounds < 1 or interval < 1 or not 0 <= round_id < rounds:
        raise ValueError('Require positive rounds/interval and a valid round_id')
    return (round_id + 1) % interval == 0 or round_id == rounds - 1
