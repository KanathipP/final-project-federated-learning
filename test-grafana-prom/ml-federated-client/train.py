
import logging
import random
import math
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s client_id=%(client_id)s msg=%(message)s",
)

CLIENT_ID = os.getenv("CLIENT_ID", "client-unknown")
ROUNDS = int(os.getenv("ROUNDS", "10"))
SLEEP_BETWEEN_RUNS = int(os.getenv("SLEEP_BETWEEN_RUNS", "10"))


class ClientAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        ctx = {"client_id": CLIENT_ID}
        if "extra" in kwargs:
            ctx.update(kwargs["extra"])
        kwargs["extra"] = ctx
        return msg, kwargs


logger = ClientAdapter(logging.getLogger(__name__), {})


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def run_one_training(rounds: int) -> None:
    # dataset ง่าย ๆ: x ~ U(-3,3), label = 1 ถ้า x > 0
    random.seed(42)
    n_samples = 200
    xs = [random.uniform(-3.0, 3.0) for _ in range(n_samples)]
    ys = [1 if x > 0 else 0 for x in xs]

    w = 0.0
    b = 0.0
    lr = 0.1

    for rnd in range(1, rounds + 1):
        dw = 0.0
        db = 0.0
        loss = 0.0

        for x, y in zip(xs, ys):
            z = w * x + b
            p = sigmoid(z)
            loss += -(y * math.log(p + 1e-8) + (1 - y) * math.log(1 - p + 1e-8))
            dw += (p - y) * x
            db += (p - y)

        loss /= n_samples
        dw /= n_samples
        db /= n_samples

        w -= lr * dw
        b -= lr * db

        logger.info(
            "fl_round=%d loss=%.4f w=%.3f b=%.3f",
            rnd,
            loss,
            w,
            b,
        )

        time.sleep(1.0)

    logger.info(
        "training_run_completed rounds=%d final_loss=%.4f w=%.3f b=%.3f",
        rounds,
        loss,
        w,
        b,
    )


def main():
    logger.info(
        "client_start federation=true rounds=%d sleep_between_runs=%d",
        ROUNDS,
        SLEEP_BETWEEN_RUNS,
    )
    while True:
        logger.info("training_run_started")
        run_one_training(ROUNDS)
        logger.info("sleeping_before_next_run seconds=%d", SLEEP_BETWEEN_RUNS)
        time.sleep(SLEEP_BETWEEN_RUNS)


if __name__ == "__main__":
    main()
