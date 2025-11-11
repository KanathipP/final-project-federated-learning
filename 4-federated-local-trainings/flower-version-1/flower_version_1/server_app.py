# server_app.py

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower_version_1.task import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config["fraction-train"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    train_config = ConfigRecord(
        {
            "lr": lr,
            "local-epochs": local_epochs,
        }
    )

    strategy = FedAvg(fraction_train=fraction_train)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
