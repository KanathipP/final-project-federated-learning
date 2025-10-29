import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_version_1.task import Net, load_data
from flower_version_1.task import test as test_fn
from flower_version_1.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load data
    partition_id = context.node_config["partition-id"] + 1
    train_dataloader, val_dataloader, _ = load_data(partition_id)

    train_loss, val_loss, val_accuracy = train_fn(
        model,
        train_dataloader,
        val_dataloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num-examples": int(len(train_dataloader.dataset)),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load data
    partition_id = context.node_config["partition-id"] + 1
    _, _, test_dataloader = load_data(partition_id)

    test_loss, test_acc = test_fn(net= model, test_dataloader= test_dataloader, device= device)

    metrics = {
        "partition_id": partition_id,
        "eval_loss": test_loss,
        "eval_acc": test_acc,
        "num-examples": int(len(test_dataloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
