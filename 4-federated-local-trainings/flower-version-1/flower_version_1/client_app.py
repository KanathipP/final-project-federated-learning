# client_app.py

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_version_1.task import Net, load_data
from flower_version_1.task import train as train_fn
from flower_version_1.task import validate as val_fn
from flower_version_1.task import test as test_fn

import flower_version_1.log_printer as log_printer

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    print("hello from client")
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # supernode ส่ง node-config=partition-id=<0/1> เข้ามา
    partition_id = int(context.node_config["partition-id"]) + 1

    fl_training_id = msg.content["config"]["fl-training-id"]
    server_round = int(msg.content["config"]["server-round"])

    
    local_epochs = int(context.run_config["local-epochs"])
    
    lr = float(msg.content["config"]["lr"])
    
    state="load"
    text_payload="[client_app]:train | Changed state to load, Calling [task]:load_data for train_dataloader and val_dataloader ..."
    log_printer.readline(text_payload)
    log_printer.setstate(fl_training_id=fl_training_id,partition_id=partition_id,state=state)


    train_dataloader, val_dataloader, _ = load_data(
        partition_id=partition_id,
    )

    state="train"
    text_payload="[client_app]:train | Changed state to train, Calling [task]:train for training ..."
    log_printer.readline(text_payload)
    log_printer.setstate(fl_training_id=fl_training_id,partition_id=partition_id,state=state)

    train_loss, val_loss, val_accuracy = train_fn(
        net=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        fl_training_id=fl_training_id,
        server_round=server_round,
        partition_id=partition_id,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "partition_id": partition_id,
        "server_round": server_round,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "num-examples": int(len(train_dataloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_config["partition-id"]) + 1
    fl_training_id = msg.content["config"]["fl-training-id"]

    server_round = int(msg.content["config"]["server-round"])

    state="load"
    text_payload="[client_app]:evaluate | Changed state to load, Calling [task]:load_data for test_dataloader..."
    log_printer.readline(text_payload)
    log_printer.setstate(fl_training_id=fl_training_id,partition_id=partition_id,state=state)

    _, _, test_dataloader = load_data(
        partition_id=partition_id,
    )


    state="test"
    text_payload="[client_app]:evaluate | Changed state to load, Calling [task]:test for the accuracy"
    log_printer.readline(text_payload)
    log_printer.setstate(fl_training_id=fl_training_id,partition_id=partition_id,state=state)
    test_loss, test_acc = test_fn(
        net=model,
        test_dataloader=test_dataloader,
        device=device,
        fl_training_id=fl_training_id,
        server_round=server_round,
        partition_id=partition_id,
    )

    metrics = {
        "partition_id": partition_id,
        "server_round": server_round,
        "eval_loss": float(test_loss),
        "eval_acc": float(test_acc),
        "num-examples": int(len(test_dataloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
