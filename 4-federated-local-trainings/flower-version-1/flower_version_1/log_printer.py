import json

def readline(text):
    message = {
        "event": "READLINE",
        "payload": {
            "text": text,
        },
    }
    print(json.dumps(message))  # <<< ตรงนี้

def setstate(fl_training_id, partition_id, state):
    message = {
        "event": "SETSTATE",
        "payload": {
            "fl_training_id": fl_training_id,
            "partition_id": partition_id,
            "state": state,
        },
    }
    print(json.dumps(message))

def create_training_graph(fl_training_id, partition_id, server_round,
                          optimizer, learning_rate, num_epochs, batch_size):
    message = {
        "event": "CREATE_TRAINING_GRAPH",
        "payload": {
            "fl_training_id": fl_training_id,
            "partition_id": partition_id,
            "server_round": server_round,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
    }
    print(json.dumps(message))

def add_one_epoch_training_graph_point(fl_training_id, partition_id,
                                       server_round, trained_batch,
                                       current_epoch, train_loss, val_loss,
                                       accuracy, epoch_elapsed_time):
    message = {
        "event": "ADD_ONE_EPOCH_TRAINING_GRAPH_POINT",
        "payload": {
            "fl_training_id": fl_training_id,
            "partition_id": partition_id,
            "server_round": server_round,
            "trained_batch": trained_batch,
            "current_epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "epoch_training_elapsed_time": epoch_elapsed_time,
        },
    }
    print(json.dumps(message))

def create_testing_graph(fl_training_id, partition_id):
    message = {
        "event": "CREATE_TESTING_GRAPH",
        "payload": {
            "fl_training_id": fl_training_id,
            "partition_id": partition_id,
        },
    }
    print(json.dumps(message))

def add_one_server_round_testing_graph_point(fl_training_id, partition_id,
                                             server_round, criterion,
                                             batch_size, test_loss, accuracy):
    message = {
        "event": "ADD_ONE_SERVER_ROUND_TESTING_GRAPH_POINT",
        "payload": {
            "fl_training_id": fl_training_id,
            "partition_id": partition_id,
            "server_round": server_round,
            "criterion": criterion,
            "batch_size": batch_size,
            "test_loss": test_loss,
            "accuracy": accuracy,
        },
    }
    print(json.dumps(message))
