from flwr.app import MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

def custom_aggregration_fn(records: list[RecordDict], weighting_metric_name: str) -> MetricRecord:
    out = MetricRecord()

    total_weight = 0.0
    sum_acc_weight = 0.0
    sum_loss_weight = 0.0
    accuracies = []
    losses = []

    for record in records:
        for metric_record in record.metric_records.values():  
            partition_id  = metric_record.get("partition_id", None)
            accuracy  = metric_record.get("eval_acc", None)
            loss = metric_record.get("eval_loss", None)
            weight = metric_record.get(weighting_metric_name, 1)

            partition_id = int(partition_id) if partition_id is not None else -1
            weight = float(weight)

            if accuracy is not None:
                accuracy = float(accuracy)
                out[f"P{partition_id}_eval_acc"] = accuracy
                accuracies.append(accuracy)
                sum_acc_weight += accuracy * weight

            if loss is not None:
                loss = float(loss)
                out[f"P{partition_id}_eval_loss"] = loss
                losses.append(loss)
                sum_loss_weight += loss * weight

            # Helpful to see sample sizes used for weighting
            out[f"P{partition_id}_{weighting_metric_name}"] = weight
            out[f"P{partition_id}_partition_id"] = partition_id

            # Count this client towards total weight iff it reported something
            if (accuracy is not None) or (loss is not None):
                total_weight += weight

    # Add global aggregates
    if total_weight > 0:
        out["eval_acc_average (this FL round)"]  = sum_acc_weight / total_weight if sum_acc_weight != 0 else 0.0
        out["eval_loss_average (this FL round)"] = sum_loss_weight / total_weight if sum_loss_weight != 0 else 0.0

    if accuracies:
        out["eval_acc_average (all FL round)"] = sum(accuracies) / len(accuracies)
    if losses:
        out["eval_loss_average (all FL round)"] = sum(losses) / len(losses)

    return out
