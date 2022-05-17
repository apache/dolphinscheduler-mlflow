from sklearn.metrics import classification_report


def eval_classification_metrics(y_true, y_pred):
    result: dict = classification_report(y_true, y_pred, output_dict=True)
    metrics = result["weighted avg"]
    metrics["accuracy"] = result["accuracy"]
    return metrics
