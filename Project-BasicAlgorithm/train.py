import click
import mlflow
import mlflow.sklearn

from core.data import load_data


def get_training_func(algorithm):
    if algorithm == "svm":
        from core.training.svm import train_svc as training_func

    elif algorithm == "lightgbm":
        from core.training.lightgbm import train_lightgbm as training_func

    elif algorithm == "xgboost":
        from core.training.xgboost import train_xgboost as training_func

    elif algorithm == "lr":
        from core.training.lr import train_lr as training_func

    else:
        assert f"{algorithm} not supported"

    return training_func


def create_model_version(model_name, run_id=None, auto_replace=True):
    client = mlflow.tracking.MlflowClient()
    filter_string = "name='{}'".format(model_name)
    versions = client.search_model_versions(filter_string)

    if not versions:
        client.create_registered_model(model_name)

    # TODO: 根据与上一个version对比来判断是否更换Production的模型版本
    for version in versions:
        if version.current_stage == "Production":
            client.transition_model_version_stage(
                model_name, version=version.version, stage="Archived"
            )

    uri = f"runs:/{run_id}/sklearn_model"
    mv = mlflow.register_model(uri, model_name)
    client.transition_model_version_stage(
        model_name, version=mv.version, stage="Production"
    )
    return versions


@click.command()
@click.option("--algorithm")
@click.option("--data_path")
@click.option("--label_column", default="label")
@click.option("--model_name", default=None)
@click.option("--random_state", default=0)
@click.option("--param_file", default=None)
@click.option("--params", default=None)
@click.option("--search_params", default=None)
def main(algorithm, data_path, label_column, model_name, random_state, param_file, params, search_params):

    train_x, train_y, test_x, test_y = load_data(
        data_path, label_column, random_state=random_state
    )
    training_func = get_training_func(algorithm)

    with mlflow.start_run() as run:
        model, metrics = training_func(train_x,
                                       train_y,
                                       test_x,
                                       test_y,
                                       param_file=param_file,
                                       params=params,
                                       search_params=search_params,
                                       )
        print(metrics)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="sklearn_model")

    if model_name:
        create_model_version(model_name, run_id=run.info.run_id)


if __name__ == "__main__":
    main()
