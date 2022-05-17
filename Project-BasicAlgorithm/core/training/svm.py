import mlflow
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from core.metrics import eval_classification_metrics
from core.utils import get_onehot_encoder, train_model

from .params import SVMParams


def train_svc(
    train_x, train_y, test_x, test_y, param_file=None, params=None, search_params=None
):
    pipeline_mods = []
    mlflow.autolog()
    pipeline_mods.append(("onehot_encoder", get_onehot_encoder()))

    pipeline = Pipeline(steps=pipeline_mods)
    train_x = pipeline.fit_transform(train_x)

    params = SVMParams(
        SVC, param_file=param_file, param_str=params, search_params=search_params
    )

    model = train_model(SVC, params, train_x, train_y)

    pipeline.steps.append(("model", model))

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
