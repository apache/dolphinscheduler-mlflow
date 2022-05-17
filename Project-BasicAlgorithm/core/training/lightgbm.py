import mlflow
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from core.metrics import eval_classification_metrics
from core.utils import get_oridinal_encoder, train_model

from .params import LightGBMParams


def train_lightgbm(
    train_x, train_y, test_x, test_y, param_file=None, params=None, search_params=None
):
    pipeline_mods = []
    mlflow.autolog()

    pipeline_mods.append(("oridinal_encoder", get_oridinal_encoder()))
    pipeline = Pipeline(steps=pipeline_mods)
    train_x = pipeline.fit_transform(train_x)

    params = LightGBMParams(
        LGBMClassifier,
        param_file=param_file,
        param_str=params,
        search_params=search_params,
    )

    model = train_model(LGBMClassifier, params, train_x, train_y)

    pipeline.steps.append(("model", model))

    y_pred = pipeline.predict(test_x)

    metrics = eval_classification_metrics(test_y, y_pred)
    return pipeline, metrics
