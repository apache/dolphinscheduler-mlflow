# Licensed to Apache Software Foundation (ASF) under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Apache Software Foundation (ASF) licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pickle

import numpy as np
import pandas as pd
from flaml import AutoML

from automl.metrics import eval_classification_metrics
from automl.mod.tool import BasePredictor, Tool
from automl.params import Params


def convert_y(y):
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy().reshape(-1)
    return y


class FLAML(Tool):
    model_path = "flaml.pkl"

    conda_env = {
        "name": "MLflow-AutoML",
        "dependencies": [
            "python=3.8.2",
            "pip",
            {
                "pip": [
                    "mlflow",
                    "click==8.0.",
                    "scikit-learn==0.24.2",
                    "boto3==1.22.2",
                    "pandas>=1.0.0",
                    "setuptools<59.6.0",
                    "auto-sklearn==0.14.6",
                    "flaml==1.0.1"
                ],
            },
        ]
    }

    @staticmethod
    def train_automl(train_x, train_y, other_params=None, **kwargs):
        params = Params(param_str=other_params, **kwargs)
        automl = AutoML(**params.input_params)
        automl.predict
        train_y = convert_y(train_y)
        automl.fit(train_x, train_y)

        return automl

    @staticmethod
    def eval_automl(automl: AutoML, test_x, test_y, task="classification"):
        y_pred = automl.predict(test_x)

        test_y = convert_y(test_y)
        if task == "classification":
            metrics = eval_classification_metrics(test_y, y_pred)
        else:
            metrics = Tool.eval_automl(automl, test_x, test_y)
        return metrics

    @staticmethod
    def save_automl(automl: AutoML, save_path: str):
        automl.pickle(save_path)


class Predictor(BasePredictor):
    def load_automl(self, model_path):
        with open(model_path, "rb") as r_f:
            self.automl: AutoML = pickle.load(r_f)

    def predict(self, inputs):
        if self.automl._settings.get("task") == "classification":
            result = self.predict_classification(inputs)
        else:
            result = self.automl.predict(inputs)
        return result

    def predict_classification(self, inputs):
        pred_proba = self.automl.predict_proba(inputs)
        label_indexes = pred_proba.argmax(axis=1)
        probs = pred_proba[np.arange(pred_proba.shape[0]), label_indexes]
        labels = self.automl._label_transformer.inverse_transform(
            pd.Series(label_indexes.astype(int))
        )
        result = []
        for label, pro in zip(labels, probs):
            result.append({"label": label, "confidence": float(pro)})
        return result
