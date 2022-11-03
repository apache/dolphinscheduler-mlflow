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
from autosklearn.classification import AutoSklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from automl.metrics import eval_classification_metrics
from automl.mod.tool import BasePredictor, Tool
from automl.params import Params


class AutoSklearn(Tool):
    model_path = "autosklearn.pkl"

    conda_env = {
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
        ],
        "name": "MLflow-AutoML",
    }

    @staticmethod
    def train_automl(train_x, train_y, other_params=None, **kwargs):
        params = Params(param_str=other_params, **kwargs)
        print(params)
        pipeline_mods = []

        pipeline_mods.append(
            (
                "oridinal_encoder",
                OrdinalEncoder(
                    unknown_value=np.nan, handle_unknown="use_encoded_value"
                ),
            )
        )
        pipeline = Pipeline(steps=pipeline_mods)
        feat_type = [
            "Categorical" if x.name in {"object", "category"} else "Numerical"
            for x in train_x.dtypes
        ]
        train_x = pipeline.fit_transform(train_x)
        classifier = AutoSklearnClassifier(**params.input_params)
        classifier.fit(train_x, train_y, feat_type=feat_type)

        pipeline.steps.append(("classifier", classifier))
        return pipeline

    @staticmethod
    def eval(pipeline: Pipeline, test_x, test_y, task="classification"):
        oridinal_encoder = pipeline.steps[0][1]
        classifier = pipeline.steps[1][1]
        test_x = oridinal_encoder.transform(test_x)
        y_pred = classifier.predict(test_x)
        if task == "classification":
            metrics = eval_classification_metrics(test_y, y_pred)
        else:
            metrics = super().eval_automl(automl, test_x, test_y)

        return metrics

    @staticmethod
    def save_automl(classifier: AutoSklearnClassifier, save_path: str):
        with open(save_path, "wb") as w_f:
            pickle.dump(classifier, w_f)


class Predictor(BasePredictor):
    def load_automl(self, model_path):
        with open(model_path, "rb") as r_f:
            self.pipeline: AutoSklearnClassifier = pickle.load(r_f)
        self.oridinal_encoder = self.pipeline.steps[0][1]
        self.automl = self.pipeline.steps[1][1]

    def predict(self, inputs):
        if isinstance(self.automl, AutoSklearnClassifier):
            result = self.predict_classification(inputs)
        else:
            result = self.automl.predict(inputs)
        return result

    def predict_classification(self, inputs):

        inputs = self.oridinal_encoder.transform(inputs)

        pred_proba = self.classifier.predict_proba(inputs)
        label_indexes = pred_proba.argmax(axis=1)
        probs = pred_proba[np.arange(pred_proba.shape[0]), label_indexes]
        labels = (
            self.classifier.automl_.InputValidator.target_validator.inverse_transform(
                label_indexes
            )
        )
        result = []
        for label, pro in zip(labels, probs):
            result.append({"label": label, "confidence": float(pro)})
        return result
