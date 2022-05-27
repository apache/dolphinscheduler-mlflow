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

class Tool:
    model_path = None

    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.8.2",
            {
                "pip": [
                    "mlflow",
                    "scikit-learn==0.24.2",
                    "boto3==1.22.2",
                    "pandas==1.3.5",
                    "setuptools<59.6.0",
                ],
            },
        ],
        "name": "mlflow-env",
    }

    @staticmethod
    def train_automl(train_x, train_y, other_params=None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def eval_automl(automl, test_x, test_y):
        score = automl.score(test_x, test_y)
        return {"score": score}

    @staticmethod
    def save_automl(automl, save_path: str):
        raise NotImplementedError


class BasePredictor:
    def __init__(self, automl_path=None):
        self.load_automl(automl_path)

    def predict(self, inputs):
        return {}

    def load_automl(self, path):
        ...
