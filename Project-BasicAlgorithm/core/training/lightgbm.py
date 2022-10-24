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
