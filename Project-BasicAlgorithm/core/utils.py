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

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def get_onehot_encoder(sparse=False, handle_unknown="ignore"):
    return OneHotEncoder(sparse=sparse, handle_unknown=handle_unknown)


def get_oridinal_encoder(unknown_value=np.nan, handle_unknown="use_encoded_value"):
    return OrdinalEncoder(unknown_value=unknown_value, handle_unknown=handle_unknown)


def train_model(model_cls, params, train_x, train_y):
    """
    train model directly, or train model with searching params
    """

    model = model_cls(**params.input_params)

    if params.search_params:
        optimized_model = GridSearchCV(estimator=model, param_grid=params.search_params)
        optimized_model.fit(train_x, train_y)
        model = optimized_model.best_estimator_
        params = optimized_model.cv_results_['params']
        mean_test_score = optimized_model.cv_results_['mean_test_score']
        for param, score in zip(params, mean_test_score):
            print(param, score)
    else:
        model.fit(train_x, train_y)
    return model
