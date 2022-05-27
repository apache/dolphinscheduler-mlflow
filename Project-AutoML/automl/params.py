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

import json
import warnings
from logging import getLogger

logger = getLogger(__name__)


class Params:
    def __init__(self, param_file=None, param_str=None, **kwargs):
        input_params = self.parse_file(param_file)
        str_input_params = self.parse_param_str(param_str)
        input_params.update(str_input_params)
        input_params.update(kwargs)

        self.input_params = self.check_input_params(input_params)
        logger.info(f"params : {self}")

    def check_input_params(self, input_params):
        # check and adaptive parameter type for input_params
        for key, value in input_params.items():
            try:
                value = eval(value, {}, {})
            except Exception as e:
                value = value
                continue
            input_params[key] = value
        return input_params

    @staticmethod
    def parse_file(path):
        path = path or ""
        if not path.strip():
            return {}
        with open(path, "r") as r_f:
            params = json.load(r_f)
        return params

    @staticmethod
    def parse_param_str(param_str):
        param_str = param_str or ""
        if not param_str.strip():
            return {}
        name_value_pairs = param_str.split(";")
        pairs = []
        for name_value_pair in name_value_pairs:
            if not name_value_pair.strip():
                continue
            k_v = name_value_pair.split("=")
            if len(k_v) != 2:
                warnings.warn(f"{name_value_pair} error, will be ignore")
                continue
            key, value = name_value_pair.split("=")
            pairs.append((key.strip(), value.strip()))
        params = dict(pairs)
        return params

    def __str__(self):
        input_params_message = str(self.input_params)
        message = f"input_params: {input_params_message}"
        return message

    __repr__ = __str__
