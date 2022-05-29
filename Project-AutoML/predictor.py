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

from mlflow.pyfunc import PythonModel


class PredictorWrapper(PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model_path"]
        if "autosklearn" in model_path:
            self.predictor = self.load_autosklearn_predictor(model_path)

        elif "flaml" in model_path:
            self.predictor = self.load_flaml_predictor(model_path)

        else:
            assert f"cant not load model from path {model_path}"

    def predict(self, context, model_input):
        results = self.predictor.predict(model_input)
        return {"results": results}

    def load_autosklearn_predictor(self, path):
        from automl.mod.mod_autosklearn import Predictor

        predictor = Predictor(path)
        return predictor

    def load_flaml_predictor(self, path):
        from automl.mod.mod_flaml import Predictor

        predictor = Predictor(path)
        return predictor
