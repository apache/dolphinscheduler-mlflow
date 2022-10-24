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

import logging

import click
import mlflow
import mlflow.sklearn

from automl.data import load_data


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

ARTIFACT_TAG = "artifact"


def get_tool(tool_name):
    tool_name = tool_name.lower()
    assert tool_name in {"autosklearn", "flaml"}
    if tool_name.lower() == "autosklearn":
        from automl.mod.mod_autosklearn import AutoSklearn as Tool

    elif tool_name.lower() == "flaml":
        from automl.mod.mod_flaml import FLAML as Tool

    else:
        raise Exception(f"Does not support {tool_name}")
    return Tool


def create_model_version(model_name, key_metrics=None, run_id=None, auto_replace=True):
    client = mlflow.tracking.MlflowClient()
    filter_string = "name='{}'".format(model_name)
    versions = client.search_model_versions(filter_string)

    if not versions:
        client.create_registered_model(model_name)

    for version in versions:
        if version.current_stage == "Production":
            client.transition_model_version_stage(
                model_name, version=version.version, stage="Archived"
            )

    if run_id:
        uri = f"runs:/{run_id}/{ARTIFACT_TAG}"
        mv = mlflow.register_model(uri, model_name)

        if not key_metrics:
            client.transition_model_version_stage(
                model_name, version=mv.version, stage="Production"
            )
            logger.info("register last version to Production")

    if key_metrics:
        version2metrics = []
        versions = client.search_model_versions(filter_string)
        for version in versions:
            metrics = client.get_run(version.run_id).data.metrics[key_metrics]
            version2metrics.append((version.version, metrics))

        logger.info(f"version2metrics({key_metrics}): {version2metrics}")

        best_version = max(version2metrics, key=lambda x: x[1])[0]

        logger.info("register version: %s to Production", best_version)
        client.transition_model_version_stage(
            model_name, version=best_version, stage="Production"
        )

    return versions


@click.command()
@click.option("--tool")
@click.option("--data_path")
@click.option("--label_column", default="label")
@click.option("--model_name", default=None)
@click.option("--random_state", default=0)
@click.option("--params", default=None)
def main(
    tool,
    data_path,
    label_column,
    model_name,
    random_state,
    params,
):

    Tool = get_tool(tool)

    train_x, train_y, test_x, test_y = load_data(
        data_path, label_column, random_state=random_state
    )

    automl = Tool.train_automl(train_x, train_y, other_params=params)

    metrics = Tool.eval_automl(automl, test_x, test_y)
    logger.info(f"metrics: {metrics}")
    mlflow.log_metrics(metrics)

    Tool.save_automl(automl, Tool.model_path)

    from predictor import PredictorWrapper

    artifacts = {"model_path": Tool.model_path}

    model_info = mlflow.pyfunc.log_model(
        artifact_path=ARTIFACT_TAG,
        python_model=PredictorWrapper(),
        artifacts=artifacts,
        conda_env=Tool.conda_env,
        code_path=["automl/", "predictor.py"],
    )
    if model_name:
        create_model_version(
            model_name, key_metrics='f1-score', run_id=model_info.run_id)


if __name__ == "__main__":
    main()
