import numpy as np
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
        print(optimized_model.cv_results_)
    else:
        model.fit(train_x, train_y)
    return model
