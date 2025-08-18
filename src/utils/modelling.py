import uuid
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import wandb
from sklearn.model_selection import KFold
from wandb.integration.xgboost import WandbCallback
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

import config


def separate_features_and_target(
    processed_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str] = config.FEATURE_COL_NAMES,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable from the processed DataFrame.

    :param processed_df: The processed DataFrame containing features and target.
    :param target_col: The target column for the experiment.
    :return: A tuple containing the features DataFrame and the target Series.
    """
    feature_cols = [
        col
        for col in feature_cols
        # remove target
        if col != target_col
        # remove excluded features for this target
        and col not in config.TARGET_EXCLUDE_COLS.get(target_col, [])
    ]

    # filter dataset
    X = processed_df[feature_cols]
    y = processed_df[target_col]
    return X, y


def init_wandb(
    run_name_base: str,
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
) -> wandb.Run:
    """
    Initialize Weights & Biases for tracking experiments.

    :param run_name_base: The base name for the W&B run.
    :param target_col: The target column for the experiment.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # initialize Weights & Biases run
    short_guid = uuid.uuid4().hex[:8]
    run_name = f"{run_name_base}_{short_guid}"
    return wandb.init(name=run_name, mode=wandb_mode)


def train_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    wandb_run: wandb.Run = wandb.Run(wandb.Settings(mode="disabled")),
    model_output_dir: Path = config.MODEL_OUTPUT_DIR,
):
    """
    Train an XGBoost model on the processed dataset for a specific target column.

    :param target_col: The target column to predict.
    :param output_model_dir: The directory to save the trained model.
    """
    # get random state from W&B config
    random_state = wandb.config.get("random_state", None)

    # Cross-validation setup
    kfold = KFold(**config.CV_PARAMS, random_state=random_state)
    cv_scores = []
    cv_models = []

    # perform cross-validation
    for train_idx, val_idx in kfold.split(X, y):
        # create train and val sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # add callback to log metrics to W&B
        callbacks = [
            EarlyStopping(**config.XGB_EARLY_STOPPING_PARAMS),
            WandbCallback(),
        ]

        # init the model
        model = XGBRegressor(**wandb.config.as_dict(), callbacks=callbacks)

        # train the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # save the best score and model
        cv_scores.append(model.best_score)
        cv_models.append(model)

    # find the lowest val error from the cross-validation
    best_cv_score = min(cv_scores)
    best_cv_model = cv_models[cv_scores.index(best_cv_score)]

    # log the best score across all cv folds to W&B for the sweep
    wandb_run.log({"val-rmse": best_cv_score})

    # save the model to the output directory
    model_path = model_output_dir / f"{wandb_run.name}_model.json"
    best_cv_model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # upload the model to W&B
    wandb_run.save(str(model_path), base_path=model_output_dir)

    # finish the W&B run
    wandb_run.finish()


def wandb_sweep_func(
    X: pd.DataFrame,
    y: pd.Series,
    run_base_name: str = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
    model_output_dir: Path = config.MODEL_OUTPUT_DIR,
):
    """
    Wrapper function for sweeping hyperparameters using Weights & Biases.

    :param processed_df: The processed DataFrame containing features and target.
    :param feature_cols: The feature columns for the experiment.
    :param run_base_name: The base name for the W&B run.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # init W&B run
    run = init_wandb(
        run_name_base=run_base_name,
        wandb_mode=wandb_mode,
    )

    # train the model
    train_model_cv(X, y, wandb_run=run, model_output_dir=model_output_dir)


def wandb_sweep(
    processed_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str] = config.FEATURE_COL_NAMES,
    trials: int = config.WANDB_DEFAULT_SWEEP_TRIALS,
    run_base_name: str = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
    model_output_dir: Path = config.MODEL_OUTPUT_DIR,
):
    # setup hyperparameters sweep
    sweep_id = wandb.sweep(
        config.WANDB_SWEEP_CONFIG,
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
    )

    # separate features and target
    X, y = separate_features_and_target(
        processed_df, target_col, feature_cols=feature_cols
    )

    # run the sweep
    wandb.agent(
        sweep_id=sweep_id,
        function=wandb_sweep_func(
            X,
            y,
            run_base_name=run_base_name,
            wandb_mode=wandb_mode,
            model_output_dir=model_output_dir,
        ),
        count=trials,
    )

    # clean up W&B sweep
    wandb.teardown()
