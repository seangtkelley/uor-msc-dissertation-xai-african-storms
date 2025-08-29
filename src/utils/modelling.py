#!/usr/bin/env python

"""modelling.py: Helper functions for training models using Weights & Biases"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import json
import tempfile
import uuid
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Literal, Optional

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from wandb.integration.xgboost import WandbCallback
from wandb.sdk.wandb_run import Run
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

import config
import wandb


def get_features_and_target(
    processed_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str] = config.ALL_FEATURE_COLS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable from the processed DataFrame.

    :param processed_df: The processed DataFrame containing features and target.
    :param target_col: The target column for the experiment.
    :param feature_cols: Iterable of feature column names to use (default=config.FEATURE_COLS). The target column and any columns excluded for this target will be removed from this list.
    :return: A tuple containing the features DataFrame and the target Series.
    """
    feature_cols = [
        col
        for col in feature_cols
        # remove target
        if col != target_col
        # remove excluded features for this target
        and col not in config.TARGET_EXCLUDE_COLS_MAP.get(target_col, [])
        # remove all target excluded features
        and col not in config.ALL_TARGET_EXCLUDE_COLS
    ]

    # filter dataset
    X = processed_df[feature_cols]
    y = processed_df[target_col]
    return X, y


def init_wandb(
    run_name_base: str,
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
) -> Run:
    """
    Initialize Weights & Biases for tracking experiments.

    :param run_name_base: The base name for the W&B run.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # initialize Weights & Biases run
    short_guid = uuid.uuid4().hex[:8]
    run_name = f"{run_name_base}_{short_guid}"
    return wandb.init(name=run_name, mode=wandb_mode)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    wandb_run: Optional[Run] = None,
):
    """
    Train an XGBoost model using train/val/test splits.

    :param X: Features DataFrame.
    :param y: Target Series.
    :param wandb_run: Weights & Biases run object.
    """
    # if wandb_run is None, provide a no-op run
    if wandb_run is None:
        wandb_run = wandb.init(name="test", mode="disabled")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # train/val split
    if config.VAL_SIZE <= 0 or config.VAL_SIZE >= 1:
        raise ValueError(
            "val_size must be between 0 and 1 (exclusive) for early stopping"
        )

    # adjust val size for train size
    # source: https://datascience.stackexchange.com/a/15136
    val_size = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=config.RANDOM_STATE
    )

    # add callback to log metrics to Weights & Biases
    callbacks = [
        EarlyStopping(**config.XGB_EARLY_STOPPING_PARAMS),
        WandbCallback(log_model=True),
    ]

    # init the model
    model = XGBRegressor(**config.XGB_HYPERPARAMS, callbacks=callbacks)

    # train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    print(f"test-rmse: {test_rmse}")

    # log evaluation metric to Weights & Biases
    wandb_run.log({"test-rmse": test_rmse})

    # save the model to temp dir
    temp_dir = Path(tempfile.gettempdir())
    model_path = temp_dir / f"{wandb_run.name}_model.json"
    model.save_model(model_path)

    # upload the model to W&B
    wandb_run.save(str(model_path), base_path=temp_dir)

    # update run config with test dataset index
    wandb_run.config.update({"test_dataset_index": y_test.index.tolist()})


def train_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    wandb_run: Optional[Run] = None,
):
    """
    Train an XGBoost model using cross-validation.

    :param X: Features DataFrame.
    :param y: Target Series.
    :param wandb_run: Weights & Biases run object.
    """
    # if wandb_run is None, provide a no-op run
    if wandb_run is None:
        wandb_run = wandb.init(name="test", mode="disabled")

    # train/test split
    # test set is ignored here as it's not needed for cross-validation
    # but with the random state, we can ensure reproducibility for later
    # testing of the best model without data leakage
    X_train, _, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Cross-validation setup
    kfold = KFold(**config.CV_PARAMS)
    cv_scores = []
    cv_models = []

    # perform cross-validation
    for train_idx, val_idx in kfold.split(X_train, y_train):
        # create train and val sets
        X_train_fold, X_val_fold = (
            X_train.iloc[train_idx],
            X_train.iloc[val_idx],
        )
        y_train_fold, y_val_fold = (
            y_train.iloc[train_idx],
            y_train.iloc[val_idx],
        )

        # add callback to log metrics to W&B
        callbacks = [
            EarlyStopping(**config.XGB_EARLY_STOPPING_PARAMS),
            WandbCallback(),
        ]

        # init the model
        model = XGBRegressor(**wandb.config.as_dict(), callbacks=callbacks)

        # train the model
        model.fit(
            X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)]
        )

        # save the best score and model
        cv_scores.append(model.best_score)
        cv_models.append(model)

    # find the lowest val error from the cross-validation
    best_cv_score = min(cv_scores)
    best_cv_model = cv_models[cv_scores.index(best_cv_score)]

    # log the best score across all cv folds to W&B for the sweep
    wandb_run.log({"val-rmse": best_cv_score})

    # save the model to temp dir
    temp_dir = Path(tempfile.gettempdir())
    model_path = temp_dir / f"{wandb_run.name}_model.json"
    best_cv_model.save_model(model_path)

    # upload the model to W&B
    wandb_run.save(str(model_path), base_path=temp_dir)

    # update run config with test dataset index
    wandb_run.config.update({"test_dataset_index": y_test.index.tolist()})


def wandb_sweep_func(
    X: pd.DataFrame,
    y: pd.Series,
    run_base_name: str = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
):
    """
    Wrapper function for sweeping hyperparameters using Weights & Biases.

    :param X: Features DataFrame.
    :param y: Target Series.
    :param run_base_name: The base name for the W&B run.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # init W&B run
    wandb_run = init_wandb(
        run_name_base=run_base_name,
        wandb_mode=wandb_mode,
    )

    # train the model
    train_model_cv(X, y, wandb_run=wandb_run)

    # finish the W&B run
    wandb_run.finish()


def wandb_sweep(
    processed_df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    run_base_name: str,
    trials: Optional[int],
    prior_run_ids: Optional[list[str]] = None,
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled",
):
    """
    Perform a hyperparameter sweep using Weights & Biases.

    :param processed_df: The processed DataFrame containing features and target.
    :param target_col: The target column for the model.
    :param feature_cols: The feature columns for the model.
    :param run_base_name: The base name for the W&B run.
    :param trials: The number of trials for the sweep.
    :param prior_run_ids: The IDs of prior runs to use for the sweep.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # init W&B sweep
    sweep_id = wandb.sweep(
        config.WANDB_SWEEP_CONFIG,
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        prior_runs=prior_run_ids,
    )

    # separate features and target
    X, y = get_features_and_target(
        processed_df, target_col, feature_cols=feature_cols
    )

    # run the sweep
    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: wandb_sweep_func(
            X,
            y,
            run_base_name=run_base_name,
            wandb_mode=wandb_mode,
        ),
        count=trials,
    )

    # clean up W&B sweep
    wandb.teardown()


def get_exp_runs(exp_name: str):
    """
    Get the W&B runs for a specific experiment.

    :param exp_name: The name of the experiment.
    :return: The W&B runs for the experiment.
    """
    wandb_api = wandb.Api()
    return wandb_api.runs(
        path=f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}",
        filters={"displayName": {"$regex": f"^{exp_name}_[a-fA-F0-9]{{8}}$"}},
    )


def run_experiment(
    exp_name: str,
    processed_df: pd.DataFrame,
    first_points_only: bool,
    target_col: str,
    feature_cols: str | list[str],
    trials: Optional[int],
    wandb_mode: Literal["online", "offline", "disabled"],
):
    """
    Run a W&B sweep for the given experiment.

    :param exp_name: The name of the experiment.
    :param processed_df: The processed DataFrame containing features and target.
    :param first_points_only: Whether to use only the first points of each storm.
    :param target_col: The target column for the model.
    :param feature_cols: The feature columns for the model.
    :param wandb_mode: The mode for W&B (online, offline, disabled).
    """
    # get prior run names if they exist
    prior_runs = get_exp_runs(exp_name)
    prior_run_ids = (
        [run.id for run in prior_runs] if len(prior_runs) > 0 else None
    )

    # if trials is None and prior runs exist, only run remaining trials
    if trials is None:
        if len(prior_runs) > 0:
            remaining_trials = config.WANDB_MAX_SWEEP_TRIALS - len(prior_runs)

            if remaining_trials > 0:
                print(
                    f"Running {remaining_trials} remaining trials for {exp_name}."
                )
                trials = remaining_trials
            else:
                print(
                    f"No remaining trials to run for {exp_name}. Found {len(prior_runs)} prior runs."
                )
                return
        else:
            trials = config.WANDB_MAX_SWEEP_TRIALS

    # get data
    df = (
        processed_df.groupby("storm_id").first()
        if first_points_only
        else processed_df
    )

    # set feature columns
    if isinstance(feature_cols, str):
        if feature_cols == "all":
            feature_cols = config.ALL_FEATURE_COLS
        elif feature_cols == "era5":
            feature_cols = config.ERA5_MET_FEATURE_COLS
        else:
            raise ValueError(f"Unknown feature column set: {feature_cols}")
    elif isinstance(feature_cols, list):
        feature_cols = feature_cols

    # run W&B sweep
    wandb_sweep(
        processed_df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        run_base_name=exp_name,
        trials=trials,
        prior_run_ids=prior_run_ids,
        wandb_mode=wandb_mode,
    )


def get_best_run_from_exp(exp_name: str) -> SimpleNamespace | Run:
    """
    Get the best run from an experiment.

    :param exp_name: The name of the W&B experiment.
    :return: The best run from the experiment.
    """
    exp_best_run_id_cache = config.WANDB_LOG_DIR / "exp_best_run_id_cache.json"
    if not exp_best_run_id_cache.exists():
        # init the cache
        with open(exp_best_run_id_cache, "w") as f:
            f.write("")

    # load run info from cache
    file_json = {}
    with open(exp_best_run_id_cache, "r") as f:
        file_json = json.load(f)

        cache_info = file_json.get(exp_name, None)

    if cache_info is not None:
        print("Loading best run info from offline cache...")
        # load offline run info
        best_run = SimpleNamespace(
            {
                "id": cache_info["id"],
                "name": cache_info["name"],
                "config": cache_info["config"],
            }
        )
    else:
        print("Loading best run info from W&B API...")
        # get all runs from the experiment
        exp_runs = get_exp_runs(exp_name)

        # find the best run (lowest validation loss)
        best_run = min(
            exp_runs, key=lambda run: run.summary.get("val-rmse", float("inf"))
        )

        # write the run info to the cache
        file_json[exp_name] = {
            "id": best_run.id,
            "name": best_run.name,
            "config": best_run.config,
        }
        with open(exp_best_run_id_cache, "w") as f:
            f.write(json.dumps(file_json))

    return best_run


def get_model_from_run(wandb_run: SimpleNamespace | Run) -> XGBRegressor:
    """
    Get the model from a W&B run. Try to load the model fully locally
    using cache and local W&B logs before contacting W&B API.

    :param wandb_run: Either a SimpleNamespace which provides `id` and `name` attributes or a W&B run object.
    :return: The model from the run.
    """
    model = XGBRegressor()
    run_dir_search = glob(str(config.WANDB_LOG_DIR / f"*{wandb_run.id}*"))
    model_filepath = None
    try:
        if len(run_dir_search) != 1:
            raise ValueError("Multiple run directories found.")

        print("Loading model from local run directory...")

        # load the model
        model_filepath = (
            Path(run_dir_search[0]) / "files" / f"{wandb_run.name}_model.json"
        )

        # check the file exists
        if not model_filepath.exists():
            raise FileNotFoundError(f"Model file not found: {model_filepath}")

    except Exception as e:
        # setup dir for download
        manual_downloads_dir = config.WANDB_LOG_DIR / "manual_downloads"
        manual_downloads_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"{wandb_run.name}_model.json"
        model_filepath = Path(manual_downloads_dir) / model_filename

        if not model_filepath.exists():
            print("Downloading model from W&B...")

            # search run files for model json
            model_files = [
                f
                for f in wandb_run.files()  # type:ignore
                if f.name.endswith("_model.json")
            ]

            if len(model_files) == 0:
                raise FileNotFoundError(
                    f"No model files found for run: {wandb_run.id}"
                )
            elif len(model_files) > 1:
                print(
                    f"Multiple model files found for run: {wandb_run.id}. Using the first one."
                )

            # download model file
            download_filepath = model_files[0].download(exist_ok=True)

            # move to expected location
            Path(download_filepath.name).rename(model_filepath)

    finally:
        if model_filepath is not None:
            # load the model
            model.load_model(model_filepath)
        else:
            raise RuntimeError("Failed to locate or download the model file.")

    return model
