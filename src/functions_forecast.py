from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_features(df: pd.DataFrame, label: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Enhances a DataFrame with temporal features and optional label extraction, returning
    feature set and labels if specified.
    
    Parameters:
    - df: DataFrame with a DateTimeIndex.
    - label: Optional; name of the label column.

    Returns:
    - Tuple (features as DataFrame, labels as Series if label is provided).
    """
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype('int')

    df['prev_1_day_lag'] = df['transactions'].shift(1)
    df['prev_2_day_lag'] = df['transactions'].shift(2)
    df['prev_3_day_lag'] = df['transactions'].shift(3)
    df['prev_4_day_lag'] = df['transactions'].shift(4)
    df['prev_5_day_lag'] = df['transactions'].shift(5)
    df['prev_6_day_lag'] = df['transactions'].shift(6)
    df['prev_7_day_lag'] = df['transactions'].shift(7)

    df['prev_3_day_mean'] = df['transactions'].rolling(window = 3).mean()
    df['prev_6_day_mean'] = df['transactions'].rolling(window = 6).mean()
    df['prev_9_day_mean'] = df['transactions'].rolling(window = 9).mean()

    df['prev_3_day_std'] = df['transactions'].rolling(window = 3).std()
    df['prev_6_day_std'] = df['transactions'].rolling(window = 6).std()
    df['prev_9_day_std'] = df['transactions'].rolling(window = 9).std()

    df['prev_3_day_max'] = df['transactions'].rolling(window = 3).max()
    df['prev_6_day_max'] = df['transactions'].rolling(window = 6).max()
    df['prev_9_day_max'] = df['transactions'].rolling(window = 9).max()

    df['prev_3_day_min'] = df['transactions'].rolling(window = 3).min()
    df['prev_6_day_min'] = df['transactions'].rolling(window = 6).min()
    df['prev_9_day_min'] = df['transactions'].rolling(window = 9).min()
    
    X = df.drop('transactions', axis=1)
    if label:
        y = df[label]
        return X, y
    return X

def prepare_train_test(filepath: str, split_date: str) -> Dict[str, Any]:
    """
    Splits time series data from a CSV into training and testing sets, applies feature 
    engineering, and returns a dictionary of features and labels for both sets.
    
    Parameters:
    - filepath: Path to the CSV file.
    - split_date: Cutoff date for splitting the data ('DD-MM-YYYY').

    Returns:
    - Dictionary with keys "X_train", "y_train", "X_test", "y_test" ,"df_train", "df_test".
    """
    #train test split the data
    df = pd.read_csv(filepath, index_col=[0], parse_dates=[0])
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    #split the train test sets into truth and predictions
    X_train, y_train = create_features(df_train, label='transactions')
    X_test, y_test = create_features(df_test, label='transactions')

    return {"X_train":X_train,
            "y_train":y_train,
            "X_test":X_test,
            "y_test":y_test,
            "df_train":df_train,
            "df_test":df_test}

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    """
    Computes MAPE between actual and predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_model_on_data(filepath: str, model_name: str, split_date: str, hyperparams: dict = None) -> Dict[str, float]:
    """
    Trains an XGBoost model and evaluates its performance on the test set.
    
    Parameters:
    - filepath: CSV file path.
    - model_name: Name of the model to run. currently supports only "XGB".
    - split_date: Cutoff date for splitting the data ('DD-MM-YYYY').
    - hyperparams: Dictionary of hyperparameters for the model.

    Returns:
    - Dictionary: MSE, MAE, and MAPE evaluation metrics.
    """
    train_test_dict = prepare_train_test(filepath, split_date)
    X_train = train_test_dict["X_train"]
    y_train = train_test_dict["y_train"]
    X_test = train_test_dict["X_test"]
    y_test = train_test_dict["y_test"]
    df_test = train_test_dict["df_test"]
    
    if model_name=="XGB":
        #default hyper parameters are set
        if hyperparams is None:
            hyperparams = {'n_estimators': 1000, 'early_stopping_rounds': 100, 'max_depth': 2, 'eta': 0.1, 'subsample': 0.6, 'random_state': 42}

        #run model and save predictions
        model = xgb.XGBRegressor(**hyperparams)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],verbose=False)
        df_test['transactions_pred'] = model.predict(X_test)

        #calculate statistics
        mse = mean_squared_error(y_true=df_test['transactions'], y_pred=df_test['transactions_pred'])
        mae= mean_absolute_error(y_true=df_test['transactions'], y_pred=df_test['transactions_pred'])
        mape = mean_absolute_percentage_error(y_true=df_test['transactions'], y_pred=df_test['transactions_pred'])
    
    return {"mse": mse,
            "mae": mae,
            "mape": mape}

def optimize_helper(learning_rate: float, subsample: float) -> float:
    """
    run the model on two params

    Parameters:
     - learning_rate: 0<learning_rate<=1
     - subsample: 0<subsample<=1

    Returns:
     - mape result: ranges from 0 to 100. lower the better.
    """
    FILE_PATH='./transactions.csv'
    MODEL='XGB'
    SPLIT_DATE='01-01-2020'
    #can use subsample, n_estimators, max_depth etc
    HYPERPARAMS={'n_estimators': 1000,
                'early_stopping_rounds': 100,
                'max_depth': 2,
                'eta': 0.1,
                'subsample': subsample,
                'learning_rate':0.01,
                'random_state': 42}
    results = run_model_on_data(filepath=FILE_PATH, model_name=MODEL, split_date=SPLIT_DATE, hyperparams=HYPERPARAMS)
    return results['mape']
    