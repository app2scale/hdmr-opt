"""
Time Series Forecasting Module for HDMR Hyperparameter Optimization

This module provides a unified interface for various forecasting algorithms
that can be optimized using HDMR. The following algorithms are supported:

1. **XGBoost**: Gradient boosting for time series regression
2. **LightGBM**: Fast gradient boosting framework
3. **ARIMA**: AutoRegressive Integrated Moving Average
4. **ETS**: Exponential Smoothing State Space Model
5. **N-BEATS**: Neural basis expansion analysis for interpretable time series

Each algorithm's hyperparameters can be optimized using HDMR to minimize
forecast error (MAPE, MAE, RMSE, etc.).

Typical Workflow:
----------------
1. Load time series data from data/transactions.csv
2. Create features (lags, rolling statistics, etc.)
3. Split into train/validation/test sets
4. Define hyperparameter space for optimization
5. Use HDMR to find optimal hyperparameters
6. Evaluate on test set

Author: APP2SCALE Team
Date: 2026-01-13
Version: 2.1.0
"""

from typing import Dict, Tuple, Optional, Any, Callable, List
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Machine Learning Libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

# Statistical Forecasting Libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def mean_absolute_percentage_error(
    y_true: NDArray[np.float64], 
    y_pred: NDArray[np.float64]
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (100/n) * Î£ |yáµ¢ - Å·áµ¢| / |yáµ¢|
    
    Parameters
    ----------
    y_true : NDArray[np.float64]
        Actual values.
    y_pred : NDArray[np.float64]
        Predicted values.
    
    Returns
    -------
    float
        MAPE percentage (0-100+).
    
    Notes
    -----
    - MAPE is undefined when y_true contains zeros
    - Small values in y_true can lead to very large MAPE
    - Not symmetric: MAPE(y, Å·) â‰  MAPE(Å·, y)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        raise ValueError("MAPE undefined: y_true contains only zeros")
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(
    y_true: NDArray[np.float64], 
    y_pred: NDArray[np.float64]
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Î£ |yáµ¢ - Å·áµ¢| / (|yáµ¢| + |Å·áµ¢|)
    
    Parameters
    ----------
    y_true : NDArray[np.float64]
        Actual values.
    y_pred : NDArray[np.float64]
        Predicted values.
    
    Returns
    -------
    float
        SMAPE percentage (0-100).
    
    Notes
    -----
    - Symmetric version of MAPE
    - Bounded between 0 and 100
    - Handles zeros better than MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    
    # Avoid division by zero
    mask = denominator != 0
    if not mask.any():
        return 0.0
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mean_absolute_scaled_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_train: NDArray[np.float64],
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE = MAE(test) / MAE(naive_forecast_on_train)
    
    Parameters
    ----------
    y_true : NDArray[np.float64]
        Actual test values.
    y_pred : NDArray[np.float64]
        Predicted test values.
    y_train : NDArray[np.float64]
        Training data for scaling.
    seasonality : int, optional
        Seasonal period (default: 1 for non-seasonal).
    
    Returns
    -------
    float
        MASE value (< 1 means better than naive forecast).
    
    Notes
    -----
    - Scale-independent metric
    - MASE < 1: Better than naive seasonal forecast
    - MASE > 1: Worse than naive seasonal forecast
    """
    mae_test = mean_absolute_error(y_true, y_pred)
    
    # Naive forecast MAE on training data
    naive_forecast = y_train[:-seasonality] if seasonality > 1 else y_train[:-1]
    naive_actual = y_train[seasonality:] if seasonality > 1 else y_train[1:]
    mae_naive = mean_absolute_error(naive_actual, naive_forecast)
    
    if mae_naive == 0:
        return np.inf if mae_test > 0 else 0.0
    
    return mae_test / mae_naive


def calculate_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_train: Optional[NDArray[np.float64]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive set of forecast evaluation metrics.
    
    Parameters
    ----------
    y_true : NDArray[np.float64]
        Actual values.
    y_pred : NDArray[np.float64]
        Predicted values.
    y_train : NDArray[np.float64], optional
        Training data (required for MASE).
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'mape': Mean Absolute Percentage Error
        - 'smape': Symmetric MAPE
        - 'mase': Mean Absolute Scaled Error (if y_train provided)
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }
    
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError:
        metrics['mape'] = np.inf
    
    metrics['smape'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    if y_train is not None:
        metrics['mase'] = mean_absolute_scaled_error(y_true, y_pred, y_train)
    
    return metrics


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_time_features(
    df: pd.DataFrame,
    target_col: str = 'transactions',
    include_lags: bool = True,
    lag_periods: List[int] = [1, 2, 3, 7, 14, 28],
    include_rolling: bool = True,
    rolling_windows: List[int] = [7, 14, 28],
    include_date_features: bool = True
) -> pd.DataFrame:
    """
    Create time series features for forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date index and target column.
    target_col : str
        Name of the target column (default: 'transactions').
    include_lags : bool
        Create lag features.
    lag_periods : List[int]
        List of lag periods to create.
    include_rolling : bool
        Create rolling window statistics.
    rolling_windows : List[int]
        List of window sizes for rolling features.
    include_date_features : bool
        Extract date-based features.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional features.
    """
    df = df.copy()
    
    # Ensure date index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")
    
    # Lag features
    if include_lags:
        for lag in lag_periods:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    if include_rolling:
        for window in rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    # Date features
    if include_date_features:
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_start'] = df.index.is_year_start.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
    
    # Drop NaN values created by lags and rolling
    df = df.dropna()
    
    return df


def prepare_train_test(
    file_path: str,
    split_date: str,
    target_col: str = 'transactions',
    date_col: str = 'date',
    **feature_kwargs
) -> Dict[str, Any]:
    """
    Load data and prepare train/test splits with features.
    
    Parameters
    ----------
    file_path : str
        Path to CSV file (e.g., 'data/transactions.csv').
    split_date : str
        Date to split train/test (format: 'YYYY-MM-DD' or 'MM-DD-YYYY').
    target_col : str
        Name of target column (default: 'transactions').
    date_col : str
        Name of date column (default: 'date').
    **feature_kwargs
        Additional arguments for create_time_features().
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'X_train': Training features
        - 'y_train': Training target
        - 'X_test': Test features
        - 'y_test': Test target
        - 'feature_names': List of feature names
    """
    # Load data
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Create features
    df = create_time_features(df, target_col=target_col, **feature_kwargs)
    
    # Split by date
    split_date = pd.to_datetime(split_date)
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_cols
    }


# ============================================================================
# BASE FORECASTER CLASS
# ============================================================================

class BaseForcaster(ABC):
    """
    Abstract base class for all forecasters.
    
    All forecasters must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_hyperparameter_space(): Define optimization space
    """
    
    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """Train the forecasting model."""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """Make predictions on test data."""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return hyperparameter optimization space.
        
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Dictionary mapping parameter names to (min, max) bounds.
        """
        pass


# ============================================================================
# XGBOOST FORECASTER
# ============================================================================

class XGBoostForecaster(BaseForcaster):
    """
    XGBoost regression forecaster.
    
    Optimizable Hyperparameters:
    ---------------------------
    - learning_rate: Step size shrinkage [0.001, 0.3]
    - max_depth: Maximum tree depth [1, 10]
    - subsample: Fraction of samples per tree [0.5, 1.0]
    - colsample_bytree: Fraction of features per tree [0.5, 1.0]
    - min_child_weight: Minimum sum of instance weight [1, 10]
    - gamma: Minimum loss reduction [0, 5]
    """
    
    def __init__(
        self,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
        **kwargs
    ):
        """
        Initialize XGBoost forecaster.
        
        Parameters
        ----------
        learning_rate : float
            Learning rate (eta).
        max_depth : int
            Maximum tree depth.
        subsample : float
            Subsample ratio of training instances.
        colsample_bytree : float
            Subsample ratio of features.
        min_child_weight : int
            Minimum sum of instance weight.
        gamma : float
            Minimum loss reduction for split.
        n_estimators : int
            Number of boosting rounds.
        early_stopping_rounds : int
            Stop if no improvement for N rounds.
        random_state : int
            Random seed.
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.learning_rate = learning_rate
        self.max_depth = int(max_depth)
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = int(min_child_weight)
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model = None
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """Train XGBoost model."""
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None
        
        self.model = xgb.XGBRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            random_state=self.random_state,
            verbosity=0
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X_test)
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return XGBoost hyperparameter space for optimization."""
        return {
            'learning_rate': (0.001, 0.3),
            'max_depth': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0, 5)
        }


# ============================================================================
# LIGHTGBM FORECASTER
# ============================================================================

class LightGBMForecaster(BaseForcaster):
    """
    LightGBM regression forecaster.
    
    Optimizable Hyperparameters:
    ---------------------------
    - learning_rate: Step size [0.001, 0.3]
    - num_leaves: Maximum tree leaves [2, 256]
    - min_data_in_leaf: Minimum data in leaf [1, 100]
    - feature_fraction: Feature sampling ratio [0.5, 1.0]
    - bagging_fraction: Data sampling ratio [0.5, 1.0]
    """
    
    def __init__(
        self,
        learning_rate=0.1,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=42,
        **kwargs
    ):
        """Initialize LightGBM forecaster."""
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        self.learning_rate = learning_rate
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model = None
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """Train LightGBM model."""
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None
        
        self.model = lgb.LGBMRegressor(
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=-1
        )
        
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)] if eval_set else []
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X_test)
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return LightGBM hyperparameter space."""
        return {
            'learning_rate': (0.001, 0.3),
            'num_leaves': (2, 256),
            'min_data_in_leaf': (1, 100),
            'feature_fraction': (0.5, 1.0),
            'bagging_fraction': (0.5, 1.0)
        }


# ============================================================================
# ARIMA FORECASTER
# ============================================================================

class ARIMAForecaster(BaseForcaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    Optimizable Hyperparameters:
    ---------------------------
    - p: AR order [0, 5]
    - d: Differencing order [0, 2]
    - q: MA order [0, 5]
    """
    
    def __init__(self, p=1, d=1, q=1):
        """
        Initialize ARIMA forecaster.
        
        Parameters
        ----------
        p : int
            Order of AR term.
        d : int
            Degree of differencing.
        q : int
            Order of MA term.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not available")
        
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.model = None
    
    def fit(self, X_train, y_train, **kwargs):
        """Train ARIMA model."""
        # ARIMA typically uses only the target series
        self.model = ARIMA(y_train, order=(self.p, self.d, self.q))
        self.model = self.model.fit()
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        forecast = self.model.forecast(steps=len(X_test))
        return np.array(forecast)
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return ARIMA hyperparameter space."""
        return {
            'p': (0, 5),
            'd': (0, 2),
            'q': (0, 5)
        }


# ============================================================================
# ETS FORECASTER
# ============================================================================

class ETSForecaster(BaseForcaster):
    """
    Exponential Smoothing (ETS) forecaster.
    
    Optimizable Hyperparameters:
    ---------------------------
    - seasonal_periods: Seasonality [1, 52]
    """
    
    def __init__(self, trend='add', seasonal='add', seasonal_periods=7):
        """
        Initialize ETS forecaster.
        
        Parameters
        ----------
        trend : str, optional
            Type of trend component ('add', 'mul', or None).
        seasonal : str, optional
            Type of seasonal component ('add', 'mul', or None).
        seasonal_periods : int, optional
            Number of periods in seasonal cycle.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not available")
        
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)
        self.model = None
    
    def fit(self, X_train, y_train, **kwargs):
        """Train ETS model."""
        self.model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods
        )
        self.model = self.model.fit()
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        forecast = self.model.forecast(steps=len(X_test))
        return np.array(forecast)
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return ETS hyperparameter space."""
        return {
            'seasonal_periods': (1, 52)
        }



# ============================================================================
# DEEP LEARNING FORECASTERS
# ============================================================================

class LSTMForecaster(BaseForcaster):
    """
    LSTM (Long Short-Term Memory) forecaster.
    
    Optimizable Hyperparameters:
    ---------------------------
    - hidden_size: Number of LSTM units [16, 256]
    - num_layers: Number of stacked LSTM layers [1, 4]
    - dropout: Dropout probability [0.0, 0.5]
    - learning_rate: Optimizer learning rate [1e-4, 1e-2]
    """
    
    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,
        random_state=42
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters
        ----------
        hidden_size : int
            Number of LSTM hidden units.
        num_layers : int
            Number of stacked LSTM layers.
        dropout : float
            Dropout probability between layers.
        learning_rate : float
            Adam optimizer learning rate.
        epochs : int
            Maximum training epochs.
        batch_size : int
            Training batch size.
        patience : int
            Early stopping patience.
        random_state : int
            Random seed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_size: int):
        """Build LSTM model architecture."""
        
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                # x shape: (batch, seq_len=1, features)
                lstm_out, _ = self.lstm(x)
                # Take last output
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        return LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """Train LSTM model."""
        torch.manual_seed(self.random_state)
        
        # Convert to numpy arrays
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        self.input_size = X_train_np.shape[1]
        self.model = self._build_model(self.input_size).to(self.device)
        
        # Prepare data loaders
        X_tensor = torch.FloatTensor(X_train_np).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y_train_np).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation (if provided)
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_np = X_test.values if hasattr(X_test, 'values') else X_test
                    y_val_np = y_test.values if hasattr(y_test, 'values') else y_test
                    
                    X_val_tensor = torch.FloatTensor(X_val_np).unsqueeze(1).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val_np).reshape(-1, 1).to(self.device)
                    
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
        
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_tensor = torch.FloatTensor(X_test_np).unsqueeze(1).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().reshape(-1)
        
        return predictions
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return LSTM hyperparameter space."""
        return {
            'hidden_size': (16, 256),
            'num_layers': (1, 4),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-4, 1e-2)
        }


class GRUForecaster(BaseForcaster):
    """
    GRU (Gated Recurrent Unit) forecaster.
    
    Similar to LSTM but with simpler architecture and often faster training.
    
    Optimizable Hyperparameters:
    ---------------------------
    - hidden_size: Number of GRU units [16, 256]
    - num_layers: Number of stacked GRU layers [1, 4]
    - dropout: Dropout probability [0.0, 0.5]
    - learning_rate: Optimizer learning rate [1e-4, 1e-2]
    """
    
    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,
        random_state=42
    ):
        """Initialize GRU forecaster."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_size: int):
        """Build GRU model architecture."""
        
        class GRUNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                out = self.fc(gru_out[:, -1, :])
                return out
        
        return GRUNet(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """Train GRU model (similar to LSTM)."""
        torch.manual_seed(self.random_state)
        
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        self.input_size = X_train_np.shape[1]
        self.model = self._build_model(self.input_size).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train_np).unsqueeze(1)
        y_tensor = torch.FloatTensor(y_train_np).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_np = X_test.values if hasattr(X_test, 'values') else X_test
                    y_val_np = y_test.values if hasattr(y_test, 'values') else y_test
                    
                    X_val_tensor = torch.FloatTensor(X_val_np).unsqueeze(1).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val_np).reshape(-1, 1).to(self.device)
                    
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
        
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_tensor = torch.FloatTensor(X_test_np).unsqueeze(1).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().reshape(-1)
        
        return predictions
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return GRU hyperparameter space."""
        return {
            'hidden_size': (16, 256),
            'num_layers': (1, 4),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-4, 1e-2)
        }


class NBeatsForecaster(BaseForcaster):
    """
    N-BEATS (Neural Basis Expansion Analysis for Time Series) forecaster.
    
    A powerful deep learning architecture specifically designed for time series
    forecasting with interpretable basis expansion.
    
    Reference: Oreshkin et al. (2020) - "N-BEATS: Neural basis expansion 
    analysis for interpretable time series forecasting"
    
    Optimizable Hyperparameters:
    ---------------------------
    - stack_width: Width of each stack [128, 512]
    - num_blocks: Number of blocks per stack [1, 5]
    - num_layers: Layers per block [2, 6]
    - learning_rate: Optimizer learning rate [1e-4, 1e-2]
    """
    
    def __init__(
        self,
        stack_width=256,
        num_blocks=3,
        num_layers=4,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        patience=10,
        random_state=42
    ):
        """
        Initialize N-BEATS forecaster.
        
        Parameters
        ----------
        stack_width : int
            Hidden layer width for each stack.
        num_blocks : int
            Number of blocks in each stack.
        num_layers : int
            Number of fully connected layers per block.
        learning_rate : float
            Adam optimizer learning rate.
        epochs : int
            Maximum training epochs.
        batch_size : int
            Training batch size.
        patience : int
            Early stopping patience.
        random_state : int
            Random seed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.stack_width = int(stack_width)
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_size: int):
        """Build simplified N-BEATS model architecture."""
        
        class NBeatsBlock(nn.Module):
            """Single N-BEATS block."""
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                layers = []
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())
                
                self.fc_stack = nn.Sequential(*layers)
                self.backcast = nn.Linear(hidden_size, input_size)
                self.forecast = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                h = self.fc_stack(x)
                backcast = self.backcast(h)
                forecast = self.forecast(h)
                return backcast, forecast
        
        class NBeatsNet(nn.Module):
            """Simplified N-BEATS network."""
            def __init__(self, input_size, stack_width, num_blocks, num_layers):
                super().__init__()
                self.blocks = nn.ModuleList([
                    NBeatsBlock(input_size, stack_width, num_layers)
                    for _ in range(num_blocks)
                ])
            
            def forward(self, x):
                residual = x
                forecast = torch.zeros((x.size(0), 1), device=x.device)
                
                for block in self.blocks:
                    backcast, block_forecast = block(residual)
                    residual = residual - backcast
                    forecast = forecast + block_forecast
                
                return forecast
        
        return NBeatsNet(input_size, self.stack_width, self.num_blocks, self.num_layers)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """Train N-BEATS model."""
        torch.manual_seed(self.random_state)
        
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        self.input_size = X_train_np.shape[1]
        self.model = self._build_model(self.input_size).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train_np)
        y_tensor = torch.FloatTensor(y_train_np).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_np = X_test.values if hasattr(X_test, 'values') else X_test
                    y_val_np = y_test.values if hasattr(y_test, 'values') else y_test
                    
                    X_val_tensor = torch.FloatTensor(X_val_np).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val_np).reshape(-1, 1).to(self.device)
                    
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
        
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_tensor = torch.FloatTensor(X_test_np).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().reshape(-1)
        
        return predictions
    
    def get_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return N-BEATS hyperparameter space."""
        return {
            'stack_width': (128, 512),
            'num_blocks': (1, 5),
            'num_layers': (2, 6),
            'learning_rate': (1e-4, 1e-2)
        }



# ============================================================================
# OPTIMIZATION OBJECTIVE FUNCTIONS
# ============================================================================

def create_optimization_objective(
    model_class: type,
    data_dict: Dict[str, Any],
    metric: str = 'mape',
    fixed_params: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create an objective function for HDMR optimization.
    
    Parameters
    ----------
    model_class : type
        Forecaster class (e.g., XGBoostForecaster).
    data_dict : Dict[str, Any]
        Dictionary from prepare_train_test().
    metric : str, optional
        Metric to optimize ('mape', 'rmse', 'mae', 'smape').
    fixed_params : Dict[str, Any], optional
        Fixed hyperparameters not subject to optimization.
    
    Returns
    -------
    Callable
        Objective function f(x) -> float where x is hyperparameter vector.
    
    Examples
    --------
    >>> data = prepare_train_test('data/transactions.csv', '2020-01-01')
    >>> obj_func = create_optimization_objective(XGBoostForecaster, data, 'mape')
    >>> mape = obj_func(np.array([0.1, 5, 0.8]))  # [learning_rate, max_depth, subsample]
    """
    fixed_params = fixed_params or {}
    
    def objective(hyperparams: NDArray[np.float64]) -> float:
        """
        Objective function to minimize.
        
        Parameters
        ----------
        hyperparams : NDArray[np.float64]
            Hyperparameter vector.
        
        Returns
        -------
        float
            Metric value (lower is better).
        """
        try:
            # Map hyperparameters
            param_space = model_class().get_hyperparameter_space()
            param_names = list(param_space.keys())
            
            if len(hyperparams) != len(param_names):
                raise ValueError(f"Expected {len(param_names)} hyperparameters, "
                               f"got {len(hyperparams)}")
            
            # Build parameter dictionary
            params = {**fixed_params}
            for i, name in enumerate(param_names):
                params[name] = float(hyperparams[i])
            
            # Handle integer parameters
            int_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 
                         'min_child_weight', 'p', 'd', 'q', 'seasonal_periods']
            for param in int_params:
                if param in params:
                    params[param] = int(params[param])
            
            # Train model
            model = model_class(**params)
            model.fit(
                data_dict['X_train'],
                data_dict['y_train'],
                X_test=data_dict['X_test'],
                y_test=data_dict['y_test']
            )
            
            # Predict and evaluate
            y_pred = model.predict(data_dict['X_test'])
            metrics = calculate_metrics(
                data_dict['y_test'],
                y_pred,
                data_dict['y_train'].values
            )
            
            return metrics[metric]
            
        except Exception as e:
            # Return large penalty for failed evaluations
            warnings.warn(f"Optimization failed: {str(e)}")
            return 1e6
    
    return objective


# ============================================================================
# HELPER FUNCTION (Backward Compatibility)
# ============================================================================

def optimize_helper(learning_rate: float, subsample: float) -> float:
    """
    Simple 2-parameter XGBoost optimization (backward compatible).
    
    This function is maintained for compatibility with existing code.
    For new projects, use create_optimization_objective() instead.
    
    Parameters
    ----------
    learning_rate : float
        XGBoost learning rate (0 < Î· â‰¤ 1).
    subsample : float
        Fraction of samples to use (0 < subsample â‰¤ 1).
    
    Returns
    -------
    float
        MAPE on test set.
    
    Notes
    -----
    Uses hardcoded paths:
    - Data file: './src/data/transactions.csv' or './transactions.csv'
    - Split date: '01-01-2020' or '2020-01-01'
    
    For production use, consider the more flexible create_optimization_objective().
    """
    # Try both possible file locations
    import os
    if os.path.exists('./src/data/transactions.csv'):
        FILE_PATH = './src/data/transactions.csv'
    elif os.path.exists('./transactions.csv'):
        FILE_PATH = './transactions.csv'
    else:
        warnings.warn("transactions.csv not found in ./src/data/ or ./")
        return 1e6
    
    SPLIT_DATE = '2020-01-01'
    
    HYPERPARAMS = {
        'n_estimators': 1000,
        'early_stopping_rounds': 100,
        'max_depth': 2,
        'subsample': subsample,
        'learning_rate': learning_rate,
        'random_state': 42
    }
    
    try:
        data = prepare_train_test(FILE_PATH, SPLIT_DATE)
        
        model = XGBoostForecaster(**HYPERPARAMS)
        model.fit(
            data['X_train'],
            data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        
        y_pred = model.predict(data['X_test'])
        metrics = calculate_metrics(data['y_test'], y_pred)
        
        return metrics['mape']
        
    except Exception as e:
        warnings.warn(f"optimize_helper failed: {str(e)}")
        return 1e6


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of forecasting module capabilities.
    """
    print("=" * 70)
    print("TIME SERIES FORECASTING MODULE")
    print("=" * 70)
    
    print("\n1. Available Forecasters:")
    print("-" * 70)
    forecasters = {
        'XGBoost': XGBoostForecaster if XGB_AVAILABLE else None,
        'LightGBM': LightGBMForecaster if LGB_AVAILABLE else None,
        'ARIMA': ARIMAForecaster if STATSMODELS_AVAILABLE else None,
        'ETS': ETSForecaster if STATSMODELS_AVAILABLE else None,
        'LSTM': LSTMForecaster if TORCH_AVAILABLE else None,
        'GRU': GRUForecaster if TORCH_AVAILABLE else None,
        'N-BEATS': NBeatsForecaster if TORCH_AVAILABLE else None
    }
    
    for name, forecaster_class in forecasters.items():
        status = "âœ“ Available" if forecaster_class else "âœ— Not installed"
        print(f"   {name:15s} {status}")
        
        if forecaster_class:
            space = forecaster_class().get_hyperparameter_space()
            print(f"      Hyperparameters: {list(space.keys())}")
    
    print("\n2. Supported Metrics:")
    print("-" * 70)
    metrics_list = ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MASE']
    print(f"   {', '.join(metrics_list)}")
    
    print("\n3. Data Format:")
    print("-" * 70)
    print("   Expected file: src/data/transactions.csv")
    print("   Required columns: date, transactions")
    print("   Date format: MM/DD/YYYY")
    
    print("\n" + "=" * 70)
    print("For usage examples, see documentation or forecast_example.py")
    print("=" * 70)