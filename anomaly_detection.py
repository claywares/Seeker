import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy import stats

class AnomalyDetector:
    """Base class for anomaly detection algorithms"""
    
    def __init__(self, name):
        self.name = name
        
    def fit(self, data):
        """Fit the model to the data"""
        pass
        
    def predict(self, data):
        """Predict anomalies in the data"""
        pass
    
    def evaluate(self, data, true_anomalies):
        """Evaluate the model performance"""
        predictions = self.predict(data)
        precision = precision_score(true_anomalies, predictions)
        recall = recall_score(true_anomalies, predictions)
        f1 = f1_score(true_anomalies, predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class StatisticalDetector(AnomalyDetector):
    """Statistical anomaly detection using Z-score"""
    
    def __init__(self, threshold=3.0):
        super().__init__(f"Z-Score (threshold={threshold})")
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, data):
        """Compute mean and standard deviation of the data"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        return self
        
    def predict(self, data):
        """Detect anomalies using Z-score"""
        if self.mean is None or self.std is None:
            self.fit(data)
            
        z_scores = np.abs((data - self.mean) / self.std)
        return (z_scores > self.threshold).astype(int)


class EWMADetector(AnomalyDetector):
    """Exponentially Weighted Moving Average anomaly detector"""
    
    def __init__(self, span=20, threshold=3.0):
        super().__init__(f"EWMA (span={span}, threshold={threshold})")
        self.span = span
        self.threshold = threshold
        
    def fit(self, data):
        """Nothing to fit for this detector"""
        return self
        
    def predict(self, data):
        """Detect anomalies using EWMA"""
        ewma = pd.Series(data).ewm(span=self.span).mean()
        ewmstd = pd.Series(data).ewm(span=self.span).std()
        
        upper_bound = ewma + self.threshold * ewmstd
        lower_bound = ewma - self.threshold * ewmstd
        
        anomalies = ((data > upper_bound) | (data < lower_bound)).astype(int)
        return anomalies


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    def __init__(self, contamination=0.01, random_state=42):
        super().__init__(f"Isolation Forest (contamination={contamination})")
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        
    def fit(self, data):
        """Fit Isolation Forest model"""
        # Reshape data for sklearn
        X = np.array(data).reshape(-1, 1)
        self.model.fit(X)
        return self
        
    def predict(self, data):
        """Detect anomalies using Isolation Forest"""
        X = np.array(data).reshape(-1, 1)
        # Convert from model output (-1 for anomalies, 1 for normal) to binary (1 for anomalies, 0 for normal)
        predictions = (self.model.predict(X) == -1).astype(int)
        return predictions


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM anomaly detector"""
    
    def __init__(self, nu=0.01):
        super().__init__(f"One-Class SVM (nu={nu})")
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.scaler = StandardScaler()
        
    def fit(self, data):
        """Fit One-Class SVM model"""
        X = np.array(data).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self
        
    def predict(self, data):
        """Detect anomalies using One-Class SVM"""
        X = np.array(data).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        # Convert from model output (1 for inliers, -1 for outliers) to binary (1 for anomalies, 0 for normal)
        predictions = (self.model.predict(X_scaled) == -1).astype(int)
        return predictions


class STLDecompositionDetector(AnomalyDetector):
    """STL Decomposition-based anomaly detector"""
    
    def __init__(self, period=50, threshold=3.0):
        super().__init__(f"STL Decomposition (threshold={threshold})")
        self.period = period
        self.threshold = threshold
        self.residual_mean = None
        self.residual_std = None
        
    def fit(self, data, timestamps=None):
        """Fit STL decomposition model"""
        # Create a pandas Series with a DatetimeIndex if timestamps are provided
        if timestamps is not None:
            ts = pd.Series(data, index=timestamps)
        else:
            ts = pd.Series(data)
            
        # Apply STL decomposition
        stl = STL(ts, period=self.period)
        result = stl.fit()
        
        # Calculate statistics of the residual component
        self.residual_mean = np.mean(result.resid)
        self.residual_std = np.std(result.resid)
        
        return self
        
    def predict(self, data, timestamps=None):
        """Detect anomalies using STL decomposition"""
        # Create a pandas Series with a DatetimeIndex if timestamps are provided
        if timestamps is not None:
            ts = pd.Series(data, index=timestamps)
        else:
            ts = pd.Series(data)
            
        # Apply STL decomposition
        stl = STL(ts, period=self.period)
        result = stl.fit()
        
        # Calculate z-scores of residuals
        if self.residual_mean is None or self.residual_std is None:
            residual_mean = np.mean(result.resid)
            residual_std = np.std(result.resid)
        else:
            residual_mean = self.residual_mean
            residual_std = self.residual_std
            
        z_scores = np.abs((result.resid - residual_mean) / residual_std)
        
        # Detect anomalies
        anomalies = (z_scores > self.threshold).astype(int)
        return anomalies


def visualize_anomalies(df, predictions_dict, title="Anomaly Detection Results"):
    """
    Visualize the original time series and the detected anomalies
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with 'timestamp' and 'value' columns
    predictions_dict : dict
        Dictionary mapping detector names to their anomaly predictions
    title : str
        Title of the plot
    """
    n_detectors = len(predictions_dict)
    fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(15, 3 * (n_detectors + 1)), sharex=True)
    
    # Plot the original time series with ground truth anomalies
    ax = axes[0]
    ax.plot(df['timestamp'], df['value'], label='Time Series')
    
    if 'is_anomaly' in df.columns:
        # Plot ground truth anomalies
        anomaly_points = df[df['is_anomaly'] == 1]
        ax.scatter(anomaly_points['timestamp'], anomaly_points['value'], 
                   color='red', label='True Anomalies', s=50, zorder=5)
    
    ax.set_title('Original Time Series with Ground Truth Anomalies')
    ax.legend()
    ax.grid(True)
    
    # Plot the detected anomalies for each detector
    for i, (detector_name, predictions) in enumerate(predictions_dict.items(), 1):
        ax = axes[i]
        ax.plot(df['timestamp'], df['value'], label='Time Series')
        
        # Plot detected anomalies
        detected_anomalies = df.copy()
        detected_anomalies['is_detected'] = predictions
        anomaly_points = detected_anomalies[detected_anomalies['is_detected'] == 1]
        
        ax.scatter(anomaly_points['timestamp'], anomaly_points['value'], 
                   color='orange', label=f'Detected by {detector_name}', s=50, zorder=5)
        
        ax.set_title(f'Anomalies Detected by {detector_name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    return fig


def evaluate_detectors(df, detectors, true_anomalies=None):
    """
    Evaluate multiple anomaly detectors on the given data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data with 'timestamp' and 'value' columns
    detectors : list
        List of AnomalyDetector instances
    true_anomalies : array-like, optional
        Ground truth anomaly labels. If None, uses df['is_anomaly'] if available
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing evaluation metrics for each detector
    predictions_dict : dict
        Dictionary mapping detector names to their anomaly predictions
    """
    if true_anomalies is None:
        if 'is_anomaly' in df.columns:
            true_anomalies = df['is_anomaly'].values
        else:
            raise ValueError("Ground truth anomalies not provided and 'is_anomaly' column not found in df")
    
    results = []
    predictions_dict = {}
    
    for detector in detectors:
        # Fit the detector
        detector.fit(df['value'].values)
        
        # Make predictions
        predictions = detector.predict(df['value'].values)
        predictions_dict[detector.name] = predictions
        
        # Calculate metrics
        precision = precision_score(true_anomalies, predictions, zero_division=0)
        recall = recall_score(true_anomalies, predictions, zero_division=0)
        f1 = f1_score(true_anomalies, predictions, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_anomalies, predictions, labels=[0, 1]).ravel()
        
        results.append({
            'Detector': detector.name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn
        })
    
    results_df = pd.DataFrame(results)
    return results_df, predictions_dict
