"""
基础异常检测算法模块

包含传统统计方法和机器学习方法
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """异常检测器基类"""
    
    @abstractmethod
    def detect(self, data, **kwargs):
        """检测异常"""
        pass


class ZScoreDetector(BaseDetector):
    """Z-score异常检测器"""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def detect(self, data, **kwargs):
        """Z-score检测"""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > self.threshold
        return anomalies, z_scores


class IQRDetector(BaseDetector):
    """IQR异常检测器"""
    
    def __init__(self, k=1.5):
        self.k = k
    
    def detect(self, data, **kwargs):
        """IQR检测"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.k * IQR
        upper_bound = Q3 + self.k * IQR
        anomalies = (data < lower_bound) | (data > upper_bound)
        deviations = np.maximum((lower_bound - data)/IQR, (data - upper_bound)/IQR)
        deviations = np.maximum(deviations, 0)
        return anomalies, deviations


class EWMADetector(BaseDetector):
    """EWMA异常检测器"""
    
    def __init__(self, span=15, threshold=2.0):
        self.span = span
        self.threshold = threshold
    
    def detect(self, data, **kwargs):
        """EWMA检测"""
        ewma_mean = pd.Series(data).ewm(span=self.span).mean()
        ewma_std = pd.Series(data).ewm(span=self.span).std()
        deviations = np.abs((data - ewma_mean) / ewma_std)
        anomalies = deviations > self.threshold
        return anomalies.values, deviations.values


class IsolationForestDetector(BaseDetector):
    """Isolation Forest检测器"""
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
    
    def detect(self, data, **kwargs):
        """Isolation Forest检测"""
        model = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state
        )
        data_2d = data.reshape(-1, 1)
        anomalies = model.fit_predict(data_2d) == -1
        scores = -model.score_samples(data_2d)
        return anomalies, scores


class LOFDetector(BaseDetector):
    """LOF检测器"""
    
    def __init__(self, n_neighbors=20, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
    
    def detect(self, data, **kwargs):
        """LOF检测"""
        model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        data_2d = data.reshape(-1, 1)
        anomalies = model.fit_predict(data_2d) == -1
        scores = -model.negative_outlier_factor_
        return anomalies, scores


class MultiDetector:
    """多算法检测器管理器"""
    
    def __init__(self):
        self.detectors = {
            'zscore': ZScoreDetector(threshold=2.5),
            'iqr': IQRDetector(k=1.8),
            'ewma': EWMADetector(span=12, threshold=2.0),
            'iforest': IsolationForestDetector(contamination=0.03),
            'lof': LOFDetector(n_neighbors=15, contamination=0.03)
        }
    
    def detect_all(self, data):
        """运行所有检测算法"""
        results = {}
        for name, detector in self.detectors.items():
            anomalies, scores = detector.detect(data)
            results[name] = {
                'anomalies': anomalies,
                'scores': scores
            }
        return results
