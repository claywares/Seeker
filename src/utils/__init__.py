"""
工具函数模块

包含特征提取、数据处理、可视化等工具
"""
import numpy as np
import pandas as pd
from ..detectors import MultiDetector


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.detector = MultiDetector()
        
    def extract_features(self, data):
        """提取异常检测特征"""
        cpu_usage = data['cpu_usage'].values
        
        # 运行基础检测算法
        detection_results = self.detector.detect_all(cpu_usage)
        
        # 基础检测特征
        features = pd.DataFrame({
            'zscore_anomaly': detection_results['zscore']['anomalies'].astype(int),
            'iqr_anomaly': detection_results['iqr']['anomalies'].astype(int),
            'ewma_anomaly': detection_results['ewma']['anomalies'].astype(int),
            'iforest_anomaly': detection_results['iforest']['anomalies'].astype(int),
            'lof_anomaly': detection_results['lof']['anomalies'].astype(int),
        })
        
        # 统计特征
        features['consensus_ratio'] = features.iloc[:, :5].sum(axis=1) / 5
        features['deviation_zscore'] = detection_results['zscore']['scores']
        features['deviation_iqr'] = detection_results['iqr']['scores']
        features['deviation_ewma'] = detection_results['ewma']['scores']
        features['deviation_iforest'] = detection_results['iforest']['scores']
        features['deviation_lof'] = detection_results['lof']['scores']
        features['max_deviation'] = features[['deviation_zscore', 'deviation_iqr', 'deviation_ewma']].max(axis=1)
        features['mean_deviation'] = features[['deviation_zscore', 'deviation_iqr', 'deviation_ewma']].mean(axis=1)
        
        # 时序特征
        features['prev_anomaly_count'] = features['iforest_anomaly'].rolling(window=self.window_size).sum().fillna(0)
        
        # 趋势特征
        cpu_series = pd.Series(cpu_usage)
        features['trend_slope'] = cpu_series.rolling(window=self.window_size).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == self.window_size else 0
        ).fillna(0)
        
        # 波动性
        features['volatility'] = cpu_series.rolling(window=self.window_size).std().fillna(0)
        
        # 季节性得分
        if 'timestamp' in data.columns:
            hours = pd.to_datetime(data['timestamp']).dt.hour
            hour_mean = cpu_series.groupby(hours).mean()
            seasonality_baseline = np.array([hour_mean.get(h, cpu_series.mean()) for h in hours])
            features['seasonality_score'] = abs(cpu_usage - seasonality_baseline)
        else:
            features['seasonality_score'] = 0
            
        return features
    
    def get_feature_names(self):
        """获取特征名称"""
        return [
            'zscore_anomaly', 'iqr_anomaly', 'ewma_anomaly', 'iforest_anomaly', 'lof_anomaly',
            'consensus_ratio', 'deviation_zscore', 'deviation_iqr', 'deviation_ewma',
            'deviation_iforest', 'deviation_lof', 'max_deviation', 'mean_deviation',
            'prev_anomaly_count', 'trend_slope', 'volatility', 'seasonality_score'
        ]


class DataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_cpu_data(n_points=500, seed=42):
        """生成CPU使用率测试数据"""
        np.random.seed(seed)
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='5min')
        
        # 基础数据
        base_cpu = np.random.normal(loc=30, scale=8, size=n_points)
        
        # 日周期性
        hours = np.array([ts.hour for ts in timestamps])
        daily_pattern = 10 * np.sin(2 * np.pi * hours / 24) + 5
        cpu_usage = base_cpu + daily_pattern
        cpu_usage = np.clip(cpu_usage, 0, 100)
        
        # 插入异常
        anomaly_indices = []
        
        # 突发峰值
        spike_indices = [50, 150, 300, 450]
        cpu_usage[spike_indices] = [95, 92, 88, 94]
        anomaly_indices.extend(spike_indices)
        
        # 突发低谷
        dip_indices = [80, 200, 350]
        cpu_usage[dip_indices] = [2, 1, 3]
        anomaly_indices.extend(dip_indices)
        
        # 持续性异常
        sustained_start, sustained_end = 120, 130
        cpu_usage[sustained_start:sustained_end] = np.random.normal(85, 3, sustained_end-sustained_start)
        anomaly_indices.extend(range(sustained_start, sustained_end))
        
        # 趋势异常
        trend_start, trend_end = 250, 270
        trend_values = np.linspace(cpu_usage[trend_start], 80, trend_end-trend_start)
        cpu_usage[trend_start:trend_end] = trend_values
        anomaly_indices.extend(range(trend_start, trend_end))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage,
            'is_anomaly': 0
        })
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        return df


class MetricsCalculator:
    """性能指标计算器"""
    
    @staticmethod
    def evaluate_performance(true_labels, scores, threshold_percentile=95):
        """评估检测性能"""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        threshold = np.percentile(scores, threshold_percentile)
        predictions = (scores > threshold).astype(int)
        
        metrics = {
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'auc': roc_auc_score(true_labels, scores) if len(np.unique(true_labels)) > 1 else 0,
            'predictions': predictions,
            'threshold': threshold
        }
        
        return metrics
