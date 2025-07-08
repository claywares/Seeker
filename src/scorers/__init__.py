"""
智能评分器模块

包含三维评分体系、Random Forest评分器、神经网络评分器等
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from abc import ABC, abstractmethod
from ..detectors import MultiDetector
from ..utils import FeatureExtractor


class BaseScorer(ABC):
    """评分器基类"""
    
    @abstractmethod
    def fit(self, data, labels=None):
        """训练评分器"""
        pass
    
    @abstractmethod
    def score(self, data):
        """计算异常评分"""
        pass


class OriginalScorer(BaseScorer):
    """原始三维评分体系"""
    
    def __init__(self, 
                 consensus_weight=0.4,
                 deviation_weight=0.4, 
                 persistence_weight=0.2):
        self.consensus_weight = consensus_weight
        self.deviation_weight = deviation_weight
        self.persistence_weight = persistence_weight
        self.detector = MultiDetector()
        self.feature_extractor = FeatureExtractor()
        
    def fit(self, data, labels=None):
        """原始评分器不需要训练"""
        return self
    
    def score(self, data):
        """计算三维评分"""
        # 运行基础检测
        detection_results = self.detector.detect_all(data['cpu_usage'].values)
        
        # 方法一致性评分
        consensus_count = sum([
            detection_results[method]['anomalies'].astype(int) 
            for method in detection_results.keys()
        ])
        consensus_score = consensus_count / len(detection_results)
        
        # 偏离程度评分
        max_deviation = np.maximum.reduce([
            detection_results['zscore']['scores'],
            detection_results['iqr']['scores'],
            detection_results['ewma']['scores']
        ])
        deviation_score = max_deviation / (max_deviation.max() + 1e-8)
        
        # 持续性评分
        iforest_anomalies = detection_results['iforest']['anomalies']
        persistence_score = pd.Series(iforest_anomalies.astype(int)).rolling(
            window=3, center=True
        ).sum().fillna(0) / 3
        
        # 综合评分
        anomaly_scores = (
            self.consensus_weight * consensus_score +
            self.deviation_weight * deviation_score +
            self.persistence_weight * persistence_score
        )
        
        return anomaly_scores.values


class RandomForestScorer(BaseScorer):
    """Random Forest评分器"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.feature_extractor = FeatureExtractor()
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _generate_pseudo_labels(self, features, threshold_percentile=95):
        """生成伪标签"""
        # 使用原始评分器生成伪标签
        original_scorer = OriginalScorer()
        pseudo_data = pd.DataFrame({'cpu_usage': features['cpu_usage']})
        if 'timestamp' in features.columns:
            pseudo_data['timestamp'] = features['timestamp']
            
        original_scores = original_scorer.score(pseudo_data)
        threshold = np.percentile(original_scores, threshold_percentile)
        labels = (original_scores > threshold).astype(int)
        
        return labels, original_scores
    
    def fit(self, data, labels=None):
        """训练Random Forest评分器"""
        # 提取特征
        features = self.feature_extractor.extract_features(data)
        
        # 生成伪标签
        if labels is None:
            labels, _ = self._generate_pseudo_labels(data)
            
        # 标准化特征
        feature_matrix = features.select_dtypes(include=[np.number]).values
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # 训练模型
        self.rf_model.fit(feature_matrix_scaled, labels)
        self.is_fitted = True
        
        return self
    
    def score(self, data):
        """预测异常评分"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit() 方法")
            
        features = self.feature_extractor.extract_features(data)
        feature_matrix = features.select_dtypes(include=[np.number]).values
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # 返回异常概率
        proba = self.rf_model.predict_proba(feature_matrix_scaled)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
            
        feature_names = self.feature_extractor.get_feature_names()
        return pd.DataFrame({
            'feature': feature_names[:len(self.rf_model.feature_importances_)],
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)


class NeuralNetworkScorer(BaseScorer):
    """神经网络评分器"""
    
    def __init__(self, 
                 hidden_layer_sizes=(32, 16, 8),
                 max_iter=500,
                 random_state=42):
        self.feature_extractor = FeatureExtractor()
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, data, labels=None):
        """训练神经网络评分器"""
        # 提取特征
        features = self.feature_extractor.extract_features(data)
        
        # 生成伪标签
        if labels is None:
            rf_scorer = RandomForestScorer()
            labels, _ = rf_scorer._generate_pseudo_labels(data)
            
        # 标准化特征
        feature_matrix = features.select_dtypes(include=[np.number]).values
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # 训练模型
        self.nn_model.fit(feature_matrix_scaled, labels)
        self.is_fitted = True
        
        return self
    
    def score(self, data):
        """预测异常评分"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
            
        features = self.feature_extractor.extract_features(data)
        feature_matrix = features.select_dtypes(include=[np.number]).values
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        proba = self.nn_model.predict_proba(feature_matrix_scaled)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


class AnomalyClassifier:
    """异常分级器"""
    
    @staticmethod
    def classify(scores, n_levels=3):
        """分级异常"""
        if len(scores) == 0:
            return np.array([])
            
        thresholds = np.quantile(scores, [1-1/n_levels, 1-2/n_levels])
        
        levels = np.full(len(scores), f'P{n_levels-1}')
        levels[scores >= thresholds[1]] = f'P{n_levels-2}'
        levels[scores >= thresholds[0]] = 'P0'
        
        return levels
