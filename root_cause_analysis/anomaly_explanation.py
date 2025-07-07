"""
异常根因分析与解释性
为检测到的异常提供可解释的原因分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyExplanation:
    """异常解释结果"""
    timestamp: pd.Timestamp
    value: float
    anomaly_score: float
    primary_cause: str
    contributing_factors: List[Tuple[str, float]]  # (因子名称, 贡献度)
    pattern_type: str  # 'spike', 'drop', 'drift', 'oscillation'
    context: Dict[str, Any]

class RootCauseAnalyzer:
    """异常根因分析器"""
    
    def __init__(self):
        self.feature_importance = {}
        self.historical_patterns = {}
        self.context_analyzers = []
        
    def extract_features(self, data: pd.Series, index: int, window_size: int = 10) -> Dict[str, float]:
        """
        提取异常点周围的特征
        
        Args:
            data: 时间序列数据
            index: 异常点索引
            window_size: 分析窗口大小
        """
        start_idx = max(0, index - window_size)
        end_idx = min(len(data), index + window_size + 1)
        
        # 基础窗口数据
        window_data = data.iloc[start_idx:end_idx]
        before_window = data.iloc[max(0, start_idx-window_size):start_idx] if start_idx > 0 else pd.Series([])
        
        features = {}
        
        # 1. 统计特征
        features['current_value'] = data.iloc[index]
        features['window_mean'] = window_data.mean()
        features['window_std'] = window_data.std()
        features['window_median'] = window_data.median()
        features['window_range'] = window_data.max() - window_data.min()
        
        # 2. 相对位置特征
        if len(before_window) > 0:
            features['relative_to_before'] = data.iloc[index] / (before_window.mean() + 1e-6)
            features['change_from_before'] = data.iloc[index] - before_window.mean()
        else:
            features['relative_to_before'] = 1.0
            features['change_from_before'] = 0.0
        
        # 3. 局部趋势特征
        if index >= 3:
            recent_trend = np.polyfit(range(3), data.iloc[index-2:index+1], 1)[0]
            features['recent_trend'] = recent_trend
        else:
            features['recent_trend'] = 0.0
        
        # 4. 变化率特征
        if index > 0:
            features['rate_of_change'] = data.iloc[index] - data.iloc[index-1]
            features['acceleration'] = features['rate_of_change'] - (
                data.iloc[index-1] - data.iloc[index-2] if index > 1 else 0
            )
        else:
            features['rate_of_change'] = 0.0
            features['acceleration'] = 0.0
        
        # 5. 周期性特征
        for period in [7, 24, 144]:  # 周、日、小时周期
            if index >= period:
                seasonal_value = data.iloc[index - period]
                features[f'seasonal_diff_{period}'] = data.iloc[index] - seasonal_value
                features[f'seasonal_ratio_{period}'] = data.iloc[index] / (seasonal_value + 1e-6)
            else:
                features[f'seasonal_diff_{period}'] = 0.0
                features[f'seasonal_ratio_{period}'] = 1.0
        
        # 6. 局部极值特征
        local_data = window_data
        features['is_local_max'] = float(data.iloc[index] == local_data.max())
        features['is_local_min'] = float(data.iloc[index] == local_data.min())
        features['local_rank'] = (local_data < data.iloc[index]).sum() / len(local_data)
        
        # 7. 波动性特征
        if len(window_data) > 2:
            features['local_volatility'] = window_data.rolling(3).std().mean()
        else:
            features['local_volatility'] = 0.0
        
        return features
    
    def identify_pattern_type(self, data: pd.Series, index: int, window_size: int = 5) -> str:
        """识别异常模式类型"""
        value = data.iloc[index]
        
        # 获取周围数据
        start_idx = max(0, index - window_size)
        end_idx = min(len(data), index + window_size + 1)
        surrounding = data.iloc[start_idx:end_idx]
        
        # 计算统计量
        mean_surrounding = surrounding.mean()
        std_surrounding = surrounding.std()
        
        # 模式识别
        if value > mean_surrounding + 2 * std_surrounding:
            if index > 0 and index < len(data) - 1:
                # 检查是否为尖峰
                before = data.iloc[index - 1] if index > 0 else mean_surrounding
                after = data.iloc[index + 1] if index < len(data) - 1 else mean_surrounding
                if abs(value - before) > std_surrounding and abs(value - after) > std_surrounding:
                    return 'spike'
            return 'high_value'
        
        elif value < mean_surrounding - 2 * std_surrounding:
            if index > 0 and index < len(data) - 1:
                # 检查是否为尖谷
                before = data.iloc[index - 1] if index > 0 else mean_surrounding
                after = data.iloc[index + 1] if index < len(data) - 1 else mean_surrounding
                if abs(value - before) > std_surrounding and abs(value - after) > std_surrounding:
                    return 'drop'
            return 'low_value'
        
        # 检查趋势变化
        if index >= 3:
            recent_values = data.iloc[index-2:index+1]
            if len(recent_values) == 3:
                trend = np.polyfit(range(3), recent_values, 1)[0]
                if abs(trend) > std_surrounding / 2:
                    return 'drift'
        
        # 检查振荡
        if index >= 2 and index < len(data) - 2:
            local_window = data.iloc[index-2:index+3]
            if len(local_window) == 5:
                changes = local_window.diff().dropna()
                if (changes > 0).sum() >= 2 and (changes < 0).sum() >= 2:
                    return 'oscillation'
        
        return 'unknown'
    
    def analyze_anomaly(self, data: pd.Series, anomaly_index: int, 
                       context_data: Dict[str, pd.Series] = None) -> AnomalyExplanation:
        """
        分析单个异常点的根因
        
        Args:
            data: 主要时间序列数据
            anomaly_index: 异常点索引
            context_data: 上下文数据（如其他指标）
        """
        # 提取特征
        features = self.extract_features(data, anomaly_index)
        
        # 识别模式类型
        pattern_type = self.identify_pattern_type(data, anomaly_index)
        
        # 计算异常分数
        value = data.iloc[anomaly_index]
        window_data = data.iloc[max(0, anomaly_index-10):anomaly_index+11]
        z_score = abs(value - window_data.mean()) / (window_data.std() + 1e-6)
        
        # 分析主要原因
        primary_cause = self._identify_primary_cause(features, pattern_type)
        
        # 计算贡献因子
        contributing_factors = self._calculate_contributing_factors(features)
        
        # 上下文分析
        context = self._analyze_context(data, anomaly_index, context_data)
        
        return AnomalyExplanation(
            timestamp=data.index[anomaly_index] if hasattr(data.index, '__getitem__') else anomaly_index,
            value=value,
            anomaly_score=min(z_score / 3.0, 1.0),  # 标准化到0-1
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            pattern_type=pattern_type,
            context=context
        )
    
    def _identify_primary_cause(self, features: Dict[str, float], pattern_type: str) -> str:
        """识别主要原因"""
        # 根据特征和模式类型确定主要原因
        if pattern_type == 'spike':
            if features['rate_of_change'] > features['window_std'] * 2:
                return "突发性峰值"
            elif features.get('seasonal_diff_24', 0) > features['window_std']:
                return "偏离日常模式"
            else:
                return "局部异常峰值"
        
        elif pattern_type == 'drop':
            if features['rate_of_change'] < -features['window_std'] * 2:
                return "突发性下降"
            elif features.get('seasonal_diff_24', 0) < -features['window_std']:
                return "低于预期水平"
            else:
                return "局部异常低值"
        
        elif pattern_type == 'drift':
            if abs(features['recent_trend']) > features['window_std']:
                return "趋势异常变化"
            else:
                return "渐进性偏移"
        
        elif pattern_type == 'oscillation':
            return "异常振荡模式"
        
        else:
            # 基于特征重要性判断
            if abs(features['relative_to_before'] - 1) > 0.5:
                return "相对历史水平异常"
            elif abs(features['change_from_before']) > features['window_std']:
                return "偏离基线水平"
            else:
                return "统计异常值"
    
    def _calculate_contributing_factors(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """计算贡献因子"""
        factor_scores = []
        
        # 基于特征值计算贡献度
        feature_weights = {
            'relative_to_before': 0.25,
            'change_from_before': 0.20,
            'rate_of_change': 0.15,
            'recent_trend': 0.10,
            'local_volatility': 0.10,
            'seasonal_diff_24': 0.20
        }
        
        for feature, weight in feature_weights.items():
            if feature in features:
                # 标准化特征值
                normalized_value = min(abs(features[feature]) / (features.get('window_std', 1) + 1e-6), 2.0)
                contribution = weight * normalized_value
                
                # 映射到可读名称
                readable_name = {
                    'relative_to_before': '相对历史比例',
                    'change_from_before': '基线偏移',
                    'rate_of_change': '变化率',
                    'recent_trend': '短期趋势',
                    'local_volatility': '局部波动',
                    'seasonal_diff_24': '日周期偏差'
                }.get(feature, feature)
                
                factor_scores.append((readable_name, contribution))
        
        # 按贡献度排序
        factor_scores.sort(key=lambda x: x[1], reverse=True)
        
        return factor_scores[:5]  # 返回前5个主要因子
    
    def _analyze_context(self, data: pd.Series, index: int, 
                        context_data: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """分析上下文信息"""
        context = {}
        
        # 时间上下文
        if hasattr(data.index, '__getitem__') and hasattr(data.index[index], 'hour'):
            timestamp = data.index[index]
            context['hour_of_day'] = timestamp.hour
            context['day_of_week'] = timestamp.weekday()
            context['is_weekend'] = timestamp.weekday() >= 5
            context['is_business_hours'] = 9 <= timestamp.hour <= 17
        
        # 历史对比
        window_data = data.iloc[max(0, index-50):index]
        if len(window_data) > 0:
            context['percentile_in_history'] = (window_data < data.iloc[index]).sum() / len(window_data)
            context['historical_max'] = window_data.max()
            context['historical_min'] = window_data.min()
        
        # 关联指标分析
        if context_data:
            correlations = {}
            for name, series in context_data.items():
                if index < len(series):
                    # 计算短期相关性
                    start_idx = max(0, index - 20)
                    data_window = data.iloc[start_idx:index+1]
                    context_window = series.iloc[start_idx:index+1]
                    
                    if len(data_window) > 3 and len(context_window) > 3:
                        corr = data_window.corr(context_window)
                        if not np.isnan(corr):
                            correlations[name] = corr
            
            context['correlations'] = correlations
        
        return context
    
    def generate_explanation_report(self, explanation: AnomalyExplanation) -> str:
        """生成可读的解释报告"""
        report = []
        
        report.append(f"🔍 异常分析报告")
        report.append(f"=" * 50)
        report.append(f"时间点: {explanation.timestamp}")
        report.append(f"异常值: {explanation.value:.2f}")
        report.append(f"异常分数: {explanation.anomaly_score:.3f}")
        report.append(f"模式类型: {explanation.pattern_type}")
        report.append("")
        
        report.append(f"🎯 主要原因: {explanation.primary_cause}")
        report.append("")
        
        report.append(f"📊 贡献因子:")
        for factor, contribution in explanation.contributing_factors:
            bar_length = int(contribution * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            report.append(f"  {factor:.<20} {bar} {contribution:.3f}")
        report.append("")
        
        if explanation.context:
            report.append(f"🔗 上下文信息:")
            for key, value in explanation.context.items():
                if key != 'correlations':
                    report.append(f"  {key}: {value}")
            
            if 'correlations' in explanation.context:
                report.append("  相关指标:")
                for metric, corr in explanation.context['correlations'].items():
                    report.append(f"    {metric}: {corr:.3f}")
        
        return "\n".join(report)
    
    def batch_analyze_anomalies(self, data: pd.Series, anomaly_indices: List[int],
                               context_data: Dict[str, pd.Series] = None) -> List[AnomalyExplanation]:
        """批量分析多个异常点"""
        explanations = []
        
        for index in anomaly_indices:
            if 0 <= index < len(data):
                explanation = self.analyze_anomaly(data, index, context_data)
                explanations.append(explanation)
        
        return explanations
    
    def plot_anomaly_explanation(self, data: pd.Series, explanation: AnomalyExplanation, 
                                window_size: int = 20):
        """可视化异常解释"""
        index = explanation.timestamp if isinstance(explanation.timestamp, int) else None
        
        # 如果timestamp是pandas时间戳，需要找到对应的索引
        if index is None:
            try:
                index = data.index.get_loc(explanation.timestamp)
            except:
                print("无法找到对应的索引位置")
                return
        
        # 确定绘图窗口
        start_idx = max(0, index - window_size)
        end_idx = min(len(data), index + window_size + 1)
        
        window_data = data.iloc[start_idx:end_idx]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 主图：时间序列和异常点
        ax1.plot(range(len(window_data)), window_data.values, 
                label='Time Series', alpha=0.7, linewidth=2)
        
        anomaly_pos = index - start_idx
        ax1.scatter([anomaly_pos], [explanation.value], 
                   color='red', s=100, zorder=5, label='Anomaly')
        
        # 添加统计信息
        mean_val = window_data.mean()
        std_val = window_data.std()
        ax1.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label='Mean')
        ax1.axhline(y=mean_val + 2*std_val, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        ax1.axhline(y=mean_val - 2*std_val, color='orange', linestyle=':', alpha=0.7)
        
        ax1.set_title(f'Anomaly Context: {explanation.primary_cause}')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：贡献因子
        factors = [f[0] for f in explanation.contributing_factors]
        contributions = [f[1] for f in explanation.contributing_factors]
        
        bars = ax2.barh(factors, contributions, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Contribution Score')
        ax2.set_title('Contributing Factors')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, contribution in zip(bars, contributions):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{contribution:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('root_cause_analysis/anomaly_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='10min')
    
    # 基础模式：日周期 + 噪声
    base_pattern = 30 + 15 * np.sin(2 * np.pi * np.arange(200) / 144)
    noise = np.random.normal(0, 3, 200)
    cpu_data = base_pattern + noise
    
    # 插入不同类型的异常
    cpu_data[50] = 80   # 尖峰
    cpu_data[100] = 5   # 下降
    cpu_data[150:155] += 20  # 漂移
    
    # 创建DataFrame
    df = pd.DataFrame({'cpu_usage': cpu_data}, index=dates)
    
    # 创建根因分析器
    analyzer = RootCauseAnalyzer()
    
    # 分析异常点
    anomaly_indices = [50, 100, 152]
    explanations = analyzer.batch_analyze_anomalies(df['cpu_usage'], anomaly_indices)
    
    # 生成报告
    for i, explanation in enumerate(explanations):
        print(f"\n异常点 {i+1}:")
        print(analyzer.generate_explanation_report(explanation))
        print("-" * 80)
    
    # 可视化最后一个异常的解释
    if explanations:
        analyzer.plot_anomaly_explanation(df['cpu_usage'], explanations[-1])
    
    print(f"\n总共分析了 {len(explanations)} 个异常点")
