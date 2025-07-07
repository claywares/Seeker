"""
时间序列分解与季节性异常检测
结合趋势、季节性和残差分析的高级异常检测方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SeasonalAnomalyDetector:
    """季节性异常检测器"""
    
    def __init__(self, period=144):  # 144 = 24小时 * 60分钟 / 10分钟间隔
        """
        Args:
            period: 季节性周期（默认144对应日周期，10分钟间隔）
        """
        self.period = period
        self.decomposition = None
        self.residual_threshold = None
        self.trend_detector = None
        self.seasonal_detector = None
        
    def decompose_series(self, data, model='additive'):
        """
        分解时间序列为趋势、季节性和残差组件
        
        Args:
            data: 时间序列数据
            model: 分解模型 ('additive' 或 'multiplicative')
        """
        if len(data) < 2 * self.period:
            print(f"警告: 数据长度 {len(data)} 小于建议的最小长度 {2 * self.period}")
            # 调整周期长度
            self.period = max(4, len(data) // 3)
            
        self.decomposition = seasonal_decompose(
            data, 
            model=model, 
            period=self.period,
            extrapolate_trend='freq'
        )
        
        return self.decomposition
    
    def detect_trend_anomalies(self, contamination=0.02):
        """检测趋势异常"""
        if self.decomposition is None:
            raise ValueError("请先运行 decompose_series")
            
        trend = self.decomposition.trend.dropna()
        
        # 使用滑动窗口检测趋势突变
        window_size = min(10, len(trend) // 10)
        trend_diff = trend.diff(window_size).abs()
        
        # 使用IQR方法检测趋势异常
        Q1 = trend_diff.quantile(0.25)
        Q3 = trend_diff.quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        
        trend_anomalies = trend_diff > threshold
        
        # 扩展到原始数据长度
        full_trend_anomalies = pd.Series(False, index=self.decomposition.trend.index)
        full_trend_anomalies.loc[trend_anomalies.index] = trend_anomalies
        
        return full_trend_anomalies.values
    
    def detect_seasonal_anomalies(self, contamination=0.02):
        """检测季节性异常"""
        if self.decomposition is None:
            raise ValueError("请先运行 decompose_series")
            
        seasonal = self.decomposition.seasonal
        
        # 计算每个季节位置的统计信息
        seasonal_stats = pd.DataFrame({
            'position': range(len(seasonal)) % self.period,
            'value': seasonal.values
        }).groupby('position').agg({
            'value': ['mean', 'std']
        }).round(4)
        
        seasonal_stats.columns = ['mean', 'std']
        
        # 检测每个点相对于其季节位置的异常
        seasonal_anomalies = []
        for i, value in enumerate(seasonal.values):
            pos = i % self.period
            mean_val = seasonal_stats.loc[pos, 'mean']
            std_val = seasonal_stats.loc[pos, 'std']
            
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                is_anomaly = z_score > 2.5  # 2.5σ阈值
            else:
                is_anomaly = False
                
            seasonal_anomalies.append(is_anomaly)
        
        return np.array(seasonal_anomalies)
    
    def detect_residual_anomalies(self, method='iqr', contamination=0.02):
        """检测残差异常"""
        if self.decomposition is None:
            raise ValueError("请先运行 decompose_series")
            
        residual = self.decomposition.resid.dropna()
        
        if method == 'iqr':
            Q1 = residual.quantile(0.25)
            Q3 = residual.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 2.0 * IQR
            residual_anomalies = (residual < lower_bound) | (residual > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((residual - residual.mean()) / residual.std())
            residual_anomalies = z_scores > 2.5
            
        elif method == 'isolation_forest':
            clf = IsolationForest(contamination=contamination, random_state=42)
            residual_anomalies = clf.fit_predict(residual.values.reshape(-1, 1)) == -1
            residual_anomalies = pd.Series(residual_anomalies, index=residual.index)
        
        # 扩展到原始数据长度
        full_residual_anomalies = pd.Series(False, index=self.decomposition.resid.index)
        full_residual_anomalies.loc[residual_anomalies.index] = residual_anomalies
        
        return full_residual_anomalies.values
    
    def detect_all_anomalies(self, data, weights={'trend': 0.3, 'seasonal': 0.3, 'residual': 0.4}):
        """
        综合检测所有类型的异常
        
        Args:
            data: 时间序列数据
            weights: 各组件权重
        """
        # 分解时间序列
        self.decompose_series(data)
        
        # 检测各组件异常
        trend_anomalies = self.detect_trend_anomalies()
        seasonal_anomalies = self.detect_seasonal_anomalies()
        residual_anomalies = self.detect_residual_anomalies()
        
        # 计算综合异常分数
        anomaly_score = (
            weights['trend'] * trend_anomalies.astype(int) +
            weights['seasonal'] * seasonal_anomalies.astype(int) +
            weights['residual'] * residual_anomalies.astype(int)
        )
        
        # 确定最终异常
        final_anomalies = anomaly_score > 0.5  # 超过50%权重
        
        return {
            'final_anomalies': final_anomalies,
            'trend_anomalies': trend_anomalies,
            'seasonal_anomalies': seasonal_anomalies,
            'residual_anomalies': residual_anomalies,
            'anomaly_score': anomaly_score,
            'decomposition': self.decomposition
        }
    
    def plot_decomposition_with_anomalies(self, data, results, figsize=(15, 12)):
        """绘制分解结果和异常检测"""
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 原始数据与最终异常
        axes[0].plot(data, label='Original Data', alpha=0.7)
        axes[0].scatter(np.where(results['final_anomalies'])[0], 
                       data[results['final_anomalies']], 
                       color='red', label='Detected Anomalies', s=50)
        axes[0].set_title('Original Data with Detected Anomalies')
        axes[0].legend()
        
        # 趋势与趋势异常
        trend = results['decomposition'].trend
        axes[1].plot(trend, label='Trend', color='blue')
        axes[1].scatter(np.where(results['trend_anomalies'])[0],
                       trend[results['trend_anomalies']],
                       color='orange', label='Trend Anomalies', s=30)
        axes[1].set_title('Trend Component')
        axes[1].legend()
        
        # 季节性与季节性异常
        seasonal = results['decomposition'].seasonal
        axes[2].plot(seasonal, label='Seasonal', color='green')
        axes[2].scatter(np.where(results['seasonal_anomalies'])[0],
                       seasonal[results['seasonal_anomalies']],
                       color='purple', label='Seasonal Anomalies', s=30)
        axes[2].set_title('Seasonal Component')
        axes[2].legend()
        
        # 残差与残差异常
        residual = results['decomposition'].resid
        axes[3].plot(residual, label='Residual', color='brown')
        axes[3].scatter(np.where(results['residual_anomalies'])[0],
                       residual[results['residual_anomalies']],
                       color='red', label='Residual Anomalies', s=30)
        axes[3].set_title('Residual Component')
        axes[3].legend()
        
        plt.tight_layout()
        return fig

def seasonal_anomaly_detection(data, period=144, weights=None):
    """
    季节性异常检测的便捷函数
    
    Args:
        data: 时间序列数据
        period: 季节性周期
        weights: 各组件权重
    
    Returns:
        检测结果字典
    """
    if weights is None:
        weights = {'trend': 0.3, 'seasonal': 0.3, 'residual': 0.4}
    
    detector = SeasonalAnomalyDetector(period=period)
    results = detector.detect_all_anomalies(data, weights)
    
    return results, detector

# 示例用法
if __name__ == "__main__":
    # 生成带季节性的示例数据
    np.random.seed(42)
    
    # 创建时间索引（1天，每10分钟一个点）
    timestamps = pd.date_range('2024-01-01', periods=144, freq='10min')
    
    # 生成季节性模式（日周期）
    time_hours = np.arange(144) * 10 / 60  # 转换为小时
    seasonal_pattern = 20 + 15 * np.sin(2 * np.pi * time_hours / 24)  # 日周期
    
    # 添加趋势
    trend = 0.02 * np.arange(144)
    
    # 添加噪声
    noise = np.random.normal(0, 2, 144)
    
    # 合成数据
    cpu_data = seasonal_pattern + trend + noise
    
    # 插入异常
    anomaly_indices = [30, 70, 110]
    cpu_data[anomaly_indices] = [60, 5, 65]  # 明显偏离季节性模式
    
    # 运行季节性异常检测
    results, detector = seasonal_anomaly_detection(cpu_data, period=144)
    
    # 绘制结果
    fig = detector.plot_decomposition_with_anomalies(cpu_data, results)
    plt.savefig('seasonal_methods/seasonal_decomposition_anomalies.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"检测到的异常总数: {results['final_anomalies'].sum()}")
    print(f"趋势异常: {results['trend_anomalies'].sum()}")
    print(f"季节性异常: {results['seasonal_anomalies'].sum()}")
    print(f"残差异常: {results['residual_anomalies'].sum()}")
    
    # 异常位置分析
    detected_indices = np.where(results['final_anomalies'])[0]
    true_indices = anomaly_indices
    
    print(f"真实异常位置: {true_indices}")
    print(f"检测异常位置: {detected_indices.tolist()}")
    
    # 计算检测精度
    tp = len(set(detected_indices) & set(true_indices))
    fp = len(set(detected_indices) - set(true_indices))
    fn = len(set(true_indices) - set(detected_indices))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1分数: {f1:.3f}")
