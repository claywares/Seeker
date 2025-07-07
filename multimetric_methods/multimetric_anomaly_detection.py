"""
多维度与多指标异常检测
处理多个相关时间序列的联合异常检测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.cluster import DBSCAN
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MultiMetricAnomaly:
    """多指标异常结果"""
    timestamp: pd.Timestamp
    anomaly_score: float
    affected_metrics: List[str]
    anomaly_type: str  # 'correlation', 'outlier', 'pattern', 'combined'
    individual_scores: Dict[str, float]
    correlation_changes: Dict[str, float]

class MultiMetricAnomalyDetector:
    """多指标异常检测器"""
    
    def __init__(self, contamination=0.02):
        """
        Args:
            contamination: 预期异常比例
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.baseline_correlations = None
        self.isolation_forest = None
        self.robust_cov = None
        self.fitted = False
        
    def fit(self, data: pd.DataFrame):
        """
        训练多指标检测模型
        
        Args:
            data: 多指标数据DataFrame，每列为一个指标
        """
        # 数据预处理
        scaled_data = self.scaler.fit_transform(data.fillna(method='ffill').fillna(method='bfill'))
        
        # PCA降维分析
        self.pca.fit(scaled_data)
        
        # 训练Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.isolation_forest.fit(scaled_data)
        
        # 计算基线相关性
        self.baseline_correlations = data.corr()
        
        # 稳健协方差估计
        self.robust_cov = MinCovDet(contamination=self.contamination)
        self.robust_cov.fit(scaled_data)
        
        self.fitted = True
    
    def detect_correlation_anomalies(self, data: pd.DataFrame, window_size: int = 20) -> np.ndarray:
        """检测相关性异常"""
        if self.baseline_correlations is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        correlation_anomalies = np.zeros(len(data), dtype=bool)
        correlation_changes = []
        
        for i in range(window_size, len(data)):
            # 计算滑动窗口相关性
            window_data = data.iloc[i-window_size:i]
            window_corr = window_data.corr()
            
            # 计算与基线相关性的偏差
            corr_diff = np.abs(window_corr - self.baseline_correlations)
            avg_corr_change = np.mean(corr_diff.values[np.triu_indices_from(corr_diff, k=1)])
            correlation_changes.append(avg_corr_change)
            
            # 异常判定（使用自适应阈值）
            if i >= window_size * 2:
                recent_changes = correlation_changes[-window_size:]
                threshold = np.mean(recent_changes) + 2 * np.std(recent_changes)
                correlation_anomalies[i] = avg_corr_change > threshold
        
        return correlation_anomalies
    
    def detect_multivariate_outliers(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """检测多变量离群点"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 数据标准化
        scaled_data = self.scaler.transform(data.fillna(method='ffill').fillna(method='bfill'))
        
        # Isolation Forest检测
        isolation_scores = self.isolation_forest.decision_function(scaled_data)
        isolation_anomalies = self.isolation_forest.predict(scaled_data) == -1
        
        # 马氏距离检测
        try:
            mahalanobis_distances = self.robust_cov.mahalanobis(scaled_data)
            # 使用卡方分布阈值
            threshold = np.percentile(mahalanobis_distances, (1 - self.contamination) * 100)
            mahalanobis_anomalies = mahalanobis_distances > threshold
        except:
            mahalanobis_anomalies = np.zeros(len(data), dtype=bool)
        
        # 综合判定
        combined_anomalies = isolation_anomalies | mahalanobis_anomalies
        
        return combined_anomalies, isolation_scores
    
    def detect_pattern_anomalies(self, data: pd.DataFrame) -> np.ndarray:
        """检测模式异常"""
        # PCA变换
        scaled_data = self.scaler.transform(data.fillna(method='ffill').fillna(method='bfill'))
        pca_data = self.pca.transform(scaled_data)
        
        # 重构误差
        reconstructed = self.pca.inverse_transform(pca_data)
        reconstruction_errors = np.mean((scaled_data - reconstructed) ** 2, axis=1)
        
        # 异常判定
        threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
        pattern_anomalies = reconstruction_errors > threshold
        
        return pattern_anomalies
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        correlation_window: int = 20) -> List[MultiMetricAnomaly]:
        """
        综合检测多指标异常
        
        Args:
            data: 多指标数据
            correlation_window: 相关性分析窗口大小
        """
        if not self.fitted:
            self.fit(data)
        
        # 各类型异常检测
        correlation_anomalies = self.detect_correlation_anomalies(data, correlation_window)
        multivariate_anomalies, isolation_scores = self.detect_multivariate_outliers(data)
        pattern_anomalies = self.detect_pattern_anomalies(data)
        
        # 合并异常结果
        anomalies = []
        
        for i in range(len(data)):
            anomaly_types = []
            total_score = 0
            
            if correlation_anomalies[i]:
                anomaly_types.append('correlation')
                total_score += 0.3
            
            if multivariate_anomalies[i]:
                anomaly_types.append('outlier')
                total_score += 0.4
            
            if pattern_anomalies[i]:
                anomaly_types.append('pattern')
                total_score += 0.3
            
            if anomaly_types:
                # 计算个体指标分数
                individual_scores = {}
                for col in data.columns:
                    # 基于Z-score计算个体异常分数
                    col_data = data[col].iloc[max(0, i-20):i+1]
                    if len(col_data) > 3:
                        z_score = abs(data[col].iloc[i] - col_data.mean()) / (col_data.std() + 1e-6)
                        individual_scores[col] = min(z_score / 3, 1.0)
                    else:
                        individual_scores[col] = 0.0
                
                # 识别受影响的指标
                affected_metrics = [col for col, score in individual_scores.items() if score > 0.5]
                
                # 计算相关性变化
                correlation_changes = {}
                if i >= correlation_window:
                    window_data = data.iloc[i-correlation_window:i]
                    window_corr = window_data.corr()
                    for col in data.columns:
                        baseline_avg_corr = self.baseline_correlations[col].drop(col).mean()
                        window_avg_corr = window_corr[col].drop(col).mean()
                        correlation_changes[col] = abs(window_avg_corr - baseline_avg_corr)
                
                anomaly = MultiMetricAnomaly(
                    timestamp=data.index[i],
                    anomaly_score=min(total_score, 1.0),
                    affected_metrics=affected_metrics,
                    anomaly_type='+'.join(anomaly_types),
                    individual_scores=individual_scores,
                    correlation_changes=correlation_changes
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def plot_multimetric_analysis(self, data: pd.DataFrame, anomalies: List[MultiMetricAnomaly]):
        """可视化多指标分析结果"""
        fig = plt.figure(figsize=(16, 12))
        
        # 创建子图
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 主时间序列图
        ax1 = fig.add_subplot(gs[0, :])
        
        # 绘制所有指标
        colors = plt.cm.tab10(np.linspace(0, 1, len(data.columns)))
        for i, (col, color) in enumerate(zip(data.columns, colors)):
            ax1.plot(data.index, data[col], label=col, alpha=0.7, color=color)
        
        # 标记异常点
        if anomalies:
            for anomaly in anomalies:
                ax1.axvline(x=anomaly.timestamp, color='red', alpha=0.3, linestyle='--')
                
                # 在异常点位置添加标注
                y_pos = ax1.get_ylim()[1] * 0.9
                ax1.annotate(f'Score: {anomaly.anomaly_score:.2f}',
                           xy=(anomaly.timestamp, y_pos),
                           xytext=(10, -10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        ax1.set_title('Multi-Metric Time Series with Anomalies')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 相关性热力图
        ax2 = fig.add_subplot(gs[1, 0])
        sns.heatmap(self.baseline_correlations, annot=True, cmap='coolwarm', 
                   center=0, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Baseline Correlations')
        
        # 3. PCA解释方差
        ax3 = fig.add_subplot(gs[1, 1])
        if self.fitted:
            explained_variance = self.pca.explained_variance_ratio_
            cumsum_variance = np.cumsum(explained_variance)
            
            ax3.bar(range(1, len(explained_variance) + 1), explained_variance, 
                   alpha=0.7, label='Individual')
            ax3.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
                    'ro-', label='Cumulative')
            ax3.set_xlabel('Principal Component')
            ax3.set_ylabel('Explained Variance Ratio')
            ax3.set_title('PCA Explained Variance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 异常分类统计
        ax4 = fig.add_subplot(gs[2, :])
        
        if anomalies:
            # 统计异常类型
            anomaly_types = {}
            for anomaly in anomalies:
                for atype in anomaly.anomaly_type.split('+'):
                    anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
            
            # 绘制异常类型分布
            types = list(anomaly_types.keys())
            counts = list(anomaly_types.values())
            
            bars = ax4.bar(types, counts, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax4.set_title('Anomaly Type Distribution')
            ax4.set_ylabel('Count')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('multimetric_methods/multimetric_anomaly_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, anomalies: List[MultiMetricAnomaly]) -> str:
        """生成汇总报告"""
        if not anomalies:
            return "未检测到异常"
        
        report = []
        report.append("📊 多指标异常检测报告")
        report.append("=" * 60)
        report.append(f"检测到异常总数: {len(anomalies)}")
        report.append("")
        
        # 按类型统计
        type_stats = {}
        for anomaly in anomalies:
            for atype in anomaly.anomaly_type.split('+'):
                type_stats[atype] = type_stats.get(atype, 0) + 1
        
        report.append("异常类型分布:")
        for atype, count in type_stats.items():
            report.append(f"  {atype}: {count}")
        report.append("")
        
        # 受影响指标统计
        metric_impact = {}
        for anomaly in anomalies:
            for metric in anomaly.affected_metrics:
                metric_impact[metric] = metric_impact.get(metric, 0) + 1
        
        if metric_impact:
            report.append("受影响指标频次:")
            sorted_metrics = sorted(metric_impact.items(), key=lambda x: x[1], reverse=True)
            for metric, count in sorted_metrics:
                report.append(f"  {metric}: {count} 次")
        
        report.append("")
        
        # 严重异常点
        severe_anomalies = [a for a in anomalies if a.anomaly_score > 0.7]
        if severe_anomalies:
            report.append(f"严重异常点 (分数 > 0.7): {len(severe_anomalies)}")
            for anomaly in severe_anomalies[:5]:  # 显示前5个
                report.append(f"  {anomaly.timestamp}: 分数={anomaly.anomaly_score:.3f}, "
                            f"类型={anomaly.anomaly_type}, "
                            f"影响指标={','.join(anomaly.affected_metrics)}")
        
        return "\n".join(report)

# 示例用法
if __name__ == "__main__":
    # 生成多指标示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='10min')
    
    # 创建相关的多个指标
    base_cpu = 30 + 15 * np.sin(2 * np.pi * np.arange(200) / 144)
    cpu_usage = base_cpu + np.random.normal(0, 3, 200)
    
    # 内存使用率（与CPU正相关）
    memory_usage = 40 + 0.6 * cpu_usage + np.random.normal(0, 4, 200)
    
    # 网络I/O（与CPU负相关）
    network_io = 100 - 0.4 * cpu_usage + np.random.normal(0, 5, 200)
    
    # 磁盘I/O（独立变化）
    disk_io = 50 + 20 * np.sin(2 * np.pi * np.arange(200) / 100) + np.random.normal(0, 6, 200)
    
    # 插入多类型异常
    # 1. 单指标异常
    cpu_usage[50] = 90
    
    # 2. 多指标异常（相关性异常）
    cpu_usage[100] = 80
    memory_usage[100] = 30  # 破坏正相关性
    
    # 3. 模式异常
    cpu_usage[150:155] += 25
    memory_usage[150:155] += 30
    network_io[150:155] -= 20
    
    # 创建DataFrame
    data = pd.DataFrame({
        'CPU_Usage': cpu_usage,
        'Memory_Usage': memory_usage,
        'Network_IO': network_io,
        'Disk_IO': disk_io
    }, index=dates)
    
    # 创建多指标异常检测器
    detector = MultiMetricAnomalyDetector(contamination=0.05)
    
    # 检测异常
    anomalies = detector.detect_anomalies(data)
    
    # 生成报告
    print(detector.generate_summary_report(anomalies))
    print("\n" + "="*60)
    
    # 详细异常信息
    print("\n🔍 详细异常信息:")
    for i, anomaly in enumerate(anomalies[:10]):  # 显示前10个
        print(f"\n异常 {i+1}:")
        print(f"  时间: {anomaly.timestamp}")
        print(f"  分数: {anomaly.anomaly_score:.3f}")
        print(f"  类型: {anomaly.anomaly_type}")
        print(f"  影响指标: {anomaly.affected_metrics}")
        print(f"  个体分数: {anomaly.individual_scores}")
    
    # 可视化分析
    detector.plot_multimetric_analysis(data, anomalies)
    
    print(f"\n✅ 总共检测到 {len(anomalies)} 个多指标异常")
