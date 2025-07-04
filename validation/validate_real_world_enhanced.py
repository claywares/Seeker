import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入异常检测方法
from point_anomalies.main import (
    zscore_detection,
    iqr_detection,
    ewma_detection,
    isolation_forest_detection,
    lof_detection
)

class RealWorldValidator:
    """真实数据验证类"""
    
    def __init__(self, data_file):
        """
        Args:
            data_file: CSV数据文件路径
        """
        self.df = pd.read_csv(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.score_threshold = None  # Add this line to store threshold
        
    def run_detection(self):
        """运行所有检测方法并进行多重验证"""
        # 运行基础检测
        self.df['zscore_anomaly'] = zscore_detection(self.df['cpu_usage'], threshold=2.7)
        self.df['iqr_anomaly'] = iqr_detection(self.df['cpu_usage'], k=2.0)
        self.df['ewma_anomaly'] = ewma_detection(self.df['cpu_usage'], span=12, threshold=2.2)
        self.df['iforest_anomaly'] = isolation_forest_detection(self.df['cpu_usage'], contamination=0.01)
        self.df['lof_anomaly'] = lof_detection(self.df['cpu_usage'], n_neighbors=20, contamination=0.01)
        
        # 计算方法一致性分数
        self.df['method_agreement'] = (
            self.df['zscore_anomaly'].astype(int) +
            self.df['iqr_anomaly'].astype(int) +
            self.df['ewma_anomaly'].astype(int) +
            self.df['iforest_anomaly'].astype(int) +
            self.df['lof_anomaly'].astype(int)
        )
        
        # 计算偏离程度分数
        mean = self.df['cpu_usage'].mean()
        std = self.df['cpu_usage'].std()
        self.df['deviation_score'] = abs(self.df['cpu_usage'] - mean) / std
        
        # 计算持续性分数
        window_size = 3
        self.df['persistence_score'] = (
            self.df['iforest_anomaly']
            .rolling(window=window_size, center=True)
            .sum()
            .fillna(0)
        )
        
        # 综合评分
        self.df['anomaly_score'] = (
            0.4 * (self.df['method_agreement'] / 5) +  # 方法一致性 40%
            0.4 * (self.df['deviation_score'] / self.df['deviation_score'].max()) +  # 偏离程度 40%
            0.2 * (self.df['persistence_score'] / window_size)  # 持续性 20%
        )
        
        # 确定最终异常
        self.score_threshold = self.df['anomaly_score'].quantile(0.99)  # Change this line
        self.df['verified_anomaly'] = (
            (self.df['iforest_anomaly'] == 1) &  # 是IForest检测的异常
            (
                (self.df['method_agreement'] >= 2) |  # 至少两种方法检测到
                (self.df['anomaly_score'] > self.score_threshold)  # Change this line
            )
        )
        
        # 异常分级
        self.df['anomaly_severity'] = pd.cut(
            self.df[self.df['verified_anomaly']]['anomaly_score'],
            bins=3,
            labels=['P2', 'P1', 'P0']
        )
    
    def plot_results(self):
        """绘制检测结果对比图"""
        fig, axes = plt.subplots(5, 1, figsize=(12, 25))
        methods = [
            ('Z-score', 'zscore_anomaly', 'orange'),
            ('IQR', 'iqr_anomaly', 'green'),
            ('EWMA', 'ewma_anomaly', 'blue'),
            ('Isolation Forest', 'iforest_anomaly', 'purple'),
            ('LOF', 'lof_anomaly', 'brown')
        ]
        
        for (title, col, color), ax in zip(methods, axes):
            # 绘制基础CPU使用率线
            ax.plot(self.df['timestamp'], self.df['cpu_usage'], 
                   label='CPU Usage', color='grey', alpha=0.5)
            
            # 绘制检测到的异常点
            anomalies = self.df[self.df[col] == 1]
            ax.scatter(anomalies['timestamp'], anomalies['cpu_usage'],
                      color=color, label=f'{title} Detection',
                      marker='o', facecolors='none', s=120)
            
            ax.set_title(f'{title} Detection Results')
            ax.set_ylabel('CPU Usage (%)')
            ax.legend()
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        
        # 保存图片
        timestamp = pd.Timestamp.now().strftime("%Y%m%d")
        save_path = os.path.join(project_root, 'validation', 
                               f'enhanced_validation_results_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")
    
    def print_statistics(self):
        """打印检测统计信息"""
        print("\nDetection Statistics:")
        for method in ['zscore', 'iqr', 'ewma', 'iforest', 'lof']:
            anomaly_count = self.df[f'{method}_anomaly'].sum()
            total_points = len(self.df)
            percentage = (anomaly_count / total_points) * 100
            print(f"{method.upper()}: {anomaly_count} anomalies "
                  f"({percentage:.2f}% of total)")
    
    def plot_enhanced_results(self):
        """绘制增强版检测结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 绘制原始数据和所有异常点
        ax1.plot(self.df['timestamp'], self.df['cpu_usage'], 
                label='CPU Usage', color='grey', alpha=0.5)
        
        # 绘制不同优先级的异常点
        for severity, color in zip(['P0', 'P1', 'P2'], ['red', 'orange', 'yellow']):
            mask = (self.df['verified_anomaly']) & (self.df['anomaly_severity'] == severity)
            if mask.any():
                ax1.scatter(
                    self.df[mask]['timestamp'],
                    self.df[mask]['cpu_usage'],
                    color=color,
                    label=f'Priority {severity}',
                    marker='*' if severity == 'P0' else 'o',
                    s=200 if severity == 'P0' else 120
                )
        
        ax1.set_title('Enhanced Anomaly Detection Results')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.legend()
        
        # 绘制异常评分
        ax2.plot(self.df['timestamp'], self.df['anomaly_score'], 
                label='Anomaly Score', color='blue', alpha=0.7)
        ax2.axhline(y=self.score_threshold, color='red', linestyle='--',  # Change this line
                   label='Score Threshold')
        ax2.set_title('Anomaly Scores')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Score')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = pd.Timestamp.now().strftime("%Y%m%d")
        save_path = os.path.join(project_root, 'validation', 
                               f'enhanced_results_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEnhanced results saved to: {save_path}")
    
    def calculate_fpr(self):
        """计算各检测方法的误报率(False Positive Rate)
        
        FPR = False Positives / (False Positives + True Negatives)
        由于我们没有真实标注，我们使用以下方法估算：
        1. 使用 verified_anomaly 作为基准真实值
        2. 计算每种方法相对于这个基准的误报率
        """
        base_truth = self.df['verified_anomaly']
        
        fpr_stats = {}
        methods = ['zscore', 'iqr', 'ewma', 'iforest', 'lof']
        
        for method in methods:
            predictions = self.df[f'{method}_anomaly']
            
            # 计算各个指标
            false_positives = ((predictions == 1) & (base_truth == 0)).sum()
            true_negatives = ((predictions == 0) & (base_truth == 0)).sum()
            
            # 计算FPR
            fpr = false_positives / (false_positives + true_negatives)
            
            # 存储结果
            fpr_stats[method] = {
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'fpr': fpr
            }
        
        return fpr_stats
    
    def print_enhanced_statistics(self):
        """打印增强版统计信息"""
        print("\nEnhanced Detection Statistics:")
        print(f"Total anomalies detected by IForest: {self.df['iforest_anomaly'].sum()}")
        print(f"Verified anomalies: {self.df['verified_anomaly'].sum()}")
        print("\nAnomaly Priority Distribution:")
        priority_counts = self.df[self.df['verified_anomaly']]['anomaly_severity'].value_counts()
        for priority, count in priority_counts.items():
            print(f"Priority {priority}: {count}")
        
        print("\nFalse Positive Rate Analysis:")
        fpr_stats = self.calculate_fpr()
        for method, stats in fpr_stats.items():
            print(f"\n{method.upper()}:")
            print(f"  False Positives: {stats['false_positives']}")
            print(f"  True Negatives: {stats['true_negatives']}")
            print(f"  False Positive Rate: {stats['fpr']:.2%}")

def main():
    """主函数"""
    # 获取最新的数据文件
    data_dir = os.path.join(project_root, 'data')
    data_files = [f for f in os.listdir(data_dir) if f.startswith('CPUUtilization_')]
    if not data_files:
        print("Error: No data files found in data directory")
        return
    
    latest_file = max(data_files, key=lambda x: os.path.getctime(
        os.path.join(data_dir, x)))
    data_path = os.path.join(data_dir, latest_file)
    
    # 运行验证
    validator = RealWorldValidator(data_path)
    validator.run_detection()
    validator.print_statistics()
    validator.plot_results()
    validator.plot_enhanced_results()
    validator.print_enhanced_statistics()

if __name__ == '__main__':
    main()
