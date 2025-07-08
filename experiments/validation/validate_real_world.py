import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    
    def run_detection(self):
        """运行所有检测方法"""
        # Z-score: 降低阈值使其更敏感
        self.df['zscore_anomaly'] = zscore_detection(
            self.df['cpu_usage'], 
            threshold=2.7  # 从3降至2.5以提高敏感度
        )

        # IQR: 增加k值使其更保守
        self.df['iqr_anomaly'] = iqr_detection(
            self.df['cpu_usage'], 
            k=2.0  # 从1.5增至2.0以降低敏感度
        )

        # EWMA: 微调参数
        self.df['ewma_anomaly'] = ewma_detection(
            self.df['cpu_usage'],
            span=12,  # 适当调整窗口大小
            threshold=2.2  # 微调阈值
        )

        # Isolation Forest: 调整contamination
        self.df['iforest_anomaly'] = isolation_forest_detection(
            self.df['cpu_usage'],
            contamination=0.01  # 设置为期望的异常比例0.1%
        )

        # LOF: 调整参数
        self.df['lof_anomaly'] = lof_detection(
            self.df['cpu_usage'],
            n_neighbors=20,  # 增加邻居数量以提高稳定性
            contamination=0.01  # 与IForest保持一致
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
                               f'validation_results_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")
    
    def calculate_fpr(self):
        """计算各检测方法的False Positive Rate
        
        基于方法一致性判定基准真实值:
        - 若至少两种方法都检测为异常，认为是真实异常
        - 据此计算每种方法的误报率
        """
        # 基于方法一致性确定基准真实值
        method_agreement = (
            self.df['zscore_anomaly'].astype(int) +
            self.df['iqr_anomaly'].astype(int) +
            self.df['ewma_anomaly'].astype(int) +
            self.df['iforest_anomaly'].astype(int) +
            self.df['lof_anomaly'].astype(int)
        )
        base_truth = (method_agreement >= 2).astype(int)
        
        fpr_stats = {}
        methods = ['zscore', 'iqr', 'ewma', 'iforest', 'lof']
        
        for method in methods:
            predictions = self.df[f'{method}_anomaly']
            
            # 计算混淆矩阵指标
            false_positives = ((predictions == 1) & (base_truth == 0)).sum()
            true_negatives = ((predictions == 0) & (base_truth == 0)).sum()
            true_positives = ((predictions == 1) & (base_truth == 1)).sum()
            false_negatives = ((predictions == 0) & (base_truth == 1)).sum()
            
            # 计算各项指标
            fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            fpr_stats[method] = {
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'true_positives': true_positives,
                'false_negatives': false_negatives,
                'fpr': fpr,
                'precision': precision,
                'recall': recall
            }
        
        return fpr_stats

    def print_statistics(self):
        """打印检测统计信息"""
        print("\nDetection Statistics:")
        for method in ['zscore', 'iqr', 'ewma', 'iforest', 'lof']:
            anomaly_count = self.df[f'{method}_anomaly'].sum()
            total_points = len(self.df)
            percentage = (anomaly_count / total_points) * 100
            print(f"{method.upper()}: {anomaly_count} anomalies "
                  f"({percentage:.2f}% of total)")
        
        print("\nPerformance Analysis:")
        fpr_stats = self.calculate_fpr()
        for method, stats in fpr_stats.items():
            print(f"\n{method.upper()}:")
            print(f"  False Positives: {stats['false_positives']}")
            print(f"  True Negatives: {stats['true_negatives']}")
            print(f"  True Positives: {stats['true_positives']}")
            print(f"  False Negatives: {stats['false_negatives']}")
            print(f"  False Positive Rate: {stats['fpr']:.2%}")
            print(f"  Precision: {stats['precision']:.2%}")
            print(f"  Recall: {stats['recall']:.2%}")

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

if __name__ == '__main__':
    main()
