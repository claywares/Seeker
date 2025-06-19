import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_cpu_data(n_points=200, seed=42):
    """生成模拟的CPU使用率数据,包含已知的异常点"""
    # loc: 分布的均值(mean)，表示 CPU 使用率的平均水平为 30%
    # scale: 分布的标准差(std)，表示 CPU 使用率的波动范围
    # size: 生成的样本数量
    # 正常波动在20~40之间
    np.random.seed(seed)
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='min')
    cpu_usage = np.random.normal(loc=30, scale=5, size=n_points)  # 生成正态分布的CPU使用率数据

    # 插入异常点
    anomaly_indices = [50, 120, 180]
    cpu_usage[anomaly_indices] = [90, 5, 85]  # 设置明显的异常值

    # 创建DataFrame并标记真实异常
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'is_anomaly': 0
    })
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    return df

def zscore_detection(data, threshold=3):
    """使用Z-score方法检测异常"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

def iqr_detection(data, k=1.5):
    """使用IQR方法检测异常"""
    Q1 = np.percentile(data, 25)    # 第一四分位数
    Q3 = np.percentile(data, 75)    # 第三四分位数
    IQR = Q3 - Q1                   # 四分位距
    lower_bound = Q1 - k * IQR      # 下界
    upper_bound = Q3 + k * IQR      # 上界
    return (data < lower_bound) | (data > upper_bound)


def hybrid_detection(data, zscore_threshold=3, iqr_k=1.5):
    """结合Z-score和IQR方法进行异常检测"""
    zscore_anomalies = zscore_detection(data, zscore_threshold)
    iqr_anomalies = iqr_detection(data, iqr_k)
    return zscore_anomalies | iqr_anomalies  # 取并集

def plot_comparison(df):
    """Z-score、IQR和混合检测方法的图示比较"""
    # 使用所有方法检测异常情况
    df['zscore_anomaly'] = zscore_detection(df['cpu_usage'])
    df['iqr_anomaly'] = iqr_detection(df['cpu_usage'])
    df['hybrid_anomaly'] = hybrid_detection(df['cpu_usage'])

    # 创建分情节
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 绘制Z-score分数结果
    ax1.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage')
    ax1.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax1.scatter(df.loc[df['zscore_anomaly']==1, 'timestamp'],
                df.loc[df['zscore_anomaly']==1, 'cpu_usage'],
                color='orange', label='Z-score Detected', marker='o', facecolors='none', s=120)
    ax1.set_title('Z-score Detection (threshold=3)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.legend()

    # 绘制IQR结果图
    ax2.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage')
    ax2.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax2.scatter(df.loc[df['iqr_anomaly']==1, 'timestamp'],
                df.loc[df['iqr_anomaly']==1, 'cpu_usage'],
                color='green', label='IQR Detected', marker='o', facecolors='none', s=120)
    ax2.set_title('IQR Detection (k=1.5)')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.legend()

    # 绘制混合成果图
    ax3.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage')
    ax3.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax3.scatter(df.loc[df['hybrid_anomaly']==1, 'timestamp'],
                df.loc[df['hybrid_anomaly']==1, 'cpu_usage'],
                color='purple', label='Hybrid Detected', marker='o', facecolors='none', s=120)
    ax3.set_title('Hybrid Detection (Z-score + IQR)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CPU Usage (%)')
    ax3.legend()

    plt.tight_layout()

    # 保存图片到指定目录
    save_path = 'point_anomalies/CPU_anomaly_detection_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaving picture to...: {save_path}")

    # # 显示图片
    # plt.show()

def main():
    # Generate data
    df = generate_cpu_data()

    # Plot comparison
    plot_comparison(df)

    # Print detection statistics
    print("\nDetection Results:")
    print(f"True Anomalies: {df['is_anomaly'].sum()}")
    print(f"Z-score Detected: {df['zscore_anomaly'].sum()}")
    print(f"IQR Detected: {df['iqr_anomaly'].sum()}")
    print(f"Hybrid Detected: {df['hybrid_anomaly'].sum()}")

    # Print detection boundaries
    mean = np.mean(df['cpu_usage'])
    std = np.std(df['cpu_usage'])
    Q1 = np.percentile(df['cpu_usage'], 25)
    Q3 = np.percentile(df['cpu_usage'], 75)
    IQR = Q3 - Q1

    print("\nDetection Boundaries:")
    print(f"Z-score: [{mean-3*std:.2f}, {mean+3*std:.2f}]")
    print(f"IQR: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")

if __name__ == '__main__':
    main()
