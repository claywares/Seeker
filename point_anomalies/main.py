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

def ewma_detection(data, span=15, threshold=2):
    """使用EWMA方法检测异常
    
    参数:
        data: 输入的时间序列数据
        span: 移动窗口大小（改小以提高敏感度）
        threshold: 异常判定的阈值（改小以降低检测门槛）
    """
    ewm_mean = pd.Series(data).ewm(span=span).mean()
    ewm_std = pd.Series(data).ewm(span=span).std()
    diff = np.abs(data - ewm_mean)
    threshold_values = threshold * ewm_std
    
    # 打印调试信息
    print("\nEWMA 检测调试信息:")
    print(f"最大偏差值: {diff.max():.2f}")
    print(f"平均偏差值: {diff.mean():.2f}")
    print(f"最大阈值: {threshold_values.max():.2f}")
    print(f"平均阈值: {threshold_values.mean():.2f}")
    
    return diff > threshold_values

def hybrid_detection(data, zscore_threshold=3, iqr_k=1.5):
    """结合Z-score和IQR方法进行异常检测"""
    zscore_anomalies = zscore_detection(data, zscore_threshold)
    iqr_anomalies = iqr_detection(data, iqr_k)
    return zscore_anomalies | iqr_anomalies  # 取并集

def plot_comparison(df):
    """Z-score、IQR和EWMA检测方法的图示比较"""
    # 使用三种方法检测异常
    df['zscore_anomaly'] = zscore_detection(df['cpu_usage'])
    df['iqr_anomaly'] = iqr_detection(df['cpu_usage'])
    df['ewma_anomaly'] = ewma_detection(df['cpu_usage'])
    # df['hybrid_anomaly'] = hybrid_detection(df['cpu_usage'])  # 暂时注释掉

    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 绘制Z-score检测结果
    ax1.plot(df['timestamp'], df['cpu_usage'], label='CPU Utilization')
    ax1.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax1.scatter(df.loc[df['zscore_anomaly']==1, 'timestamp'],
                df.loc[df['zscore_anomaly']==1, 'cpu_usage'],
                color='orange', label='Z-score Detection', marker='o', facecolors='none', s=120)
    ax1.set_title('Z-score Detection (Threshold=3)')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.legend()

    # 绘制IQR检测结果
    ax2.plot(df['timestamp'], df['cpu_usage'], label='CPU Utilization')
    ax2.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax2.scatter(df.loc[df['iqr_anomaly']==1, 'timestamp'],
                df.loc[df['iqr_anomaly']==1, 'cpu_usage'],
                color='green', label='IQR Detection', marker='o', facecolors='none', s=120)
    ax2.set_title('IQR Detection (k=1.5)')
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.legend()

    # 绘制EWMA检测结果
    ax3.plot(df['timestamp'], df['cpu_usage'], label='CPU Utilization')
    ax3.scatter(df.loc[df['is_anomaly']==1, 'timestamp'],
                df.loc[df['is_anomaly']==1, 'cpu_usage'],
                color='red', label='True Anomaly', marker='x', s=100)
    ax3.scatter(df.loc[df['ewma_anomaly']==1, 'timestamp'],
                df.loc[df['ewma_anomaly']==1, 'cpu_usage'],
                color='blue', label='EWMA Detection', marker='o', facecolors='none', s=120)
    ax3.set_title('EWMA Detection (span=20, Threshold=3)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CPU Utilization (%)')
    ax3.legend()

    plt.tight_layout()

    # 保存图片到指定目录
    save_path = 'point_anomalies/CPU_anomaly_detection_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {save_path}")

def main():
    # 生成数据
    df = generate_cpu_data()

    # 绘制对比图
    plot_comparison(df)

    # 打印检测统计信息
    print("\n检测结果统计：")
    print(f"真实异常数量: {df['is_anomaly'].sum()}")
    print(f"Z-score检测数量: {df['zscore_anomaly'].sum()}")
    print(f"IQR检测数量: {df['iqr_anomaly'].sum()}")
    print(f"EWMA检测数量: {df['ewma_anomaly'].sum()}")
    # print(f"混合检测数量: {df['hybrid_anomaly'].sum()}")  # 暂时注释掉

    # 打印检测边界
    mean = np.mean(df['cpu_usage'])
    std = np.std(df['cpu_usage'])
    Q1 = np.percentile(df['cpu_usage'], 25)
    Q3 = np.percentile(df['cpu_usage'], 75)
    IQR = Q3 - Q1

    # print("\n检测边界值：")
    # print(f"Z-score边界: [{mean-3*std:.2f}, {mean+3*std:.2f}]")
    # print(f"IQR边界: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")

if __name__ == '__main__':
    main()
