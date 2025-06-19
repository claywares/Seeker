import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 生成模拟CPU使用率数据
np.random.seed(42)
n_points = 200
timestamps = pd.date_range('2025-06-19', periods=n_points, freq='min')

# loc: 分布的均值(mean)，表示 CPU 使用率的平均水平为 30%
# scale: 分布的标准差(std)，表示 CPU 使用率的波动范围
# size: 生成的样本数量
# 正常波动在20~40之间
cpu_usage = np.random.normal(loc=30, scale=5, size=n_points)

# 2. 插入3个异常点
anomaly_indices = [50, 120, 180]
cpu_usage[anomaly_indices] = [90, 5, 85]  # 明显异常, 超出标准差

# 3. 构建DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'cpu_usage': cpu_usage,
    'is_anomaly': 0
})
df.loc[anomaly_indices, 'is_anomaly'] = 1

# 4. 用Z-score检测异常
mean = df['cpu_usage'].mean()
std = df['cpu_usage'].std()
z_scores = np.abs((df['cpu_usage'] - mean) / std)
threshold = 3  # 通常3倍标准差
df['detected_anomaly'] = (z_scores > threshold).astype(int)

# 5. 可视化
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage')
plt.scatter(df.loc[df['is_anomaly']==1, 'timestamp'], 
            df.loc[df['is_anomaly']==1, 'cpu_usage'], 
            color='red', label='True Anomaly', marker='x', s=100)
plt.scatter(df.loc[df['detected_anomaly']==1, 'timestamp'], 
            df.loc[df['detected_anomaly']==1, 'cpu_usage'], 
            color='orange', label='Detected Anomaly', marker='o', facecolors='none', s=120)
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage with Point Anomalies')
plt.legend()
plt.tight_layout()
plt.show()
