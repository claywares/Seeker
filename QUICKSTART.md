# Seeker 快速开始指南

## 🚀 立即开始

### 1. 运行演示
```bash
python main.py --demo
```

### 2. 检测您的数据
```bash
# CSV 文件
python main.py --data your_data.csv

# 指定检测方法
python main.py --data your_data.csv --methods zscore iqr ewma isolation_forest

# 自定义输出目录
python main.py --data your_data.csv --output my_results
```

## 📊 编程接口

### 简单检测
```python
import numpy as np
import pandas as pd
from seeker_framework import quick_detect

# 生成示例数据
data = np.random.normal(30, 5, 200)
data[50] = 90  # 插入异常

# 快速检测
results = quick_detect(data, methods=['zscore', 'isolation_forest'])
```

### 高级使用
```python
from seeker_framework import SeekerAnomalyDetector, SeekerConfig

# 自定义配置
config = SeekerConfig(
    zscore_threshold=2.5,
    isolation_forest_contamination=0.02,
    enable_visualization=True
)

# 创建检测器
detector = SeekerAnomalyDetector(config)

# 执行检测
results = detector.detect_single_metric(data, methods=['zscore', 'iqr', 'ewma'])

# 生成可视化
detector.plot_comprehensive_analysis(results)

# 生成报告
reports = detector.generate_report(results)
```

### 多指标检测
```python
from seeker_framework import SeekerAnomalyDetector

# 多指标数据
data = pd.DataFrame({
    'cpu_usage': np.random.normal(30, 5, 200),
    'memory_usage': np.random.normal(60, 10, 200),
    'network_io': np.random.normal(100, 20, 200)
})

detector = SeekerAnomalyDetector()
results = detector.detect_multi_metric(data)
```

### 流式检测
```python
from seeker_framework import SeekerAnomalyDetector

detector = SeekerAnomalyDetector()
streaming_detector = detector.create_streaming_detector(window_size=50)

# 逐点检测
for value in data_stream:
    alert = streaming_detector.detect_anomaly(value)
    if alert:
        print(f"异常: {alert.value} (分数: {alert.anomaly_score})")
```

## 📁 检测方法

### 基础统计方法
- **zscore**: 基于标准分数的异常检测
- **iqr**: 基于四分位距的异常检测  
- **ewma**: 指数加权移动平均异常检测

### 机器学习方法
- **isolation_forest**: 孤立森林异常检测
- **lof**: 局部离群因子异常检测

### 高级方法 (需要额外依赖)
- **lstm_autoencoder**: LSTM自编码器异常检测
- **seasonal**: 时间序列分解与季节性异常检测

## 🔧 配置选项

```python
config = SeekerConfig(
    # 统计方法参数
    zscore_threshold=3.0,           # Z-score阈值
    iqr_k=1.5,                     # IQR倍数
    ewma_span=15,                  # EWMA窗口大小
    ewma_threshold=2.0,            # EWMA阈值
    
    # 机器学习参数
    isolation_forest_contamination=0.02,  # 异常比例
    lof_n_neighbors=20,                   # LOF邻居数
    lof_contamination=0.02,               # LOF异常比例
    
    # 深度学习参数
    lstm_sequence_length=30,        # LSTM序列长度
    lstm_encoding_dim=10,          # 编码维度
    lstm_epochs=50,                # 训练轮数
    
    # 输出控制
    enable_visualization=True,      # 启用可视化
    enable_reports=True,           # 启用报告
    output_dir="results"           # 输出目录
)
```

## 📈 结果解释

### DetectionResult 结构
```python
@dataclass
class DetectionResult:
    method: str                    # 检测方法名
    anomalies: np.ndarray         # 异常位置 (布尔数组)
    scores: Optional[np.ndarray]  # 异常分数 (可选)
    parameters: Optional[Dict]    # 使用的参数
    execution_time: Optional[float]  # 执行时间
    metadata: Optional[Dict]      # 额外信息
```

### 获取异常位置
```python
# 获取异常索引
anomaly_indices = np.where(result.anomalies)[0]

# 获取异常值
anomaly_values = data[result.anomalies]

# 异常统计
anomaly_count = result.anomalies.sum()
anomaly_rate = anomaly_count / len(result.anomalies) * 100
```

## 🎯 最佳实践

1. **数据准备**
   - 确保数据质量，处理缺失值
   - 考虑数据的时间特性和业务背景

2. **方法选择**
   - 对于简单数据：zscore、iqr
   - 对于复杂模式：isolation_forest、lof
   - 对于时间序列：ewma、seasonal
   - 对于高维数据：lstm_autoencoder

3. **参数调优**
   - 从默认参数开始
   - 根据业务需求调整敏感度
   - 使用集成方法提高鲁棒性

4. **结果验证**
   - 结合业务知识验证异常
   - 使用多种方法交叉验证
   - 分析异常的时间分布和模式

## 🔗 扩展功能

- **根因分析**: 自动解释异常原因
- **多指标联合**: 检测多个相关指标的异常
- **实时监控**: 支持数据流实时异常检测
- **自定义方法**: 易于集成新的检测算法
