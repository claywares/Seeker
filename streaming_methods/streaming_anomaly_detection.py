"""
实时流数据异常检测
适用于在线数据流的增量式异常检测方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyAlert:
    """异常告警数据结构"""
    timestamp: datetime
    value: float
    anomaly_score: float
    method: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float

class StreamingAnomalyDetector:
    """流式异常检测器"""
    
    def __init__(self, window_size=100, methods=['zscore', 'ewma', 'isolation_forest']):
        """
        Args:
            window_size: 滑动窗口大小
            methods: 使用的检测方法列表
        """
        self.window_size = window_size
        self.methods = methods
        self.data_buffer = deque(maxlen=window_size)
        self.alerts = []
        
        # 统计信息
        self.running_mean = 0.0
        self.running_var = 0.0
        self.count = 0
        
        # EWMA参数
        self.ewm_mean = None
        self.ewm_std = None
        self.alpha = 0.1
        
        # Isolation Forest (简化版在线实现)
        self.isolation_scores = deque(maxlen=window_size)
        
        # 阈值配置
        self.thresholds = {
            'zscore': 3.0,
            'ewma': 2.5,
            'isolation_forest': 0.6
        }
        
        # 告警配置
        self.severity_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.9
        }
    
    def update_statistics(self, value: float):
        """增量更新统计信息"""
        self.count += 1
        if self.count == 1:
            self.running_mean = value
            self.running_var = 0.0
        else:
            # Welford在线算法
            delta = value - self.running_mean
            self.running_mean += delta / self.count
            delta2 = value - self.running_mean
            self.running_var += delta * delta2
    
    def zscore_detect(self, value: float) -> Tuple[bool, float]:
        """在线Z-score检测"""
        if self.count < 3:  # 需要最少样本
            return False, 0.0
        
        std = np.sqrt(self.running_var / (self.count - 1))
        if std == 0:
            return False, 0.0
        
        z_score = abs(value - self.running_mean) / std
        is_anomaly = z_score > self.thresholds['zscore']
        
        return is_anomaly, z_score / self.thresholds['zscore']
    
    def ewma_detect(self, value: float) -> Tuple[bool, float]:
        """在线EWMA检测"""
        if self.ewm_mean is None:
            self.ewm_mean = value
            self.ewm_std = 0.0
            return False, 0.0
        
        # 更新EWMA均值
        self.ewm_mean = self.alpha * value + (1 - self.alpha) * self.ewm_mean
        
        # 更新EWMA标准差
        squared_diff = (value - self.ewm_mean) ** 2
        if self.ewm_std == 0:
            self.ewm_std = np.sqrt(squared_diff)
        else:
            ewm_var = self.alpha * squared_diff + (1 - self.alpha) * (self.ewm_std ** 2)
            self.ewm_std = np.sqrt(ewm_var)
        
        if self.ewm_std == 0:
            return False, 0.0
        
        deviation = abs(value - self.ewm_mean) / self.ewm_std
        is_anomaly = deviation > self.thresholds['ewma']
        
        return is_anomaly, deviation / self.thresholds['ewma']
    
    def isolation_forest_detect(self, value: float) -> Tuple[bool, float]:
        """简化的在线孤立森林检测"""
        if len(self.data_buffer) < 10:
            return False, 0.0
        
        # 计算相对孤立度（简化版本）
        window_data = list(self.data_buffer)
        window_data.append(value)
        
        # 计算与窗口内其他点的平均距离
        distances = [abs(value - x) for x in window_data[:-1]]
        avg_distance = np.mean(distances)
        
        # 标准化距离
        std_distance = np.std(distances)
        if std_distance == 0:
            isolation_score = 0.0
        else:
            isolation_score = avg_distance / (std_distance + 1e-6)
        
        # 更新历史分数
        self.isolation_scores.append(isolation_score)
        
        # 计算相对异常程度
        if len(self.isolation_scores) >= 10:
            score_percentile = np.percentile(list(self.isolation_scores), 90)
            normalized_score = isolation_score / (score_percentile + 1e-6)
        else:
            normalized_score = 0.0
        
        is_anomaly = normalized_score > self.thresholds['isolation_forest']
        
        return is_anomaly, min(normalized_score, 2.0)  # 限制最大值
    
    def detect_anomaly(self, value: float, timestamp: datetime = None) -> Optional[AnomalyAlert]:
        """检测单个数据点的异常"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # 添加到缓冲区
        self.data_buffer.append(value)
        self.update_statistics(value)
        
        # 运行所有检测方法
        detection_results = {}
        
        if 'zscore' in self.methods:
            detection_results['zscore'] = self.zscore_detect(value)
        
        if 'ewma' in self.methods:
            detection_results['ewma'] = self.ewma_detect(value)
        
        if 'isolation_forest' in self.methods:
            detection_results['isolation_forest'] = self.isolation_forest_detect(value)
        
        # 综合判断
        anomaly_methods = []
        total_score = 0.0
        
        for method, (is_anomaly, score) in detection_results.items():
            if is_anomaly:
                anomaly_methods.append(method)
            total_score += score
        
        # 计算平均异常分数
        avg_score = total_score / len(detection_results) if detection_results else 0.0
        
        # 判断是否异常（至少一种方法检测到或平均分数较高）
        is_anomaly = len(anomaly_methods) > 0 or avg_score > 0.5
        
        if is_anomaly:
            # 确定严重性等级
            severity = 'LOW'
            for level, threshold in sorted(self.severity_thresholds.items(), 
                                         key=lambda x: x[1], reverse=True):
                if avg_score >= threshold:
                    severity = level
                    break
            
            # 计算置信度
            confidence = min(avg_score, 1.0)
            
            # 创建告警
            alert = AnomalyAlert(
                timestamp=timestamp,
                value=value,
                anomaly_score=avg_score,
                method='+'.join(anomaly_methods) if anomaly_methods else 'combined',
                severity=severity,
                confidence=confidence
            )
            
            self.alerts.append(alert)
            return alert
        
        return None
    
    def get_statistics(self) -> Dict:
        """获取检测器统计信息"""
        return {
            'processed_points': self.count,
            'buffer_size': len(self.data_buffer),
            'total_alerts': len(self.alerts),
            'running_mean': self.running_mean,
            'running_std': np.sqrt(self.running_var / (self.count - 1)) if self.count > 1 else 0,
            'ewm_mean': self.ewm_mean,
            'ewm_std': self.ewm_std,
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'value': alert.value,
                    'score': alert.anomaly_score,
                    'severity': alert.severity
                }
                for alert in self.alerts[-10:]  # 最近10个告警
            ]
        }

class StreamingSimulator:
    """流数据模拟器"""
    
    def __init__(self, detector: StreamingAnomalyDetector):
        self.detector = detector
        self.simulation_data = []
        self.alerts_history = []
    
    def generate_stream_data(self, duration_minutes=60, interval_seconds=10):
        """生成流式数据进行模拟"""
        start_time = datetime.now()
        current_time = start_time
        
        print(f"开始流式异常检测模拟，持续时间: {duration_minutes} 分钟")
        print("=" * 50)
        
        while (current_time - start_time).total_seconds() < duration_minutes * 60:
            # 生成模拟数据点
            base_value = 30 + 10 * np.sin(2 * np.pi * 
                        (current_time - start_time).total_seconds() / 3600)  # 小时周期
            noise = np.random.normal(0, 3)
            
            # 随机插入异常
            if np.random.random() < 0.02:  # 2% 概率
                if np.random.random() < 0.5:
                    value = base_value + np.random.uniform(30, 50)  # 高值异常
                else:
                    value = max(0, base_value - np.random.uniform(20, 30))  # 低值异常
            else:
                value = base_value + noise
            
            # 检测异常
            alert = self.detector.detect_anomaly(value, current_time)
            
            # 记录数据
            self.simulation_data.append({
                'timestamp': current_time,
                'value': value,
                'is_anomaly': alert is not None
            })
            
            if alert:
                self.alerts_history.append(alert)
                print(f"🚨 异常告警 | {alert.timestamp.strftime('%H:%M:%S')} | "
                      f"值: {alert.value:.2f} | 分数: {alert.anomaly_score:.3f} | "
                      f"严重性: {alert.severity} | 方法: {alert.method}")
            
            # 每10个点打印一次状态
            if len(self.simulation_data) % 10 == 0:
                stats = self.detector.get_statistics()
                print(f"⏱️  时间: {current_time.strftime('%H:%M:%S')} | "
                      f"处理点数: {stats['processed_points']} | "
                      f"总告警: {stats['total_alerts']} | "
                      f"当前均值: {stats['running_mean']:.2f}")
            
            current_time += timedelta(seconds=interval_seconds)
            time.sleep(0.1)  # 模拟实时处理延迟
        
        print("=" * 50)
        print("流式检测模拟完成")
    
    def plot_results(self):
        """绘制流式检测结果"""
        if not self.simulation_data:
            print("没有数据可绘制")
            return
        
        df = pd.DataFrame(self.simulation_data)
        
        plt.figure(figsize=(15, 8))
        
        # 绘制数据流
        plt.plot(df['timestamp'], df['value'], 
                label='Data Stream', alpha=0.7, linewidth=1)
        
        # 绘制异常点
        anomaly_df = df[df['is_anomaly']]
        if not anomaly_df.empty:
            plt.scatter(anomaly_df['timestamp'], anomaly_df['value'],
                       color='red', label='Detected Anomalies', s=50, zorder=5)
        
        # 添加告警信息
        for alert in self.alerts_history[-20:]:  # 显示最近20个告警
            plt.annotate(f'{alert.severity}\n{alert.anomaly_score:.2f}',
                        xy=(alert.timestamp, alert.value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8)
        
        plt.title('Real-time Streaming Anomaly Detection')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('streaming_methods/streaming_anomaly_detection.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印汇总统计
        stats = self.detector.get_statistics()
        print(f"\n📊 检测汇总:")
        print(f"总处理点数: {stats['processed_points']}")
        print(f"检测到异常: {stats['total_alerts']}")
        print(f"异常率: {stats['total_alerts']/stats['processed_points']*100:.2f}%")
        
        # 按严重性分类统计
        severity_counts = {}
        for alert in self.alerts_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        print(f"严重性分布: {severity_counts}")

# 示例用法
if __name__ == "__main__":
    # 创建流式检测器
    detector = StreamingAnomalyDetector(
        window_size=50,
        methods=['zscore', 'ewma', 'isolation_forest']
    )
    
    # 创建模拟器
    simulator = StreamingSimulator(detector)
    
    # 运行流式检测模拟（5分钟，每2秒一个数据点）
    simulator.generate_stream_data(duration_minutes=5, interval_seconds=2)
    
    # 绘制结果
    simulator.plot_results()
    
    # 输出最终统计
    final_stats = detector.get_statistics()
    print(f"\n📈 最终统计信息:")
    print(json.dumps(final_stats, indent=2, default=str))
