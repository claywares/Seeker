"""
Seeker 异常检测集成框架
统一管理和调用所有异常检测方法的主控制器
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入各种检测方法
from point_anomalies.main import (
    zscore_detection, iqr_detection, ewma_detection,
    isolation_forest_detection, lof_detection
)

# 导入新开发的高级方法
try:
    from deep_learning_methods.lstm_autoencoder import lstm_autoencoder_detection
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("警告: 深度学习方法不可用，请安装tensorflow")

try:
    from seasonal_methods.seasonal_anomaly_detection import seasonal_anomaly_detection
    SEASONAL_AVAILABLE = True
except ImportError:
    SEASONAL_AVAILABLE = False
    print("警告: 季节性检测方法不可用，请安装statsmodels")

try:
    from streaming_methods.streaming_anomaly_detection import StreamingAnomalyDetector
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

try:
    from root_cause_analysis.anomaly_explanation import RootCauseAnalyzer
    ROOT_CAUSE_AVAILABLE = True
except ImportError:
    ROOT_CAUSE_AVAILABLE = False

try:
    from multimetric_methods.multimetric_anomaly_detection import MultiMetricAnomalyDetector
    MULTIMETRIC_AVAILABLE = True
except ImportError:
    MULTIMETRIC_AVAILABLE = False

@dataclass
class DetectionResult:
    """检测结果统一数据结构"""
    method: str
    anomalies: np.ndarray  # 布尔数组，标记异常位置
    scores: Optional[np.ndarray] = None  # 异常分数
    parameters: Optional[Dict] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict] = None

@dataclass
class SeekerConfig:
    """Seeker配置类"""
    # 基础方法参数
    zscore_threshold: float = 3.0
    iqr_k: float = 1.5
    ewma_span: int = 15
    ewma_threshold: float = 2.0
    
    # 机器学习方法参数
    isolation_forest_contamination: float = 0.02
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.02
    
    # 深度学习参数
    lstm_sequence_length: int = 30
    lstm_encoding_dim: int = 10
    lstm_epochs: int = 50
    
    # 季节性检测参数
    seasonal_period: int = 144  # 日周期，10分钟间隔
    seasonal_weights: Dict[str, float] = None
    
    # 输出控制
    enable_visualization: bool = True
    enable_reports: bool = True
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.seasonal_weights is None:
            self.seasonal_weights = {'trend': 0.3, 'seasonal': 0.3, 'residual': 0.4}

class SeekerAnomalyDetector:
    """Seeker 主异常检测器"""
    
    def __init__(self, config: Optional[SeekerConfig] = None):
        """
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or SeekerConfig()
        self.results = {}
        self.data = None
        self.root_cause_analyzer = None
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        # 初始化组件
        if ROOT_CAUSE_AVAILABLE:
            self.root_cause_analyzer = RootCauseAnalyzer()
    
    def detect_single_metric(self, data: Union[pd.Series, np.ndarray], 
                           methods: List[str] = None,
                           enable_explanation: bool = False) -> Dict[str, DetectionResult]:
        """
        单指标异常检测
        
        Args:
            data: 时间序列数据
            methods: 使用的检测方法列表
            enable_explanation: 是否启用异常解释
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'ewma', 'isolation_forest', 'lof']
        
        # 数据预处理
        if isinstance(data, pd.Series):
            self.data = data
            values = data.values
        else:
            values = data
            self.data = pd.Series(values)
        
        results = {}
        
        # 基础统计方法
        if 'zscore' in methods:
            start_time = datetime.now()
            anomalies = zscore_detection(values, self.config.zscore_threshold)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results['zscore'] = DetectionResult(
                method='zscore',
                anomalies=anomalies,
                parameters={'threshold': self.config.zscore_threshold},
                execution_time=execution_time
            )
        
        if 'iqr' in methods:
            start_time = datetime.now()
            anomalies = iqr_detection(values, self.config.iqr_k)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results['iqr'] = DetectionResult(
                method='iqr',
                anomalies=anomalies,
                parameters={'k': self.config.iqr_k},
                execution_time=execution_time
            )
        
        if 'ewma' in methods:
            start_time = datetime.now()
            anomalies = ewma_detection(values, self.config.ewma_span, self.config.ewma_threshold)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results['ewma'] = DetectionResult(
                method='ewma',
                anomalies=anomalies,
                parameters={'span': self.config.ewma_span, 'threshold': self.config.ewma_threshold},
                execution_time=execution_time
            )
        
        # 机器学习方法
        if 'isolation_forest' in methods:
            start_time = datetime.now()
            anomalies = isolation_forest_detection(values, self.config.isolation_forest_contamination)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results['isolation_forest'] = DetectionResult(
                method='isolation_forest',
                anomalies=anomalies,
                parameters={'contamination': self.config.isolation_forest_contamination},
                execution_time=execution_time
            )
        
        if 'lof' in methods:
            start_time = datetime.now()
            anomalies = lof_detection(values, self.config.lof_n_neighbors, self.config.lof_contamination)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results['lof'] = DetectionResult(
                method='lof',
                anomalies=anomalies,
                parameters={'n_neighbors': self.config.lof_n_neighbors, 'contamination': self.config.lof_contamination},
                execution_time=execution_time
            )
        
        # 深度学习方法
        if 'lstm_autoencoder' in methods and DEEP_LEARNING_AVAILABLE:
            start_time = datetime.now()
            try:
                anomalies, scores, detector = lstm_autoencoder_detection(
                    values,
                    sequence_length=self.config.lstm_sequence_length,
                    encoding_dim=self.config.lstm_encoding_dim,
                    epochs=self.config.lstm_epochs
                )
                execution_time = (datetime.now() - start_time).total_seconds()
                
                results['lstm_autoencoder'] = DetectionResult(
                    method='lstm_autoencoder',
                    anomalies=anomalies,
                    scores=scores,
                    parameters={
                        'sequence_length': self.config.lstm_sequence_length,
                        'encoding_dim': self.config.lstm_encoding_dim,
                        'epochs': self.config.lstm_epochs
                    },
                    execution_time=execution_time
                )
            except Exception as e:
                print(f"LSTM自编码器检测失败: {e}")
        
        # 季节性异常检测
        if 'seasonal' in methods and SEASONAL_AVAILABLE:
            start_time = datetime.now()
            try:
                seasonal_results, detector = seasonal_anomaly_detection(
                    values,
                    period=self.config.seasonal_period,
                    weights=self.config.seasonal_weights
                )
                execution_time = (datetime.now() - start_time).total_seconds()
                
                results['seasonal'] = DetectionResult(
                    method='seasonal',
                    anomalies=seasonal_results['final_anomalies'],
                    scores=seasonal_results['anomaly_score'],
                    parameters={
                        'period': self.config.seasonal_period,
                        'weights': self.config.seasonal_weights
                    },
                    execution_time=execution_time,
                    metadata=seasonal_results
                )
            except Exception as e:
                print(f"季节性检测失败: {e}")
        
        # 异常解释（如果启用且有异常）
        if enable_explanation and ROOT_CAUSE_AVAILABLE:
            self._add_explanations(results)
        
        self.results = results
        return results
    
    def detect_multi_metric(self, data: pd.DataFrame, 
                          contamination: float = None) -> Dict[str, Any]:
        """
        多指标异常检测
        
        Args:
            data: 多指标数据DataFrame
            contamination: 预期异常比例
        """
        if not MULTIMETRIC_AVAILABLE:
            raise ImportError("多指标检测不可用，请检查依赖")
        
        contamination = contamination or self.config.isolation_forest_contamination
        
        start_time = datetime.now()
        detector = MultiMetricAnomalyDetector(contamination=contamination)
        anomalies = detector.detect_anomalies(data)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 转换为统一格式
        anomaly_mask = np.zeros(len(data), dtype=bool)
        anomaly_scores = np.zeros(len(data))
        
        for anomaly in anomalies:
            if isinstance(anomaly.timestamp, (int, np.integer)):
                idx = anomaly.timestamp
            else:
                try:
                    idx = data.index.get_loc(anomaly.timestamp)
                except:
                    continue
            
            if 0 <= idx < len(data):
                anomaly_mask[idx] = True
                anomaly_scores[idx] = anomaly.anomaly_score
        
        return {
            'detector': detector,
            'anomalies': anomalies,
            'anomaly_mask': anomaly_mask,
            'anomaly_scores': anomaly_scores,
            'execution_time': execution_time
        }
    
    def create_streaming_detector(self, window_size: int = 100, 
                                methods: List[str] = None) -> 'StreamingAnomalyDetector':
        """
        创建流式检测器
        
        Args:
            window_size: 滑动窗口大小
            methods: 使用的检测方法
        """
        if not STREAMING_AVAILABLE:
            raise ImportError("流式检测不可用")
        
        if methods is None:
            methods = ['zscore', 'ewma', 'isolation_forest']
        
        return StreamingAnomalyDetector(window_size=window_size, methods=methods)
    
    def _add_explanations(self, results: Dict[str, DetectionResult]):
        """为检测结果添加解释"""
        if not self.root_cause_analyzer or self.data is None:
            return
        
        # 收集所有异常点
        all_anomaly_indices = set()
        for result in results.values():
            anomaly_indices = np.where(result.anomalies)[0]
            all_anomaly_indices.update(anomaly_indices)
        
        # 为主要异常点生成解释
        explanations = {}
        for idx in list(all_anomaly_indices)[:10]:  # 限制解释数量
            try:
                explanation = self.root_cause_analyzer.analyze_anomaly(self.data, idx)
                explanations[idx] = explanation
            except Exception as e:
                print(f"解释生成失败 (索引 {idx}): {e}")
        
        # 将解释添加到结果中
        for result in results.values():
            if result.metadata is None:
                result.metadata = {}
            result.metadata['explanations'] = explanations
    
    def ensemble_detection(self, results: Dict[str, DetectionResult], 
                          voting_threshold: float = 0.3) -> DetectionResult:
        """
        集成多种方法的检测结果
        
        Args:
            results: 各方法的检测结果
            voting_threshold: 投票阈值
        """
        if not results:
            raise ValueError("没有检测结果可用于集成")
        
        # 获取数据长度
        data_length = len(next(iter(results.values())).anomalies)
        
        # 投票集成
        vote_matrix = np.zeros((len(results), data_length))
        method_names = []
        
        for i, (method, result) in enumerate(results.items()):
            vote_matrix[i] = result.anomalies.astype(int)
            method_names.append(method)
        
        # 计算投票分数
        vote_scores = np.mean(vote_matrix, axis=0)
        ensemble_anomalies = vote_scores >= voting_threshold
        
        return DetectionResult(
            method='ensemble',
            anomalies=ensemble_anomalies,
            scores=vote_scores,
            parameters={'voting_threshold': voting_threshold, 'methods': method_names},
            metadata={'vote_matrix': vote_matrix, 'method_names': method_names}
        )
    
    def plot_comprehensive_analysis(self, results: Dict[str, DetectionResult], 
                                  title: str = "Seeker Comprehensive Anomaly Analysis"):
        """绘制综合分析图"""
        if not self.config.enable_visualization:
            return
        
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(15, 4 * (n_methods + 1)))
        
        if n_methods == 1:
            axes = [axes]
        
        # 原始数据
        data_values = self.data.values if self.data is not None else np.arange(len(next(iter(results.values())).anomalies))
        
        # 为每种方法绘制子图
        for i, (method, result) in enumerate(results.items()):
            ax = axes[i]
            
            # 绘制数据
            ax.plot(data_values, alpha=0.7, label='Data', linewidth=1)
            
            # 标记异常点
            anomaly_indices = np.where(result.anomalies)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, data_values[anomaly_indices],
                          color='red', s=50, alpha=0.8, label='Anomalies')
            
            # 添加分数（如果有）
            if result.scores is not None:
                ax2 = ax.twinx()
                ax2.plot(result.scores, color='orange', alpha=0.5, label='Scores')
                ax2.set_ylabel('Anomaly Score', color='orange')
            
            ax.set_title(f'{method.upper()} Detection (Count: {result.anomalies.sum()})')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # 集成结果
        if len(results) > 1:
            ensemble_result = self.ensemble_detection(results)
            ax = axes[-1]
            
            ax.plot(data_values, alpha=0.7, label='Data', linewidth=1)
            
            anomaly_indices = np.where(ensemble_result.anomalies)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, data_values[anomaly_indices],
                          color='purple', s=70, alpha=0.8, label='Ensemble Anomalies')
            
            # 投票分数
            ax2 = ax.twinx()
            ax2.plot(ensemble_result.scores, color='green', alpha=0.6, label='Vote Scores')
            ax2.set_ylabel('Vote Score', color='green')
            ax2.axhline(y=ensemble_result.parameters['voting_threshold'], 
                       color='green', linestyle='--', alpha=0.5, label='Threshold')
            
            ax.set_title(f'Ensemble Detection (Count: {ensemble_result.anomalies.sum()})')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config.output_dir, 'comprehensive_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"综合分析图已保存至: {save_path}")
        
        if self.config.enable_visualization:
            plt.show()
    
    def generate_report(self, results: Dict[str, DetectionResult], 
                       output_format: str = 'both') -> Dict[str, str]:
        """
        生成检测报告
        
        Args:
            results: 检测结果
            output_format: 输出格式 ('json', 'text', 'both')
        """
        if not self.config.enable_reports:
            return {}
        
        # 生成报告内容
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'data_length': len(next(iter(results.values())).anomalies),
            'methods_used': list(results.keys()),
            'summary': {},
            'performance': {},
            'detailed_results': {}
        }
        
        # 汇总统计
        for method, result in results.items():
            anomaly_count = result.anomalies.sum()
            anomaly_rate = anomaly_count / len(result.anomalies) * 100
            
            report_data['summary'][method] = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': f"{anomaly_rate:.2f}%",
                'parameters': result.parameters
            }
            
            if result.execution_time:
                report_data['performance'][method] = f"{result.execution_time:.4f}s"
            
            report_data['detailed_results'][method] = {
                'anomaly_indices': np.where(result.anomalies)[0].tolist(),
                'has_scores': result.scores is not None,
                'metadata_available': result.metadata is not None
            }
        
        # 集成结果
        if len(results) > 1:
            ensemble_result = self.ensemble_detection(results)
            ensemble_count = ensemble_result.anomalies.sum()
            ensemble_rate = ensemble_count / len(ensemble_result.anomalies) * 100
            
            report_data['ensemble'] = {
                'anomaly_count': int(ensemble_count),
                'anomaly_rate': f"{ensemble_rate:.2f}%",
                'voting_threshold': ensemble_result.parameters['voting_threshold']
            }
        
        # 保存报告
        reports = {}
        
        if output_format in ['json', 'both']:
            json_path = os.path.join(self.config.output_dir, 'detection_report.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            reports['json'] = json_path
        
        if output_format in ['text', 'both']:
            text_report = self._generate_text_report(report_data)
            text_path = os.path.join(self.config.output_dir, 'detection_report.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_report)
            reports['text'] = text_path
        
        return reports
    
    def _generate_text_report(self, report_data: Dict) -> str:
        """生成文本格式报告"""
        lines = []
        
        lines.append("🔍 SEEKER 异常检测报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {report_data['timestamp']}")
        lines.append(f"数据长度: {report_data['data_length']}")
        lines.append(f"使用方法: {', '.join(report_data['methods_used'])}")
        lines.append("")
        
        lines.append("📊 检测汇总:")
        for method, summary in report_data['summary'].items():
            lines.append(f"  {method.upper()}:")
            lines.append(f"    异常数量: {summary['anomaly_count']}")
            lines.append(f"    异常率: {summary['anomaly_rate']}")
            lines.append(f"    参数: {summary['parameters']}")
        
        if 'ensemble' in report_data:
            lines.append(f"  ENSEMBLE:")
            lines.append(f"    异常数量: {report_data['ensemble']['anomaly_count']}")
            lines.append(f"    异常率: {report_data['ensemble']['anomaly_rate']}")
            lines.append(f"    投票阈值: {report_data['ensemble']['voting_threshold']}")
        
        lines.append("")
        
        if report_data['performance']:
            lines.append("⚡ 性能统计:")
            for method, time_str in report_data['performance'].items():
                lines.append(f"  {method}: {time_str}")
            lines.append("")
        
        lines.append("🎯 建议:")
        # 根据结果生成建议
        anomaly_counts = [summary['anomaly_count'] for summary in report_data['summary'].values()]
        if max(anomaly_counts) == 0:
            lines.append("  - 未检测到异常，数据看起来正常")
        elif max(anomaly_counts) > len(report_data['methods_used']) * 5:
            lines.append("  - 检测到大量异常，建议检查数据质量或调整参数")
        else:
            lines.append("  - 检测结果合理，建议进一步分析异常原因")
        
        return "\n".join(lines)

# 便捷函数
def quick_detect(data: Union[pd.Series, np.ndarray], 
                methods: List[str] = None,
                config: SeekerConfig = None,
                enable_visualization: bool = True,
                enable_reports: bool = True) -> Dict[str, DetectionResult]:
    """
    快速异常检测函数
    
    Args:
        data: 时间序列数据
        methods: 检测方法列表
        config: 配置对象
        enable_visualization: 是否显示可视化
        enable_reports: 是否生成报告
    """
    if config is None:
        config = SeekerConfig()
    
    config.enable_visualization = enable_visualization
    config.enable_reports = enable_reports
    
    detector = SeekerAnomalyDetector(config)
    results = detector.detect_single_metric(data, methods)
    
    if enable_visualization:
        detector.plot_comprehensive_analysis(results)
    
    if enable_reports:
        reports = detector.generate_report(results)
        print(f"📄 报告已生成: {reports}")
    
    return results

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='10min')
    
    # 创建模拟CPU使用率数据
    base_pattern = 30 + 15 * np.sin(2 * np.pi * np.arange(200) / 144)
    noise = np.random.normal(0, 3, 200)
    cpu_data = base_pattern + noise
    
    # 插入异常
    cpu_data[[50, 100, 150]] = [85, 5, 90]
    
    # 创建时间序列
    ts_data = pd.Series(cpu_data, index=dates)
    
    print("🚀 开始 Seeker 综合异常检测演示")
    print("=" * 60)
    
    # 使用快速检测函数
    methods_to_test = ['zscore', 'iqr', 'ewma', 'isolation_forest', 'lof']
    
    # 添加高级方法（如果可用）
    if DEEP_LEARNING_AVAILABLE:
        methods_to_test.append('lstm_autoencoder')
    if SEASONAL_AVAILABLE:
        methods_to_test.append('seasonal')
    
    # 自定义配置
    config = SeekerConfig(
        zscore_threshold=2.5,
        iqr_k=2.0,
        ewma_span=20,
        enable_visualization=True,
        enable_reports=True,
        output_dir="seeker_results"
    )
    
    # 执行检测
    results = quick_detect(
        data=ts_data,
        methods=methods_to_test,
        config=config,
        enable_visualization=True,
        enable_reports=True
    )
    
    print(f"\n✅ 检测完成！使用了 {len(results)} 种方法")
    print(f"📁 结果已保存到: {config.output_dir}")
    
    # 打印简要统计
    for method, result in results.items():
        count = result.anomalies.sum()
        rate = count / len(result.anomalies) * 100
        time_str = f" ({result.execution_time:.3f}s)" if result.execution_time else ""
        print(f"  {method}: {count} 个异常 ({rate:.1f}%){time_str}")
