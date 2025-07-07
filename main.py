#!/usr/bin/env python3
"""
Seeker 主入口文件
提供命令行接口和简单的使用示例
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from seeker_framework import SeekerAnomalyDetector, SeekerConfig, quick_detect

def load_data(file_path: str) -> pd.Series:
    """加载数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 根据文件扩展名选择加载方式
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        
        # 尝试自动识别时间和数值列
        if 'timestamp' in df.columns and len(df.columns) >= 2:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            # 选择第一个数值列
            value_col = [col for col in df.columns if df[col].dtype in ['float64', 'int64']][0]
            return df[value_col]
        else:
            # 假设第一列是数值
            return pd.Series(df.iloc[:, 0].values)
    
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
        return pd.Series(df.iloc[:, 0].values)
    
    else:
        # 尝试作为文本文件读取
        data = np.loadtxt(file_path)
        return pd.Series(data)

def run_demo():
    """运行演示示例"""
    print("🚀 Seeker 异常检测演示")
    print("=" * 60)
    
    # 生成演示数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='5min')
    
    print("📊 生成模拟数据...")
    # 创建复杂的模拟数据
    # 1. 基础日周期模式
    hours = np.arange(300) * 5 / 60  # 转换为小时
    daily_pattern = 30 + 20 * np.sin(2 * np.pi * hours / 24)
    
    # 2. 添加工作日/周末模式
    day_of_week = (hours / 24) % 7
    weekday_boost = np.where((day_of_week >= 1) & (day_of_week <= 5), 10, -5)
    
    # 3. 添加业务时间模式
    hour_of_day = hours % 24
    business_hours = np.where((hour_of_day >= 9) & (hour_of_day <= 17), 15, 0)
    
    # 4. 随机噪声
    noise = np.random.normal(0, 4, 300)
    
    # 5. 合成基础数据
    cpu_data = daily_pattern + weekday_boost + business_hours + noise
    cpu_data = np.clip(cpu_data, 0, 100)  # 限制在合理范围
    
    # 6. 插入各种类型的异常
    print("🎯 插入已知异常...")
    # 尖峰异常
    cpu_data[50] = 95
    cpu_data[150] = 90
    
    # 低值异常
    cpu_data[100] = 2
    cpu_data[200] = 5
    
    # 趋势异常（连续异常）
    cpu_data[250:255] += 30
    
    # 振荡异常
    cpu_data[280:290] = cpu_data[280:290] + 20 * np.sin(np.arange(10) * 2)
    
    # 创建时间序列
    ts_data = pd.Series(cpu_data, index=dates)
    
    print(f"✅ 数据生成完成，包含 {len(ts_data)} 个数据点")
    print(f"   时间范围: {ts_data.index[0]} 至 {ts_data.index[-1]}")
    print(f"   数值范围: {ts_data.min():.1f} - {ts_data.max():.1f}")
    
    # 配置检测参数
    config = SeekerConfig(
        zscore_threshold=2.5,
        iqr_k=1.8,
        ewma_span=20,
        ewma_threshold=2.2,
        isolation_forest_contamination=0.025,
        lof_contamination=0.025,
        enable_visualization=True,
        enable_reports=True,
        output_dir="demo_results"
    )
    
    # 选择检测方法
    methods = ['zscore', 'iqr', 'ewma', 'isolation_forest', 'lof']
    
    print(f"🔍 开始异常检测，使用方法: {', '.join(methods)}")
    
    # 执行检测
    results = quick_detect(
        data=ts_data,
        methods=methods,
        config=config,
        enable_visualization=True,
        enable_reports=True
    )
    
    # 分析结果
    print("\n📈 检测结果汇总:")
    total_detected = 0
    for method, result in results.items():
        count = result.anomalies.sum()
        rate = count / len(result.anomalies) * 100
        time_info = f" ({result.execution_time:.3f}s)" if result.execution_time else ""
        print(f"  {method.upper():15}: {count:3d} 异常 ({rate:5.1f}%){time_info}")
        total_detected += count
    
    # 集成检测
    detector = SeekerAnomalyDetector(config)
    ensemble_result = detector.ensemble_detection(results, voting_threshold=0.3)
    ensemble_count = ensemble_result.anomalies.sum()
    ensemble_rate = ensemble_count / len(ensemble_result.anomalies) * 100
    
    print(f"  {'ENSEMBLE':15}: {ensemble_count:3d} 异常 ({ensemble_rate:5.1f}%)")
    
    print(f"\n🎯 真实异常位置: [50, 100, 150, 200, 250-254, 280-289]")
    print(f"📊 集成检测位置: {np.where(ensemble_result.anomalies)[0].tolist()}")
    
    print(f"\n✅ 演示完成！结果已保存到 'demo_results' 目录")
    
    return ts_data, results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Seeker 时间序列异常检测工具')
    parser.add_argument('--data', type=str, help='数据文件路径 (CSV, JSON, TXT)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['zscore', 'iqr', 'ewma', 'isolation_forest', 'lof', 'lstm_autoencoder', 'seasonal'],
                       default=['zscore', 'iqr', 'ewma', 'isolation_forest'],
                       help='使用的检测方法')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--contamination', type=float, default=0.02, help='预期异常比例')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化')
    parser.add_argument('--no-report', action='store_true', help='禁用报告生成')
    parser.add_argument('--demo', action='store_true', help='运行演示示例')
    
    args = parser.parse_args()
    
    if args.demo:
        return run_demo()
    
    if not args.data:
        print("错误: 请提供数据文件路径或使用 --demo 运行演示")
        print("使用 'python main.py --help' 查看帮助")
        return
    
    try:
        # 加载数据
        print(f"📂 加载数据: {args.data}")
        data = load_data(args.data)
        print(f"✅ 数据加载成功，包含 {len(data)} 个数据点")
        
        # 配置
        config = SeekerConfig(
            isolation_forest_contamination=args.contamination,
            lof_contamination=args.contamination,
            enable_visualization=not args.no_viz,
            enable_reports=not args.no_report,
            output_dir=args.output
        )
        
        # 执行检测
        print(f"🔍 开始异常检测，方法: {', '.join(args.methods)}")
        results = quick_detect(
            data=data,
            methods=args.methods,
            config=config,
            enable_visualization=not args.no_viz,
            enable_reports=not args.no_report
        )
        
        # 打印结果
        print(f"\n📊 检测完成:")
        for method, result in results.items():
            count = result.anomalies.sum()
            rate = count / len(result.anomalies) * 100
            print(f"  {method}: {count} 个异常 ({rate:.1f}%)")
        
        print(f"📁 结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
