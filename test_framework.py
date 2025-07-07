#!/usr/bin/env python3
"""
简单测试脚本，验证 Seeker 框架基本功能
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

try:
    from point_anomalies.main import zscore_detection, iqr_detection, ewma_detection
    print("✅ 基础检测方法导入成功")
except ImportError as e:
    print(f"❌ 基础方法导入失败: {e}")

try:
    from seeker_framework import SeekerAnomalyDetector, SeekerConfig, quick_detect
    print("✅ Seeker 框架导入成功")
except ImportError as e:
    print(f"❌ Seeker 框架导入失败: {e}")
    sys.exit(1)

def test_basic_detection():
    """测试基础检测功能"""
    print("\n🧪 测试基础异常检测...")
    
    # 生成测试数据
    np.random.seed(42)
    data = np.random.normal(30, 5, 100)
    data[20] = 80  # 插入异常
    data[50] = 5   # 插入异常
    
    ts_data = pd.Series(data)
    
    # 配置
    config = SeekerConfig(
        enable_visualization=False,  # 禁用可视化避免显示问题
        enable_reports=True,
        output_dir="test_results"
    )
    
    # 测试基础方法
    methods = ['zscore', 'iqr', 'ewma', 'isolation_forest', 'lof']
    
    try:
        results = quick_detect(
            data=ts_data,
            methods=methods,
            config=config,
            enable_visualization=False,
            enable_reports=True
        )
        
        print(f"✅ 检测成功，使用了 {len(results)} 种方法")
        
        for method, result in results.items():
            count = result.anomalies.sum()
            rate = count / len(result.anomalies) * 100
            print(f"  {method}: {count} 个异常 ({rate:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        return False

def test_framework_components():
    """测试框架各组件"""
    print("\n🔧 测试框架组件...")
    
    # 测试配置
    config = SeekerConfig()
    print(f"✅ 配置创建成功: {config.zscore_threshold}")
    
    # 测试检测器
    detector = SeekerAnomalyDetector(config)
    print("✅ 检测器创建成功")
    
    # 生成测试数据
    data = pd.Series(np.random.normal(30, 5, 50))
    data[10] = 70
    
    # 测试检测
    try:
        results = detector.detect_single_metric(data, methods=['zscore', 'iqr'])
        print(f"✅ 单指标检测成功: {len(results)} 个结果")
        
        # 测试集成
        ensemble_result = detector.ensemble_detection(results)
        print(f"✅ 集成检测成功: {ensemble_result.anomalies.sum()} 个异常")
        
        return True
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Seeker 框架测试")
    print("=" * 50)
    
    # 创建测试目录
    os.makedirs("test_results", exist_ok=True)
    
    # 运行测试
    basic_ok = test_basic_detection()
    framework_ok = test_framework_components()
    
    print("\n" + "=" * 50)
    if basic_ok and framework_ok:
        print("✅ 所有测试通过！Seeker 框架工作正常")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    exit(main())
