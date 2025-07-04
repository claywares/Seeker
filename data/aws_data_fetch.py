import boto3
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class MetricConfig:
    """指标配置类"""
    namespace: str
    metric_name: str
    statistic: str
    period: int
    dimension_name: str
    dimension_value: str

class CloudWatchDataFetcher:
    """AWS CloudWatch数据获取类"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        
    def fetch_metric_data(
        self,
        metric_config: MetricConfig,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        time_range_days: int = 7
    ) -> pd.DataFrame:
        """获取CloudWatch指标数据
        
        Args:
            metric_config: 指标配置对象
            start_time: 开始时间
            end_time: 结束时间
            time_range_days: 如果未指定时间范围，则获取过去多少天的数据
        """
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=time_range_days)

        # 构建查询
        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'metric_data',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': metric_config.namespace,
                            'MetricName': metric_config.metric_name,
                            'Dimensions': [
                                {
                                    'Name': metric_config.dimension_name,
                                    'Value': metric_config.dimension_value
                                }
                            ]
                        },
                        'Period': metric_config.period,
                        'Stat': metric_config.statistic
                    }
                }
            ],
            StartTime=start_time,
            EndTime=end_time
        )

        # 转换为DataFrame
        df = pd.DataFrame({
            'timestamp': response['MetricDataResults'][0]['Timestamps'],
            'value': response['MetricDataResults'][0]['Values']
        }).sort_values('timestamp')
        
        return df
    
    def save_data(self, df: pd.DataFrame, metric_config: MetricConfig) -> str:
        """保存数据到CSV文件
        
        Args:
            df: 包含指标数据的DataFrame
            metric_config: 指标配置对象
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metric_config.metric_name}_{metric_config.dimension_value}_{timestamp}.csv"
        
        data_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(data_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"数据已保存至: {filepath}")
        return filepath

def main():
    """主函数：获取并保存AWS CloudWatch指标数据"""
    # EC2 CPU使用率配置
    cpu_config = MetricConfig(
        namespace='AWS/EC2',
        metric_name='CPUUtilization',
        statistic='Average',
        period=60,  # 1分钟间隔
        dimension_name='InstanceId',
        dimension_value='i-00f458499ca38a3c7'
    )
    
    # 创建fetcher实例
    fetcher = CloudWatchDataFetcher()
    
    try:
        # 获取过去7天的数据
        print(f"正在获取 {cpu_config.metric_name} 数据...")
        df = fetcher.fetch_metric_data(
            metric_config=cpu_config,
            time_range_days=7
        )
        
        # 保存数据
        filepath = fetcher.save_data(df, cpu_config)
        print(f"成功获取 {len(df)} 条数据记录")
        
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return

if __name__ == '__main__':
    main()
