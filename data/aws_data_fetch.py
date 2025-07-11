import boto3
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class MetricConfig:
    """指标配置类"""
    namespace: str
    metric_name: str
    statistic: str
    period: int
    dimensions: list


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
        metric_query = [
            {
                'Id': 'metric_data',
                'MetricStat': {
                    'Metric': {
                        'Namespace': metric_config.namespace,
                        'MetricName': metric_config.metric_name,
                        'Dimensions': metric_config.dimensions or []
                    },
                    'Period': metric_config.period,
                    'Stat': metric_config.statistic
                }
            }
        ]
        all_data = []
        next_token = None
        while True:
            params = {
                "MetricDataQueries": metric_query,
                "StartTime": start_time,
                "EndTime": end_time,
            }
            if next_token:
                params["NextToken"] = next_token

            resp = self.cloudwatch.get_metric_data(**params)
            for result in resp["MetricDataResults"]:
                all_data.extend(
                    zip(result["Timestamps"], result["Values"])
                )

            next_token = resp.get("NextToken")
            if not next_token:
                break
        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=["timestamp", "cpu_usage"]).sort_values('timestamp')

        return df
    
    def save_data(self, df: pd.DataFrame, metric_config: MetricConfig) -> str:
        """保存数据到CSV文件
        
        Args:
            df: 包含指标数据的DataFrame
            metric_config: 指标配置对象
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metric_config.metric_name}_{timestamp}.csv"

        data_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(data_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"数据已保存至: {filepath}")
        return filepath

def main():
    """主函数：获取并保存AWS CloudWatch指标数据"""
    # EC2 CPU使用率配置
    cpu_config = MetricConfig(
        namespace='AWS/ECS',
        metric_name='CPUUtilization',
        statistic='Average',
        period=300,
        dimensions=[
            {
                'Name': "ClusterName",
                'Value': "ecs-cluster-fraggles"
            },
            {
                'Name': "ServiceName",
                'Value': "realm-EcsService-8BpiGDiz4loH"
            }
        ]
    )

    # 创建fetcher实例
    fetcher = CloudWatchDataFetcher()
    # HINT: 
    # Data points with a period of less than 60 seconds are available for 3 hours. 
    # Data points with a period of 60 seconds (1-minute) are available for 15 days.
    # Data points with a period of 300 seconds (5-minute) are available for 63 days.
    # Data points with a period of 3600 seconds (1 hour) are available for 455 days (15 months).
    try:
        print(f"正在获取 {cpu_config.metric_name} 数据...")
        df = fetcher.fetch_metric_data(
            metric_config=cpu_config,
            time_range_days=60
        )
        
        # 保存数据
        filepath = fetcher.save_data(df, cpu_config)
        print(f"成功获取 {len(df)} 条数据记录")
        
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return

if __name__ == '__main__':
    main()
