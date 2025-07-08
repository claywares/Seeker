"""
验证器模块

包含性能验证、对比分析等功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from ..scorers import OriginalScorer, RandomForestScorer, NeuralNetworkScorer
from ..utils import MetricsCalculator


class ScorerComparator:
    """评分器对比验证类"""
    
    def __init__(self, data):
        self.data = data
        self.scorers = {
            'Original': OriginalScorer(),
            'RandomForest': RandomForestScorer(),
            'NeuralNetwork': NeuralNetworkScorer()
        }
        self.results = {}
        
    def train_and_evaluate(self, train_ratio=0.7):
        """训练并评估所有评分器"""
        # 分割数据
        train_size = int(train_ratio * len(self.data))
        train_data = self.data.iloc[:train_size].copy()
        test_data = self.data.iloc[train_size:].copy()
        
        true_labels = test_data['is_anomaly'].values
        
        for name, scorer in self.scorers.items():
            print(f"🔧 训练 {name} 评分器...")
            
            # 训练
            scorer.fit(train_data)
            
            # 预测
            scores = scorer.score(test_data)
            
            # 评估
            metrics = MetricsCalculator.evaluate_performance(true_labels, scores)
            
            self.results[name] = {
                'scorer': scorer,
                'scores': scores,
                'metrics': metrics
            }
            
            print(f"   - Precision: {metrics['precision']:.3f}")
            print(f"   - Recall: {metrics['recall']:.3f}")
            print(f"   - F1-Score: {metrics['f1']:.3f}")
            print(f"   - AUC: {metrics['auc']:.3f}")
            print()
            
        return self.results
    
    def plot_comparison(self, figsize=(15, 10)):
        """绘制对比图"""
        if not self.results:
            raise ValueError("请先运行 train_and_evaluate()")
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('评分器性能对比', fontsize=16, fontweight='bold')
        
        # 1. 评分分布对比
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            ax1.hist(result['scores'], bins=30, alpha=0.6, label=name)
        ax1.set_title('评分分布对比')
        ax1.set_xlabel('异常评分')
        ax1.set_ylabel('频次')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC曲线对比
        ax2 = axes[0, 1]
        true_labels = self.data.iloc[int(0.7*len(self.data)):]['is_anomaly'].values
        
        for name, result in self.results.items():
            if len(np.unique(true_labels)) > 1:
                fpr, tpr, _ = roc_curve(true_labels, result['scores'])
                auc_score = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        ax2.set_title('ROC曲线对比')
        ax2.set_xlabel('假正率 (FPR)')
        ax2.set_ylabel('真正率 (TPR)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 性能指标对比
        ax3 = axes[1, 0]
        metrics_df = pd.DataFrame({
            name: [result['metrics']['precision'], 
                   result['metrics']['recall'], 
                   result['metrics']['f1']]
            for name, result in self.results.items()
        }, index=['Precision', 'Recall', 'F1-Score'])
        
        metrics_df.plot(kind='bar', ax=ax3)
        ax3.set_title('性能指标对比')
        ax3.set_ylabel('得分')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. 特征重要性 (仅Random Forest)
        ax4 = axes[1, 1]
        if 'RandomForest' in self.results:
            rf_scorer = self.results['RandomForest']['scorer']
            importance_df = rf_scorer.get_feature_importance().head(10)
            bars = ax4.barh(range(len(importance_df)), importance_df['importance'])
            ax4.set_yticks(range(len(importance_df)))
            ax4.set_yticklabels(importance_df['feature'], fontsize=9)
            ax4.set_title('Random Forest特征重要性')
            ax4.set_xlabel('重要性')
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self):
        """生成对比报告"""
        if not self.results:
            raise ValueError("请先运行 train_and_evaluate()")
            
        print("=" * 60)
        print("📊 评分器性能对比报告")
        print("=" * 60)
        
        # 性能对比表
        metrics_df = pd.DataFrame({
            name: result['metrics'] 
            for name, result in self.results.items()
        }).T
        
        print("\n🎯 性能指标对比:")
        print(metrics_df[['precision', 'recall', 'f1', 'auc']].round(3))
        
        # 最佳性能
        best_f1 = metrics_df['f1'].idxmax()
        best_auc = metrics_df['auc'].idxmax()
        
        print(f"\n🏆 最佳F1-Score: {best_f1} ({metrics_df.loc[best_f1, 'f1']:.3f})")
        print(f"🏆 最佳AUC: {best_auc} ({metrics_df.loc[best_auc, 'auc']:.3f})")
        
        # 改进建议
        print(f"\n💡 改进建议:")
        if 'RandomForest' in self.results:
            rf_precision = self.results['RandomForest']['metrics']['precision']
            orig_precision = self.results['Original']['metrics']['precision']
            improvement = rf_precision - orig_precision
            print(f"   - Random Forest相比原始方法Precision提升: {improvement:+.3f}")
            
        print("   - 建议在生产环境中A/B测试验证效果")
        print("   - 可考虑集成多个模型进一步提升性能")
        
        return metrics_df
