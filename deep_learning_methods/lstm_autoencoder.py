"""
LSTM Autoencoder 异常检测实现
适用于时间序列的深度学习异常检测方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class LSTMAutoencoder:
    """LSTM自编码器异常检测类"""
    
    def __init__(self, sequence_length=30, encoding_dim=10):
        """
        Args:
            sequence_length: 输入序列长度
            encoding_dim: 编码器隐藏层维度
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def build_model(self):
        """构建LSTM自编码器模型"""
        # 输入层
        input_layer = Input(shape=(self.sequence_length, 1))
        
        # 编码器
        encoded = LSTM(self.encoding_dim, return_sequences=False)(input_layer)
        
        # 解码器
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(self.encoding_dim, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(1))(decoded)
        
        # 构建模型
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return self.model
    
    def create_sequences(self, data):
        """创建训练序列"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """训练模型"""
        # 数据预处理
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        sequences = self.create_sequences(scaled_data)
        
        # 构建模型
        if self.model is None:
            self.build_model()
        
        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练模型
        history = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 计算重构误差阈值
        reconstructed = self.model.predict(sequences)
        mse = np.mean(np.power(sequences - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95分位数作为阈值
        
        return history
    
    def predict_anomalies(self, data):
        """预测异常"""
        # 数据预处理
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        sequences = self.create_sequences(scaled_data)
        
        # 预测重构
        reconstructed = self.model.predict(sequences)
        
        # 计算重构误差
        mse = np.mean(np.power(sequences - reconstructed, 2), axis=1)
        
        # 异常判定
        anomalies = mse > self.threshold
        
        # 扩展结果到原始数据长度
        extended_anomalies = np.zeros(len(data), dtype=bool)
        extended_anomalies[self.sequence_length-1:] = anomalies
        
        return extended_anomalies, mse
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Autoencoder Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def lstm_autoencoder_detection(data, sequence_length=30, encoding_dim=10, 
                              epochs=50, contamination_percentile=95):
    """
    使用LSTM自编码器进行异常检测的便捷函数
    
    Args:
        data: 时间序列数据
        sequence_length: 序列长度
        encoding_dim: 编码维度
        epochs: 训练轮数
        contamination_percentile: 异常阈值百分位数
    
    Returns:
        anomalies: 异常检测结果
        reconstruction_errors: 重构误差
    """
    detector = LSTMAutoencoder(sequence_length, encoding_dim)
    detector.build_model()
    
    # 训练
    history = detector.fit(data, epochs=epochs, verbose=0)
    
    # 预测
    anomalies, errors = detector.predict_anomalies(data)
    
    return anomalies, errors, detector

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=500, freq='min')
    normal_data = np.random.normal(30, 5, 450)
    
    # 插入异常
    anomaly_indices = [100, 200, 300, 400]
    anomaly_data = np.concatenate([
        normal_data[:100], [85], normal_data[100:200], [5], 
        normal_data[200:300], [90], normal_data[300:400], [8], 
        normal_data[400:]
    ])
    
    # 使用LSTM自编码器检测异常
    anomalies, errors, detector = lstm_autoencoder_detection(
        anomaly_data, 
        sequence_length=30, 
        epochs=30
    )
    
    # 可视化结果
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, anomaly_data, label='CPU Usage', alpha=0.7)
    plt.scatter(timestamps[anomalies], anomaly_data[anomalies], 
               color='red', label='Detected Anomalies', s=50)
    plt.scatter(timestamps[anomaly_indices], anomaly_data[anomaly_indices], 
               color='orange', label='True Anomalies', marker='x', s=100)
    plt.title('LSTM Autoencoder Anomaly Detection')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(timestamps[29:], errors, label='Reconstruction Error', color='purple')
    plt.axhline(y=detector.threshold, color='red', linestyle='--', 
                label=f'Threshold ({detector.threshold:.4f})')
    plt.title('Reconstruction Error')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('deep_learning_methods/lstm_autoencoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"检测到 {anomalies.sum()} 个异常点")
    print(f"重构误差阈值: {detector.threshold:.4f}")
