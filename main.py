import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anomaly_detection import (
    StatisticalDetector, 
    EWMADetector, 
    IsolationForestDetector, 
    OneClassSVMDetector, 
    STLDecompositionDetector,
    visualize_anomalies,
    evaluate_detectors
)

def dummy_data(n_points=1000, anomaly_percentage=0.005, random_seed=42):
    """
    Generate dummy time series data with trend, seasonality, and injected anomalies
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    anomaly_percentage : float
        Percentage of data points to be anomalies (between 0 and 1)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the generated time series data
    """
    np.random.seed(random_seed)
    
    # Generate time points
    t = np.arange(0, n_points)
    
    # Generate trend component (quadratic)
    trend = 0.0005 * (t - n_points/2)**2
    
    # Generate seasonal component (sine wave)
    seasonality = 10 * np.sin(2 * np.pi * t / 50)
    
    # Generate random noise
    noise = np.random.normal(0, 2, n_points)
    
    # Combine components
    normal_ts = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_points, freq='5min'),
        'value': normal_ts,
        'is_anomaly': 0
    })
    
    # Inject anomalies
    n_anomalies = int(n_points * anomaly_percentage)
    anomaly_positions = np.random.choice(n_points, n_anomalies, replace=False)
    df.loc[anomaly_positions, 'value'] += 50 * np.random.randn(n_anomalies)
    df.loc[anomaly_positions, 'is_anomaly'] = 1
    
    return df

def generate_complex_data(n_points=1000, anomaly_percentage=0.01, random_seed=42):
    """
    Generate more complex time series data with multiple patterns and different types of anomalies
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    anomaly_percentage : float
        Percentage of data points to be anomalies (between 0 and 1)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the generated time series data
    """
    np.random.seed(random_seed)
    
    # Generate time points
    t = np.arange(0, n_points)
    
    # Generate trend component (combination of linear and quadratic)
    trend = 0.01 * t + 0.0001 * (t - n_points/2)**2
    
    # Generate seasonal components (multiple frequencies)
    seasonality1 = 10 * np.sin(2 * np.pi * t / 50)  # Daily pattern
    seasonality2 = 5 * np.sin(2 * np.pi * t / 7)    # Weekly pattern
    seasonality = seasonality1 + seasonality2
    
    # Generate random noise
    noise = np.random.normal(0, 1, n_points)
    
    # Combine components
    normal_ts = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_points, freq='5min'),
        'value': normal_ts,
        'is_anomaly': 0
    })
    
    # Inject different types of anomalies
    n_anomalies = int(n_points * anomaly_percentage)
    anomaly_positions = np.random.choice(n_points, n_anomalies, replace=False)
    
    # Point anomalies (spikes)
    point_anomalies = anomaly_positions[:n_anomalies//2]
    df.loc[point_anomalies, 'value'] += 30 * np.random.randn(len(point_anomalies))
    df.loc[point_anomalies, 'is_anomaly'] = 1
    
    # Contextual anomalies (values that are normal in general but abnormal in context)
    contextual_anomalies = anomaly_positions[n_anomalies//2:]
    for pos in contextual_anomalies:
        # Invert the seasonal pattern
        if pos < n_points - 1:
            seasonal_value = seasonality[pos]
            df.loc[pos, 'value'] = trend[pos] - 2 * seasonal_value + noise[pos]
            df.loc[pos, 'is_anomaly'] = 1
    
    return df

def run_demo():
    """Run the anomaly detection demo"""
    import sys
    
    # Open a log file
    log_file = open('demo_log.txt', 'w')
    
    def log(message):
        """Write message to log file and print to console"""
        message_str = str(message)
        log_file.write(f"{message_str}\n")
        log_file.flush()
        print(message_str)
        sys.stdout.flush()
    
    log("Generating time series data...")
    df = dummy_data(n_points=1000, anomaly_percentage=0.005)
    
    log(f"Generated {len(df)} data points with {df['is_anomaly'].sum()} anomalies")
    
    # Create detectors with different configurations
    detectors = [
        StatisticalDetector(threshold=3.0),
        EWMADetector(span=20, threshold=3.0),
        IsolationForestDetector(contamination=0.01),
        OneClassSVMDetector(nu=0.01),
        STLDecompositionDetector(period=50, threshold=3.0)
    ]
    
    log("\nEvaluating anomaly detection algorithms...")
    results_df, predictions_dict = evaluate_detectors(df, detectors)
    
    # Print evaluation results
    log("\nEvaluation Results:")
    log(results_df[['Detector', 'Precision', 'Recall', 'F1 Score']].to_string(index=False))
    
    # Visualize results
    log("\nVisualizing results...")
    fig = visualize_anomalies(df, predictions_dict, title="Anomaly Detection Results")
    plt.savefig('simple_data_results.png')
    log("Results saved to simple_data_results.png")
    
    # Try with more complex data
    log("\n\nGenerating more complex time series data...")
    complex_df = generate_complex_data(n_points=1000, anomaly_percentage=0.01)
    
    log(f"Generated {len(complex_df)} data points with {complex_df['is_anomaly'].sum()} anomalies")
    
    log("\nEvaluating anomaly detection algorithms on complex data...")
    complex_results_df, complex_predictions_dict = evaluate_detectors(complex_df, detectors)
    
    # Print evaluation results for complex data
    log("\nEvaluation Results (Complex Data):")
    log(complex_results_df[['Detector', 'Precision', 'Recall', 'F1 Score']].to_string(index=False))
    
    # Visualize results for complex data
    log("\nVisualizing results for complex data...")
    complex_fig = visualize_anomalies(complex_df, complex_predictions_dict, title="Anomaly Detection Results (Complex Data)")
    plt.savefig('complex_data_results.png')
    log("Results saved to complex_data_results.png")

if __name__ == '__main__':
    run_demo()
