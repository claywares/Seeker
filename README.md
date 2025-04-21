# Seeker - Machine Learning for Time Series Anomaly Detection

## Overview

Seeker is a demonstration project that showcases how machine learning and statistical techniques can be used to detect anomalies in time series metrics. Instead of manually setting thresholds for thousands of services, this approach automatically identifies unusual patterns in the data.

The project implements several anomaly detection algorithms and compares their performance on synthetic time series data with injected anomalies.

## Features

- **Multiple Anomaly Detection Algorithms**:
  - Statistical methods (Z-score)
  - Exponentially Weighted Moving Average (EWMA)
  - Machine Learning methods (Isolation Forest, One-Class SVM)
  - Time series decomposition (STL)

- **Synthetic Data Generation**:
  - Simple time series with trend, seasonality, and noise
  - Complex time series with multiple seasonal patterns and different types of anomalies

- **Evaluation and Visualization**:
  - Performance metrics (precision, recall, F1 score)
  - Visual comparison of detected anomalies

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Seeker
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the demo:

```
python main.py
```

This will:
1. Generate synthetic time series data with injected anomalies
2. Apply different anomaly detection algorithms
3. Evaluate and compare their performance
4. Visualize the results

## Project Structure

- `main.py`: Main script to run the demo
- `anomaly_detection.py`: Implementation of anomaly detection algorithms and utility functions
- `requirements.txt`: Required Python packages

## Anomaly Detection Algorithms

### Statistical Detector (Z-score)
Detects anomalies based on how many standard deviations a data point is from the mean.

### EWMA Detector
Uses Exponentially Weighted Moving Average to detect anomalies, giving more weight to recent observations.

### Isolation Forest
A machine learning algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

### One-Class SVM
A machine learning algorithm that learns a decision boundary that encompasses the normal data points, treating outliers as anomalies.

### STL Decomposition
Decomposes the time series into trend, seasonality, and residual components, then identifies anomalies in the residual component.

## Extending for Real-World Use

To use this project with real-world data:

1. Replace the data generation functions with code to load your actual time series data
2. Adjust the parameters of the anomaly detection algorithms to suit your specific use case
3. Implement additional preprocessing steps if needed (e.g., handling missing values, normalization)
4. Consider implementing online/streaming anomaly detection for real-time monitoring

## License

[MIT License](LICENSE)
