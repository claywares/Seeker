# Seeker - Machine Learning for Time Series Anomaly Detection

## Definition of abnormality

Abnormality (Anomaly) in Seeker refers to a data point or pattern in a time series that significantly deviates from the expected behavior, as learned from historical data. This includes:

*Point anomalies*: Individual data points that are unusually high or low compared to the rest of the series.

*Contextual anomalies*: Data points that are only abnormal within a specific context (e.g., a value that is normal on weekends but abnormal on weekdays).

*Collective anomalies*: Sequences or patterns of data points that are abnormal together, even if individual points are not.

### Limitations
Seeker currently focuses on detecting point in univariate time series. It may not detect all types of anomalies, such as:

- Contextual anomalies
- Gradual drifts or subtle changes in trend
- Anomalies in multivariate or highly irregular time series
- Anomalies requiring domain-specific rules or external context

## Point Anomalies
Point anomalies are the simplest and most common types of anomalies, perfect for first-time experiments and demonstrations. For example, instantaneous surge or drop in CPU and Memory, these are all point anomalies.

- Z-score
- IQR

### Z-score
The choice of 3 as the threshold is based on the "Three-sigma rule" (68-95-99.7 rule) in statistics:
#### Statistical Distribution
- Within ±1σ: ~68% of data
- Within ±2σ: ~95% of data
- Within ±3σ: ~99.7% of data
#### Trade-off Consideration
- Lower threshold (e.g., 2σ): More sensitive but more false alarms
- Higher threshold (e.g., 4σ): Fewer false alarms but might miss anomalies
- 3σ: Generally good balance for most cases

### IQR
#### Quartiles:
- Q1 (25th percentile): 25% of data falls below this value
- Q3 (75th percentile): 75% of data falls below this value
- IQR = Q3 - Q1: Range containing middle 50% of data
#### k parameter (typically 1.5):
- Controls sensitivity of detection
- k=1.5 (standard): moderately strict
- k=3.0: more lenient
- Lower k = more anomalies detected
#### Thresholds:
- Lower bound = Q1 - k×IQR
- Upper bound = Q3 + k×IQR
- Any points outside these bounds are considered anomalies

### Comparison
- CPU burst peak monitoring: use Z-score first
- Long-term trend anomaly: use IQR first
