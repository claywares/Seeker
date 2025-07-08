"""
Seeker - 时间序列异常检测框架

一个智能的、多算法融合的异常检测系统，支持：
- 传统统计方法 (Z-score, IQR, EWMA)
- 机器学习方法 (Isolation Forest, LOF)
- 智能评分器 (Random Forest, Neural Network)
"""

__version__ = "0.2.0"
__author__ = "Seeker Team"

from .detectors import *
from .scorers import *
from .validators import *
from .utils import *
