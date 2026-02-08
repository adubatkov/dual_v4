"""
Backtest Reporting Module

Provides comprehensive reporting for backtest results:
- Excel reports with detailed trade information
- Trade chart generation (candlestick + IB zones)
- Folder structure management
"""

from .excel_report import ExcelReportGenerator
from .trade_charts import TradeChartGenerator
from .report_manager import BacktestReportManager

__all__ = [
    "ExcelReportGenerator",
    "TradeChartGenerator",
    "BacktestReportManager",
]
