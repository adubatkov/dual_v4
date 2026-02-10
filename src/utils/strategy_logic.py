import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Tuple, List
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
import pytz
import sys

# ================================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ
# ================================

# Пути к данным
GER40_DATA_FOLDER = r"C:\Users\User\PyCharmMiscProject\Ger40 IBR Claude improve\data"
XAU_DATA_FOLDER = r"C:\Users\User\PyCharmMiscProject\Ger40 IBR Claude improve\dataxau"


# ================================
# PRODUCTION PARAMETERS
# GER40: Baseline (FOS 08:00-09:00, uniform params)
# XAUUSD: V9 - XAUUSD_059
# ================================

# GER40 Baseline Parameters
# IB: 08:00-09:00 Europe/Berlin (Frankfurt Opening Session), IB_WAIT: 0
# Trade window: 120 min (09:00-11:00, avoiding Lunch Hours 11:00-13:00)
# Uniform baseline for all variations - starting point for optimization
GER40_PARAMS_PROD = {
    "OCAE": {
        "IB_START": "08:00",
        "IB_END": "09:00",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 0,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.0,
        "MAX_DISTANCE_PCT": 0.0,
    },
    "TCWE": {
        "IB_START": "08:00",
        "IB_END": "09:00",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 0,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.0,
        "MAX_DISTANCE_PCT": 0.0,
    },
    "Reverse": {
        "IB_START": "08:00",
        "IB_END": "09:00",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 0,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.0,
        "MAX_DISTANCE_PCT": 0.0,
    },
    "REV_RB": {
        "IB_START": "08:00",
        "IB_END": "09:00",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 0,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.0,
        "MAX_DISTANCE_PCT": 0.0,
    },
}

# XAUUSD Production Parameters (V9 - XAUUSD_059)
# IB: 09:00-09:30 Asia/Tokyo, IB_WAIT: 20 min
# Source: ib_buffer_maxdist_total_r, Combined R: 95.84, Trades: 627
XAUUSD_PARAMS_PROD = {
    "OCAE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 1.25,
        "STOP_MODE": "eq",
        "TSL_TARGET": 0.0,
        "TSL_SL": 0.5,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 0.75,
        # R: 32.74, Sharpe: 2.22, WR: 54.42%, Trades: 226, MaxDD: 7.24
    },
    "TCWE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 0.75,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.5,
        "TSL_SL": 1.5,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 0.75,
        # R: 19.04, Sharpe: 1.02, WR: 61.18%, Trades: 304, MaxDD: 11.72
    },
    "Reverse": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 1.5,
        "STOP_MODE": "eq",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "REV_RB_PCT": 1.0,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 0.75,
        # R: 44.06, Sharpe: 3.53, WR: 47.62%, Trades: 97, MaxDD: 11.47
    },
    "REV_RB": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 1.75,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_PCT": 1.0,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 0.75,
        # R: 0.0, Sharpe: 0.0, WR: 0.0%, Trades: 0, MaxDD: 0.0
    },
}


# ================================
# КОНСОЛЬНЫЙ ВЫВОД С ЦВЕТАМИ
# ================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_status(message, status="INFO"):
    """Вывод статусных сообщений с цветами"""
    colors = {
        "INFO": Colors.OKBLUE,
        "SUCCESS": Colors.OKGREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
        "HEADER": Colors.HEADER
    }
    color = colors.get(status, Colors.ENDC)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {status}: {message}{Colors.ENDC}")

# ================================
# УТИЛИТЫ ВРЕМЕНИ/СЕССИЙ
# ================================
def parse_session_time(t_str: str) -> time:
    return datetime.strptime(t_str, "%H:%M").time()

def ib_window_on_date(local_date: datetime.date, start_str: str, end_str: str, timezone_str: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Возвращает окно формирования IB для заданной даты"""
    tz = pytz.timezone(timezone_str)
    start_naive = datetime.combine(local_date, parse_session_time(start_str))
    end_naive = datetime.combine(local_date, parse_session_time(end_str))
    start = tz.localize(start_naive)
    end = tz.localize(end_naive)
    return (pd.Timestamp(start), pd.Timestamp(end))

def trade_window_on_date(local_date: datetime.date, ib_end_str: str, timezone_str: str,
                         wait_minutes: int, duration_minutes: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Возвращает торговое окно после IB + wait"""
    tz = pytz.timezone(timezone_str)
    ib_end_naive = datetime.combine(local_date, parse_session_time(ib_end_str))
    ib_end = tz.localize(ib_end_naive)
    trade_start = ib_end + timedelta(minutes=wait_minutes)
    trade_end = trade_start + timedelta(minutes=duration_minutes)
    return (pd.Timestamp(trade_start), pd.Timestamp(trade_end))

def get_local_date(ts: pd.Timestamp, timezone_str: str) -> datetime.date:
    """Получить локальную дату для временной метки в заданной тайм-зоне"""
    tz = pytz.timezone(timezone_str)
    return ts.astimezone(tz).date()

def fmt_dt_timezone_str(ts, timezone_str: str = "Europe/Berlin"):
    """Форматирование даты-времени в заданной тайм-зоне"""
    if pd.isna(ts):
        return ""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert(timezone_str)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)

# ================================
# РАБОТА С ДАННЫМИ: IB, ОКНО ТОРГОВЛИ
# ================================
def compute_ib(day_df: pd.DataFrame, day_date: datetime.date,
               start_str: str, end_str: str, timezone_str: str) -> Optional[Dict[str, float]]:
    """Вычисление Initial Balance для заданного времени и тайм-зоны"""
    start_ib, end_ib = ib_window_on_date(day_date, start_str, end_str, timezone_str)
    df_ib = day_df[(day_df["time"] >= start_ib) & (day_df["time"] < end_ib)]
    if df_ib.empty:
        return None
    ib_high = float(df_ib["high"].max())
    ib_low = float(df_ib["low"].min())
    eq = (ib_high + ib_low) / 2.0
    return {"IBH": ib_high, "IBL": ib_low, "EQ": eq}

def get_trade_window(day_df: pd.DataFrame, day_date: datetime.date,
                     ib_end_str: str, timezone_str: str,
                     wait_minutes: int, duration_minutes: int) -> pd.DataFrame:
    """Получить данные торгового окна"""
    start_trade, end_trade = trade_window_on_date(
        day_date, ib_end_str, timezone_str, wait_minutes, duration_minutes
    )
    return day_df[(day_df["time"] >= start_trade) & (day_df["time"] < end_trade)].copy()

# ================================
# ЛОГИКА REVERSE
# ================================
def process_reverse_signal_fixed(
    df_trade: pd.DataFrame,
    ibh: float,
    ibl: float,
    eq: float,
    params: Dict[str, Any],
    df_context_before: Optional[pd.DataFrame] = None,
    ib_buffer_pct: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Reverse (реверс от IB) — M2 с кастомными параметрами

    Args:
        df_trade: Trade window DataFrame
        ibh: IB High
        ibl: IB Low
        eq: Equilibrium level
        params: Strategy parameters
        df_context_before: Context data before trade window
        ib_buffer_pct: Buffer zone for sweep validation.
                       Sweep is valid if shadow goes beyond IB but close stays within buffer zone.
                       Upper sweep: high > ibh AND close <= ibh + buffer
                       Lower sweep: low < ibl AND close >= ibl - buffer
    """
    ib_range = float(ibh - ibl)
    if ib_range <= 0 or df_trade.empty:
        return None

    buffer = ib_buffer_pct * ib_range
    n = len(df_trade)

    # 1) Собираем свипы до первой инвалидации
    sweeps: List[Dict[str, Any]] = []
    invalid_at: Optional[int] = None

    for i in range(n):
        o = float(df_trade["open"].iat[i])
        c = float(df_trade["close"].iat[i])
        h = float(df_trade["high"].iat[i])
        l = float(df_trade["low"].iat[i])

        # Инвалидация: и open, и close за пределами IB+buffer по одну сторону
        if (o > ibh + buffer and c > ibh + buffer) or (o < ibl - buffer and c < ibl - buffer):
            invalid_at = i
            break

        # Upper sweep (шорт сценарий): shadow above IBH but close within buffer zone
        if h > ibh and o <= ibh + buffer and c <= ibh + buffer:
            sweeps.append({"dir": "upper", "idx": i, "ext": h})

        # Lower sweep (лонг сценарий): shadow below IBL but close within buffer zone
        if l < ibl and o >= ibl - buffer and c >= ibl - buffer:
            sweeps.append({"dir": "lower", "idx": i, "ext": l})

    if invalid_at is not None:
        sweeps = [s for s in sweeps if s["idx"] < invalid_at]

    if not sweeps:
        return None

    # 2) Группируем подряд свипы одного направления
    groups: List[Dict[str, Any]] = []
    cur = None
    for s in sweeps:
        if cur is None:
            cur = {"dir": s["dir"], "start": s["idx"], "end": s["idx"], "ext": s["ext"]}
        else:
            if s["dir"] == cur["dir"] and s["idx"] == cur["end"] + 1:
                cur["end"] = s["idx"]
                if s["dir"] == "upper":
                    cur["ext"] = max(cur["ext"], s["ext"])
                else:
                    cur["ext"] = min(cur["ext"], s["ext"])
            else:
                groups.append(cur)
                cur = {"dir": s["dir"], "start": s["idx"], "end": s["idx"], "ext": s["ext"]}
    if cur is not None:
        groups.append(cur)

    # 3) Для каждой группы ищем CISD
    for g in groups:
        prev_trade = df_trade.iloc[:g["start"]][["time","open","high","low","close"]]
        if df_context_before is not None and not df_context_before.empty:
            ctx = pd.concat([df_context_before[["time","open","high","low","close"]], prev_trade], ignore_index=True)
        else:
            ctx = prev_trade.reset_index(drop=True)

        # Последняя противоположная свеча
        last_opp = None
        for j in range(len(ctx) - 1, -1, -1):
            ro = float(ctx["open"].iat[j])
            rc = float(ctx["close"].iat[j])
            if g["dir"] == "upper":
                if rc < ro:  # bearish
                    last_opp = {"low": float(ctx["low"].iat[j]), "high": float(ctx["high"].iat[j])}
                    break
            else:
                if rc > ro:  # bullish
                    last_opp = {"low": float(ctx["low"].iat[j]), "high": float(ctx["high"].iat[j])}
                    break

        if last_opp is None:
            continue

        # 4) Ищем CISD ПОСЛЕ окончания группы свипов
        for k in range(g["end"] + 1, n):
            ck = float(df_trade["close"].iat[k])

            if g["dir"] == "upper":   # SHORT
                if ck < last_opp["low"]:
                    distance = ibh - ck
                    if distance >= 0.5 * ib_range:
                        break
                    return {
                        "type": "Reverse",
                        "direction": "short",
                        "cisd_idx": k,
                        "cisd_time": df_trade["time"].iat[k],
                        "sweep_extreme": float(g["ext"]),
                        "sweep_direction": "upper",
                    }

            else:                     # LONG
                if ck > last_opp["high"]:
                    distance = ck - ibl
                    if distance >= 0.5 * ib_range:
                        break
                    return {
                        "type": "Reverse",
                        "direction": "long",
                        "cisd_idx": k,
                        "cisd_time": df_trade["time"].iat[k],
                        "sweep_extreme": float(g["ext"]),
                        "sweep_direction": "lower",
                    }

    return None

# ================================
# ПОМОЩНИКИ ПО СИГНАЛАМ
# ================================
def eq_touched_before_idx(df: pd.DataFrame, eq: float, idx: int) -> bool:
    sub = df.iloc[:idx + 1]
    return bool(np.any((sub["low"] <= eq) & (sub["high"] >= eq)))

def first_breakout_bar(
    df_trade: pd.DataFrame,
    ibh: float,
    ibl: float,
    ib_buffer_pct: float = 0.0,
    max_distance_pct: float = 1.0
) -> Optional[Tuple[int, str]]:
    """
    Finds first candle with valid breakout.

    Args:
        df_trade: Trade window DataFrame
        ibh: IB High
        ibl: IB Low
        ib_buffer_pct: Minimum breakout filter - close must be > IBH + buffer (or < IBL - buffer)
        max_distance_pct: Maximum distance filter - reject if close is too far from IB boundary

    Returns:
        Tuple of (index, direction) or None
    """
    ib_range = ibh - ibl
    if ib_range <= 0:
        return None

    buffer = ib_buffer_pct * ib_range
    max_dist = max_distance_pct * ib_range

    for i in range(len(df_trade)):
        c = float(df_trade["close"].iat[i])

        # Long breakout
        if c > ibh + buffer:
            distance = c - ibh
            if distance <= max_dist:
                return (i, "long")

        # Short breakout
        if c < ibl - buffer:
            distance = ibl - c
            if distance <= max_dist:
                return (i, "short")

    return None

def tcwe_second_further_idx(
    df_trade: pd.DataFrame,
    ibh: float,
    ibl: float,
    eq: float,
    ib_buffer_pct: float = 0.0,
    max_distance_pct: float = 1.0
) -> Optional[Tuple[int, str]]:
    """
    Two Candles without Equilibrium: second candle goes further than first (no EQ touch before).

    Args:
        df_trade: Trade window DataFrame
        ibh: IB High
        ibl: IB Low
        eq: Equilibrium level
        ib_buffer_pct: Minimum breakout filter - close must be > IBH + buffer
        max_distance_pct: Maximum distance filter - reject if close is too far from IB

    Returns:
        Tuple of (index, direction) or None
    """
    n = len(df_trade)
    ib_range = ibh - ibl
    if ib_range <= 0:
        return None

    buffer = ib_buffer_pct * ib_range
    max_dist = max_distance_pct * ib_range

    first_idx_long = None
    first_idx_short = None

    for i in range(n):
        c = float(df_trade["close"].iat[i])
        # Check long breakout with buffer
        if (c > ibh + buffer) and not eq_touched_before_idx(df_trade, eq, i):
            distance = c - ibh
            if distance <= max_dist:
                first_idx_long = i
                break
        # Check short breakout with buffer
        if (c < ibl - buffer) and not eq_touched_before_idx(df_trade, eq, i):
            distance = ibl - c
            if distance <= max_dist:
                first_idx_short = i
                break

    if first_idx_long is not None:
        c1 = float(df_trade["close"].iat[first_idx_long])
        for j in range(first_idx_long + 1, n):
            c2 = float(df_trade["close"].iat[j])
            if eq_touched_before_idx(df_trade, eq, j):
                return None
            # Second candle must also pass buffer and distance checks
            if (c2 > ibh + buffer) and (c2 > c1):
                distance = c2 - ibh
                if distance <= max_dist:
                    return j, "long"

    if first_idx_short is not None:
        c1 = float(df_trade["close"].iat[first_idx_short])
        for j in range(first_idx_short + 1, n):
            c2 = float(df_trade["close"].iat[j])
            if eq_touched_before_idx(df_trade, eq, j):
                return None
            # Second candle must also pass buffer and distance checks
            if (c2 < ibl - buffer) and (c2 < c1):
                distance = ibl - c2
                if distance <= max_dist:
                    return j, "short"

    return None

# ================================
# CISD для обычных стратегий
# ================================
def find_cisd_level(df: pd.DataFrame, direction: str, current_idx: int,
                    entry_price: float) -> Optional[float]:
    """
    Поиск уровня CISD для стоп-лосса (для обычных стратегий)
    """
    if current_idx <= 0:
        return None

    for i in range(current_idx - 1, -1, -1):
        row = df.iloc[i]
        if direction == "long":
            if row['close'] < row['open']:
                return float(row['low'])
        else:  # short
            if row['close'] > row['open']:
                return float(row['high'])
    return None

# ================================
# ЭКЗЕКЬЮШН И ЭКЗИТЫ
# ================================
def initial_stop_price(trade_start_price: float, eq_price: float, cisd_price: Optional[float], stop_mode: str) -> float:
    """Начальная цена стопа в зависимости от STOP_MODE"""
    if stop_mode.lower() == "eq":
        return float(eq_price)
    elif stop_mode.lower() == "cisd" and cisd_price is not None:
        return float(cisd_price)
    else:
        return float(trade_start_price)

def place_sl_tp_with_min_size(direction: str, entry_price: float, stop_price: float,
                              rr_target: float, min_sl_pct: float) -> Optional[Tuple[float, float, bool]]:
    """
    Размещение SL и TP с учетом минимального размера SL.

    CRITICAL: Validates that SL is on correct side of entry:
    - LONG: SL must be BELOW entry
    - SHORT: SL must be ABOVE entry

    If SL is on wrong side, it is flipped to correct side with same distance.
    """
    min_sl_size = entry_price * min_sl_pct
    adjusted = False

    # Step 1: Validate SL is on correct side for direction
    if direction == "long":
        # For LONG, SL must be BELOW entry
        if stop_price >= entry_price:
            # SL is above or at entry - flip to correct side
            stop_price = entry_price - abs(entry_price - stop_price)
            adjusted = True
    else:
        # For SHORT, SL must be ABOVE entry
        if stop_price <= entry_price:
            # SL is below or at entry - flip to correct side
            stop_price = entry_price + abs(entry_price - stop_price)
            adjusted = True

    # Step 2: Calculate risk (now guaranteed positive in correct direction)
    risk = abs(entry_price - stop_price)

    # Step 3: Apply minimum SL size if needed
    if risk < min_sl_size:
        adjusted = True
        if direction == "long":
            stop_price = entry_price - min_sl_size
        else:
            stop_price = entry_price + min_sl_size
        risk = min_sl_size

    # Step 4: Calculate TP
    tp = entry_price + rr_target * risk if direction == "long" else entry_price - rr_target * risk

    return float(stop_price), float(tp), adjusted

def simulate_after_entry(df_trade: pd.DataFrame, start_idx: int, direction: str, entry_price: float,
                         stop: float, tp: float, tsl_target: float, tsl_sl: float) -> Dict[str, Any]:
    """
    Трейлинг-логика (ступенями) + режим без трейлинга.
    """
    risk = abs(entry_price - stop)
    if risk <= 0:
        return {"exit_reason": "invalid", "exit_time": df_trade["time"].iat[start_idx],
                "exit_price": entry_price, "R": 0.0}

    curr_stop = float(stop)
    curr_target_R = float(tp / risk) if direction == "long" else float(tp / risk)
    curr_tp = float(tp)
    no_trail = (tsl_target is None) or (float(tsl_target) <= 0.0)

    exit_next_open = False

    for i in range(start_idx + 1, len(df_trade)):
        lo = float(df_trade["low"].iat[i])
        hi = float(df_trade["high"].iat[i])
        t = df_trade["time"].iat[i]

        # Отложенный выход на следующей свече
        if exit_next_open:
            exit_price = float(df_trade["open"].iat[i])
            r = (exit_price - entry_price) / risk if direction == "long" else (entry_price - exit_price) / risk
            return {"exit_reason": "trail_conflict", "exit_time": t, "exit_price": exit_price, "R": r}

        # 1) Стоп
        if direction == "long":
            if lo <= curr_stop:
                r = (curr_stop - entry_price) / risk
                return {"exit_reason": "stop", "exit_time": t, "exit_price": curr_stop, "R": r}
        else:
            if hi >= curr_stop:
                r = (entry_price - curr_stop) / risk
                return {"exit_reason": "stop", "exit_time": t, "exit_price": curr_stop, "R": r}

        # 2) Таргет(ы)
        if no_trail:
            if (direction == "long" and hi >= curr_tp) or (direction == "short" and lo <= curr_tp):
                r = curr_target_R
                return {"exit_reason": "tp", "exit_time": t, "exit_price": curr_tp, "R": r}
        else:
            if direction == "long":
                hit = False
                while hi >= curr_tp:
                    hit = True
                    last_tp = curr_tp
                    new_stop = last_tp - tsl_sl * risk
                    curr_stop = max(curr_stop, new_stop)
                    curr_target_R += tsl_target
                    curr_tp = entry_price + curr_target_R * risk
                    if tsl_target <= 0:
                        break
                if hit and lo <= curr_stop:
                    exit_next_open = True
            else:
                hit = False
                while lo <= curr_tp:
                    hit = True
                    last_tp = curr_tp
                    new_stop = last_tp + tsl_sl * risk
                    curr_stop = min(curr_stop, new_stop)
                    curr_target_R += tsl_target
                    curr_tp = entry_price - curr_target_R * risk
                    if tsl_target <= 0:
                        break
                if hit and hi >= curr_stop:
                    exit_next_open = True

    # 3) Окончание окна — закрытие по последней цене
    last_close = float(df_trade["close"].iat[-1])
    r = (last_close - entry_price) / risk if direction == "long" else (entry_price - last_close) / risk
    return {"exit_reason": "time", "exit_time": df_trade["time"].iat[-1], "exit_price": last_close, "R": r}

# ================================
# РЕВЕРС-СДЕЛКИ ПОСЛЕ REVERSE_BLOCKED (REV_RB)
# ================================
def simulate_reverse_limit_both_sides(df_trade: pd.DataFrame,
                                      ibh: float, ibl: float, eq: float,
                                      reverse_block_time: pd.Timestamp,
                                      params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    REV_RB сделки - лимитные ордера после reverse_blocked.
    """
    if reverse_block_time is None or df_trade.empty:
        return None

    # стартовый индекс для поиска триггера
    start_idx = int(np.searchsorted(df_trade["time"].values, reverse_block_time.to_datetime64(), side="left"))

    ib_range = float(ibh - ibl)
    if ib_range <= 0:
        return None

    pct = params["REV_RB_PCT"]
    ext_up = ibh + pct * ib_range
    ext_dn = ibl - pct * ib_range

    # найти первый триггер в обе стороны
    j_up = None
    j_dn = None
    for j in range(start_idx, len(df_trade)):
        if j_up is None and float(df_trade["high"].iat[j]) >= ext_up:
            j_up = j
        if j_dn is None and float(df_trade["low"].iat[j]) <= ext_dn:
            j_dn = j
        if j_up is not None and j_dn is not None:
            break

    # выбрать более ранний триггер
    chosen = None
    if j_up is not None and j_dn is not None:
        chosen = ("long", j_up) if j_up <= j_dn else ("short", j_dn)
    elif j_up is not None:
        chosen = ("long", j_up)
    elif j_dn is not None:
        chosen = ("short", j_dn)
    else:
        return None

    direction, trig_idx = chosen

    if direction == "long":
        entry_level = float(ibh)
        stop = float(eq)  # SL=EQ для REV_RB
        risk = abs(entry_level - stop)

        # Проверка минимального размера SL
        min_sl_size = entry_level * params["MIN_SL_PCT"]
        if risk < min_sl_size:
            stop = entry_level - min_sl_size
            risk = min_sl_size

        tp = entry_level + params["RR_TARGET"] * risk

        # ждём возврата к ibh для лимитного исполнения
        fill_idx = None
        for k in range(trig_idx + 1, len(df_trade)):
            if float(df_trade["low"].iat[k]) <= entry_level:
                fill_idx = k
                break
        if fill_idx is None:
            return None

        sim = simulate_after_entry(df_trade, fill_idx, "long", entry_level, stop, tp, 
                                  params["TSL_TARGET"], params["TSL_SL"])
        return {
            "variation": "REV_RB",
            "direction": "long",
            "entry_time": df_trade["time"].iat[fill_idx],
            "entry_price": entry_level,
            "stop": stop,
            "tp": tp,
            **sim
        }

    else:  # short
        entry_level = float(ibl)
        stop = float(eq)
        risk = abs(entry_level - stop)

        min_sl_size = entry_level * params["MIN_SL_PCT"]
        if risk < min_sl_size:
            stop = entry_level + min_sl_size
            risk = min_sl_size

        tp = entry_level - params["RR_TARGET"] * risk

        fill_idx = None
        for k in range(trig_idx + 1, len(df_trade)):
            if float(df_trade["high"].iat[k]) >= entry_level:
                fill_idx = k
                break
        if fill_idx is None:
            return None

        sim = simulate_after_entry(df_trade, fill_idx, "short", entry_level, stop, tp,
                                  params["TSL_TARGET"], params["TSL_SL"])
        return {
            "variation": "REV_RB",
            "direction": "short",
            "entry_time": df_trade["time"].iat[fill_idx],
            "entry_price": entry_level,
            "stop": stop,
            "tp": tp,
            **sim
        }

# ================================
# ДЕНЬ: ПОИСК ВХОДА И ПРОГОН
# ================================
def process_day_instrument(day_df_all: pd.DataFrame, day_date: datetime.date,
                           instrument: str, params_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Обрабатывает один день для одного инструмента с кастомными параметрами
    """
    # Пытаемся найти сигнал для каждой вариации в порядке приоритета

    # 1) Определяем timezone и IB параметры из первой вариации (предполагаем одинаковые для инструмента)
    timezone_str = params_dict["Reverse"]["IB_TZ"]
    ib_start = params_dict["Reverse"]["IB_START"]
    ib_end = params_dict["Reverse"]["IB_END"]

    def make_ib_params(var_params):
        return f"{var_params['IB_START']}-{var_params['IB_END']} {var_params['IB_TZ']}"

    # 2) IB
    ib = compute_ib(day_df_all, day_date, ib_start, ib_end, timezone_str)
    if ib is None:
        return {"date": day_date, "instrument": instrument, "status": "no_ib", "trades": []}
    ibh, ibl, eq = ib["IBH"], ib["IBL"], ib["EQ"]

    # Пробуем Reverse
    reverse_params = params_dict["Reverse"]
    df_trade_reverse = get_trade_window(day_df_all, day_date, reverse_params["IB_END"], reverse_params["IB_TZ"],
                                        reverse_params["IB_WAIT"], reverse_params["TRADE_WINDOW"])

    if not df_trade_reverse.empty:
        # Контекст до окна
        ib_start_ts, ib_end_ts = ib_window_on_date(day_date, reverse_params["IB_START"],
                                                   reverse_params["IB_END"], reverse_params["IB_TZ"])
        first_trade_ts = df_trade_reverse["time"].iat[0]
        df_pre_context = day_df_all[
            (day_df_all["time"] >= ib_start_ts) & (day_df_all["time"] < first_trade_ts)
            ][["time", "open", "high", "low", "close"]].copy()

        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = reverse_params.get("IB_BUFFER_PCT", 0.0)
        reverse_signal = process_reverse_signal_fixed(
            df_trade_reverse, ibh, ibl, eq, reverse_params, df_pre_context,
            ib_buffer_pct=ib_buffer_pct
        )

        if reverse_signal is not None:
            cisd_idx = reverse_signal["cisd_idx"]
            direction = reverse_signal["direction"]
            sweep_extreme = reverse_signal["sweep_extreme"]

            if cisd_idx + 1 < len(df_trade_reverse):
                entry_idx = cisd_idx + 1
                entry_time = df_trade_reverse["time"].iat[entry_idx]
                entry_price = float(df_trade_reverse["open"].iat[entry_idx])

                # SL на экстремуме свипа
                stop_price = float(sweep_extreme)
                stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                               reverse_params["RR_TARGET"],
                                                               reverse_params["MIN_SL_PCT"])

                sim = simulate_after_entry(df_trade_reverse, entry_idx, direction, entry_price, stop, tp,
                                           reverse_params["TSL_TARGET"], reverse_params["TSL_SL"])
                trade = {
                    "variation": "Reverse",
                    "instrument": instrument,
                    "direction": direction,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "stop": stop,
                    "tp": tp,
                    "sl_adjusted": adjusted,
                    "ib_params": make_ib_params(reverse_params),
                    **sim
                }
                return {
                    "date": day_date, "instrument": instrument, "status": "done_reverse",
                    "ib_high": ibh, "ib_low": ibl, "eq": eq,
                    "trades": [trade],
                }

    # Пробуем OCAE
    ocae_params = params_dict["OCAE"]
    df_trade_ocae = get_trade_window(day_df_all, day_date, ocae_params["IB_END"], ocae_params["IB_TZ"],
                                     ocae_params["IB_WAIT"], ocae_params["TRADE_WINDOW"])

    if not df_trade_ocae.empty:
        trade_start_price = float(df_trade_ocae["open"].iat[0])
        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = ocae_params.get("IB_BUFFER_PCT", 0.0)
        max_distance_pct = ocae_params.get("MAX_DISTANCE_PCT", 1.0)
        br_result = first_breakout_bar(df_trade_ocae, ibh, ibl, ib_buffer_pct, max_distance_pct)
        if br_result is not None:
            br_idx, direction = br_result
            if eq_touched_before_idx(df_trade_ocae, eq, br_idx):
                entry_candle = df_trade_ocae.iloc[br_idx]
                entry_time = entry_candle["time"]
                entry_price = float(entry_candle["close"])

                cisd_stop = find_cisd_level(df_trade_ocae, direction, br_idx, entry_price) if ocae_params[
                                                                                                  "STOP_MODE"].lower() == "cisd" else None
                stop_price = initial_stop_price(trade_start_price, eq, cisd_stop, ocae_params["STOP_MODE"])
                stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                               ocae_params["RR_TARGET"], ocae_params["MIN_SL_PCT"])

                sim = simulate_after_entry(df_trade_ocae, br_idx, direction, entry_price, stop, tp,
                                           ocae_params["TSL_TARGET"], ocae_params["TSL_SL"])
                trade = {
                    "variation": "OCAE",
                    "instrument": instrument,
                    "direction": direction,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "stop": stop,
                    "tp": tp,
                    "sl_adjusted": adjusted,
                    "ib_params": make_ib_params(ocae_params),
                    **sim
                }
                return {
                    "date": day_date, "instrument": instrument, "status": "done",
                    "ib_high": ibh, "ib_low": ibl, "eq": eq,
                    "trade_start": trade_start_price, "trades": [trade],
                }

    # Пробуем TCWE
    tcwe_params = params_dict["TCWE"]
    df_trade_tcwe = get_trade_window(day_df_all, day_date, tcwe_params["IB_END"], tcwe_params["IB_TZ"],
                                     tcwe_params["IB_WAIT"], tcwe_params["TRADE_WINDOW"])

    if not df_trade_tcwe.empty:
        trade_start_price = float(df_trade_tcwe["open"].iat[0])
        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = tcwe_params.get("IB_BUFFER_PCT", 0.0)
        max_distance_pct = tcwe_params.get("MAX_DISTANCE_PCT", 1.0)
        tc = tcwe_second_further_idx(df_trade_tcwe, ibh, ibl, eq, ib_buffer_pct, max_distance_pct)
        if tc is not None:
            idx2, direction = tc
            entry_candle = df_trade_tcwe.iloc[idx2]
            entry_time = entry_candle["time"]
            entry_price = float(entry_candle["close"])

            cisd_stop = find_cisd_level(df_trade_tcwe, direction, idx2, entry_price) if tcwe_params[
                                                                                            "STOP_MODE"].lower() == "cisd" else None
            stop_price = initial_stop_price(trade_start_price, eq, cisd_stop, tcwe_params["STOP_MODE"])
            stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                           tcwe_params["RR_TARGET"], tcwe_params["MIN_SL_PCT"])

            sim = simulate_after_entry(df_trade_tcwe, idx2, direction, entry_price, stop, tp,
                                       tcwe_params["TSL_TARGET"], tcwe_params["TSL_SL"])
            trade = {
                "variation": "TCWE",
                "instrument": instrument,
                "direction": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop": stop,
                "tp": tp,
                "sl_adjusted": adjusted,
                "ib_params": make_ib_params(tcwe_params),
                **sim
            }
            return {
                "date": day_date, "instrument": instrument, "status": "done",
                "ib_high": ibh, "ib_low": ibl, "eq": eq,
                "trade_start": trade_start_price, "trades": [trade],
            }

    # Пробуем REV_RB
    rev_rb_params = params_dict["REV_RB"]
    if rev_rb_params["REV_RB_ENABLED"]:
        df_trade_rev = get_trade_window(day_df_all, day_date, rev_rb_params["IB_END"], rev_rb_params["IB_TZ"],
                                        rev_rb_params["IB_WAIT"], rev_rb_params["TRADE_WINDOW"])
        if not df_trade_rev.empty:
            fake_block_time = df_trade_rev["time"].iat[0]
            rev_trade = simulate_reverse_limit_both_sides(df_trade_rev, ibh, ibl, eq, fake_block_time, rev_rb_params)
            if rev_trade is not None:
                rev_trade["instrument"] = instrument
                rev_trade["ib_params"] = make_ib_params(rev_rb_params)
                return {
                    "date": day_date, "instrument": instrument, "status": "done_rev_rb",
                    "ib_high": ibh, "ib_low": ibl, "eq": eq,
                    "trades": [rev_trade],
                }

    # Ничего не сработало
    return {
        "date": day_date, "instrument": instrument, "status": "no_entry",
        "ib_high": ibh, "ib_low": ibl, "eq": eq,
        "trades": [],
    }
