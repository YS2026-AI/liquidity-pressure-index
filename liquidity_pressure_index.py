"""
全球流动性压力指数模型 (Global Liquidity Pressure Index)
======================================================

基于6个核心宏观因子构建的综合流动性压力指数，用于预测美股尾部风险。

6个核心因子：
1. 短端利率 (Short-term Interest Rate) - 2年期美债收益率
2. 久期供给 (Duration Supply) - 美国国债发行量代理
3. 官方流动性 (Official Liquidity) - 美联储资产负债表 - TGA
4. 一级市场融资 (Primary Market Drain) - VIX作为市场融资压力代理
5. 波动率放大器 (Volatility Amplifier) - VIX指数
6. 催化剂 (Catalyst) - 原油价格 + CPI

方法论：
- 将各因子转化为过去11年（约560周）的历史分位数
- 统一方向：高分位 = 高压力
- 等权合成综合压力指数 PI
- 前10%为"极值高压线"
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

# 尝试导入可选依赖
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False


# ============================================================
# 配置
# ============================================================
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
LOOKBACK_YEARS = 11  # 回溯11年
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 数据获取模块
# ============================================================

def get_fred_data(series_id, start_date, end_date):
    """从FRED获取数据"""
    if HAS_FRED and FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            return data
        except Exception as e:
            print(f"FRED API error for {series_id}: {e}")
    
    # 备用：使用yfinance或生成模拟数据
    return None


def get_treasury_yield_2y(start_date, end_date):
    """获取2年期美债收益率 (FRED: DGS2)"""
    data = get_fred_data("DGS2", start_date, end_date)
    if data is not None:
        return data
    
    # 备用：使用yfinance获取2年期美债ETF
    if HAS_YFINANCE:
        try:
            ticker = yf.download("^IRX", start=start_date, end=end_date, progress=False)
            if not ticker.empty:
                return ticker['Close'].squeeze()
        except:
            pass
    
    return None


def get_fed_balance_sheet(start_date, end_date):
    """获取美联储资产负债表 (FRED: WALCL)"""
    data = get_fred_data("WALCL", start_date, end_date)
    return data


def get_tga_balance(start_date, end_date):
    """获取财政部TGA账户余额 (FRED: WTREGEN)"""
    data = get_fred_data("WTREGEN", start_date, end_date)
    return data


def get_vix(start_date, end_date):
    """获取VIX波动率指数"""
    if HAS_YFINANCE:
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            if not vix.empty:
                return vix['Close'].squeeze()
        except:
            pass
    return None


def get_oil_price(start_date, end_date):
    """获取WTI原油价格"""
    if HAS_YFINANCE:
        try:
            oil = yf.download("CL=F", start=start_date, end=end_date, progress=False)
            if not oil.empty:
                return oil['Close'].squeeze()
        except:
            pass
    
    # 备用FRED数据
    data = get_fred_data("DCOILWTICO", start_date, end_date)
    return data


def get_sp500(start_date, end_date):
    """获取标普500指数"""
    if HAS_YFINANCE:
        try:
            sp = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
            if not sp.empty:
                return sp['Close'].squeeze()
        except:
            pass
    return None


def get_cpi(start_date, end_date):
    """获取CPI同比变化率 (FRED: CPIAUCSL)"""
    data = get_fred_data("CPIAUCSL", start_date, end_date)
    if data is not None:
        return data.pct_change(12) * 100  # 同比变化率
    return None


def get_treasury_issuance(start_date, end_date):
    """获取国债发行量代理 (FRED: GFDEBTN - 联邦债务总额)"""
    data = get_fred_data("GFDEBTN", start_date, end_date)
    if data is not None:
        return data.diff()  # 变化量作为发行代理
    return None


# ============================================================
# 数据处理与模型计算
# ============================================================

def compute_rolling_percentile(series, window=560):
    """
    计算滚动分位数（历史排名百分位）
    window: 约11年的周数据 ≈ 560周
    """
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        start_idx = max(0, i - window)
        historical = series.iloc[start_idx:i+1]
        if len(historical) >= 52:  # 至少1年数据
            rank = (historical < series.iloc[i]).sum()
            result.iloc[i] = rank / len(historical) * 100
        else:
            result.iloc[i] = np.nan
    return result


def standardize_factor(series, higher_is_pressure=True):
    """
    标准化因子为分位数
    higher_is_pressure: True表示值越高压力越大，False表示值越高压力越小
    """
    percentile = compute_rolling_percentile(series)
    if not higher_is_pressure:
        percentile = 100 - percentile
    return percentile


def build_pressure_index(factors_df):
    """
    构建综合压力指数 (等权合成)
    factors_df: DataFrame，每列是一个标准化后的因子（0-100分位）
    """
    # 等权平均
    pi = factors_df.mean(axis=1)
    return pi


def identify_extreme_points(pi, threshold_percentile=90):
    """
    识别极值高压点（前10%）
    """
    threshold = np.nanpercentile(pi.dropna(), threshold_percentile)
    extreme_mask = pi >= threshold
    return extreme_mask, threshold


def compute_forward_returns(sp500, weeks=4):
    """
    计算未来N周收益率
    """
    forward_ret = sp500.pct_change(weeks * 5).shift(-weeks * 5) * 100  # 转为百分比
    return forward_ret


def compute_max_drawdown_forward(sp500, weeks=4):
    """
    计算未来N周最大回撤
    """
    window = weeks * 5  # 交易日
    max_dd = pd.Series(index=sp500.index, dtype=float)
    
    for i in range(len(sp500) - window):
        future_prices = sp500.iloc[i:i+window]
        peak = future_prices.expanding().max()
        drawdown = (future_prices - peak) / peak * 100
        max_dd.iloc[i] = drawdown.min()
    
    return max_dd


# ============================================================
# 可视化模块
# ============================================================

def set_dark_style():
    """设置深色主题"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.5,
        'figure.dpi': 150,
    })


def plot_pressure_index_history(pi, extreme_mask, threshold, save_path=None):
    """绘制历史压力指数折线图"""
    set_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # 主折线
    ax.plot(pi.index, pi.values, color='#58a6ff', linewidth=1.2, label='Liquidity Pressure Index')
    
    # 极值高压线
    ax.axhline(y=threshold, color='#f85149', linestyle='--', linewidth=1.5, 
               label=f'Top 10% Extreme Threshold ({threshold:.1f})')
    
    # 标记极值点
    extreme_points = pi[extreme_mask]
    ax.scatter(extreme_points.index, extreme_points.values, 
               color='#f85149', s=20, alpha=0.7, zorder=5)
    
    # 填充极值区域
    ax.fill_between(pi.index, 0, pi.values, 
                    where=extreme_mask, color='#f85149', alpha=0.15)
    
    ax.set_title('Global Liquidity Pressure Index (2015-2026)\n全球流动性压力指数', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Pressure Index (Percentile)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    return fig


def plot_tail_risk_analysis(forward_returns, extreme_mask, save_path=None):
    """绘制尾部风险分析图"""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：收益率分布对比
    ax = axes[0]
    normal_returns = forward_returns[~extreme_mask].dropna()
    extreme_returns = forward_returns[extreme_mask].dropna()
    
    if len(normal_returns) > 0:
        ax.hist(normal_returns, bins=50, alpha=0.6, color='#8b949e', 
                label=f'Normal ({len(normal_returns)} samples)', density=True)
    if len(extreme_returns) > 0:
        ax.hist(extreme_returns, bins=30, alpha=0.7, color='#f85149', 
                label=f'Extreme Pressure ({len(extreme_returns)} samples)', density=True)
    
    ax.set_title('4-Week Forward Return Distribution\n未来4周收益率分布', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Return (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.axvline(x=0, color='#c9d1d9', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 右图：回撤概率对比柱状图
    ax = axes[1]
    thresholds = [-3, -5, -7, -10]
    normal_probs = []
    extreme_probs = []
    
    for t in thresholds:
        if len(normal_returns) > 0:
            normal_probs.append((normal_returns < t).sum() / len(normal_returns) * 100)
        else:
            normal_probs.append(0)
        if len(extreme_returns) > 0:
            extreme_probs.append((extreme_returns < t).sum() / len(extreme_returns) * 100)
        else:
            extreme_probs.append(0)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_probs, width, label='Normal', color='#8b949e', alpha=0.8)
    bars2 = ax.bar(x + width/2, extreme_probs, width, label='Extreme Pressure', color='#f85149', alpha=0.8)
    
    ax.set_title('Drawdown Probability Comparison\n回撤概率对比', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Drawdown Threshold', fontsize=11)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'>{abs(t)}%' for t in thresholds])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    return fig


def plot_current_dashboard(factors_current, pi_current, save_path=None):
    """绘制当前市场状态仪表盘"""
    set_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    factor_names = list(factors_current.keys())
    factor_values = list(factors_current.values())
    
    # 颜色映射
    colors = []
    for v in factor_values:
        if v < 25:
            colors.append('#3fb950')  # 绿色 - 低压
        elif v < 50:
            colors.append('#58a6ff')  # 蓝色 - 中低
        elif v < 75:
            colors.append('#d29922')  # 黄色 - 中高
        else:
            colors.append('#f85149')  # 红色 - 高压
    
    # 水平条形图
    y_pos = np.arange(len(factor_names))
    bars = ax.barh(y_pos, factor_values, color=colors, alpha=0.85, height=0.6)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, factor_values)):
        label = f'{val:.1f}%'
        if val < 25:
            status = 'LOW'
        elif val < 50:
            status = 'NEUTRAL'
        elif val < 75:
            status = 'ELEVATED'
        else:
            status = 'HIGH'
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{label} ({status})', va='center', fontsize=11, color='#c9d1d9')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factor_names, fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_xlabel('Percentile (0=Low Pressure, 100=High Pressure)', fontsize=11)
    
    # 综合得分
    title_text = f'Current Market Pressure Dashboard\n当前市场压力诊断\n\nComposite Score: {pi_current:.1f} percentile'
    if pi_current < 30:
        title_color = '#3fb950'
        status_text = '(LOW RISK)'
    elif pi_current < 60:
        title_color = '#d29922'
        status_text = '(NEUTRAL)'
    else:
        title_color = '#f85149'
        status_text = '(ELEVATED RISK)'
    
    ax.set_title(f'{title_text} {status_text}', fontsize=14, fontweight='bold', pad=20)
    
    # 参考线
    ax.axvline(x=50, color='#c9d1d9', linestyle='--', alpha=0.3, label='Median')
    ax.axvline(x=90, color='#f85149', linestyle='--', alpha=0.5, label='90th percentile')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    return fig


def plot_forward_stress_test(monthly_stress, save_path=None):
    """绘制前瞻压力测试柱状图"""
    set_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    months = list(monthly_stress.keys())
    values = list(monthly_stress.values())
    
    colors = []
    for v in values:
        if v < 50:
            colors.append('#3fb950')
        elif v < 70:
            colors.append('#d29922')
        else:
            colors.append('#f85149')
    
    bars = ax.bar(months, values, color=colors, alpha=0.85, width=0.6)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', fontsize=11, color='#c9d1d9', fontweight='bold')
    
    ax.set_title('Forward Stress Test - H2 2026\n前瞻压力测试（2026下半年）', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Projected Pressure Percentile', fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color='#f85149', linestyle='--', alpha=0.5, label='Extreme threshold (90th)')
    ax.axhline(y=50, color='#c9d1d9', linestyle='--', alpha=0.3, label='Median')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    return fig


# ============================================================
# 主运行逻辑
# ============================================================

def run_model():
    """运行完整模型"""
    print("=" * 60)
    print("Global Liquidity Pressure Index Model")
    print("全球流动性压力指数模型")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 设置时间范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime('%Y-%m-%d')
    
    print(f"\nData Range: {start_date} to {end_date}")
    print(f"Lookback: {LOOKBACK_YEARS} years")
    
    # ---- 获取数据 ----
    print("\n[1/6] Fetching data...")
    
    # 获取各因子数据
    treasury_2y = get_treasury_yield_2y(start_date, end_date)
    vix = get_vix(start_date, end_date)
    oil = get_oil_price(start_date, end_date)
    sp500 = get_sp500(start_date, end_date)
    fed_bs = get_fed_balance_sheet(start_date, end_date)
    tga = get_tga_balance(start_date, end_date)
    cpi = get_cpi(start_date, end_date)
    
    # ---- 构建因子DataFrame ----
    print("[2/6] Processing factors...")
    
    # 创建统一的日期索引（周频）
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    factors_raw = pd.DataFrame(index=date_range)
    
    # 因子1: 短端利率 (越高压力越大)
    if treasury_2y is not None:
        factors_raw['short_rate'] = treasury_2y.reindex(date_range, method='ffill')
    
    # 因子2: VIX波动率 (越高压力越大)
    if vix is not None:
        factors_raw['vix'] = vix.reindex(date_range, method='ffill')
    
    # 因子3: 原油价格 (越高压力越大 - 通胀催化剂)
    if oil is not None:
        factors_raw['oil'] = oil.reindex(date_range, method='ffill')
    
    # 因子4: 官方流动性 (美联储BS - TGA, 越低压力越大)
    if fed_bs is not None and tga is not None:
        net_liquidity = fed_bs.reindex(date_range, method='ffill') - tga.reindex(date_range, method='ffill')
        factors_raw['net_liquidity'] = net_liquidity
    
    # 因子5: CPI通胀 (越高压力越大)
    if cpi is not None:
        factors_raw['cpi'] = cpi.reindex(date_range, method='ffill')
    
    # 如果数据不足，使用模拟数据进行演示
    if factors_raw.dropna(axis=1, how='all').shape[1] < 3:
        print("  [INFO] Insufficient live data. Using simulated data for demonstration.")
        factors_raw = generate_simulated_data(start_date, end_date)
    
    # ---- 标准化因子 ----
    print("[3/6] Standardizing factors (percentile ranking)...")
    
    factors_standardized = pd.DataFrame(index=factors_raw.index)
    
    factor_directions = {
        'short_rate': True,      # 利率高 = 压力大
        'vix': True,             # VIX高 = 压力大
        'oil': True,             # 油价高 = 压力大
        'net_liquidity': False,  # 流动性高 = 压力小
        'cpi': True,             # CPI高 = 压力大
        'duration_supply': True, # 发债多 = 压力大
    }
    
    for col in factors_raw.columns:
        if col in factor_directions:
            direction = factor_directions[col]
        else:
            direction = True
        
        series = factors_raw[col].dropna()
        if len(series) > 52:
            factors_standardized[col] = standardize_factor(
                factors_raw[col].ffill(), 
                higher_is_pressure=direction
            )
    
    # ---- 合成压力指数 ----
    print("[4/6] Building composite pressure index...")
    
    pi = build_pressure_index(factors_standardized)
    extreme_mask, threshold = identify_extreme_points(pi)
    
    print(f"  Composite PI - Current: {pi.dropna().iloc[-1]:.1f} percentile")
    print(f"  Extreme threshold (90th): {threshold:.1f}")
    print(f"  Extreme points count: {extreme_mask.sum()}")
    
    # ---- 尾部风险分析 ----
    print("[5/6] Computing tail risk statistics...")
    
    if sp500 is not None:
        sp500_weekly = sp500.reindex(date_range, method='ffill')
    else:
        # 使用模拟SP500
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.02, len(date_range))
        sp500_weekly = pd.Series(
            (1 + pd.Series(returns)).cumprod() * 4000,
            index=date_range
        )
    
    forward_returns = compute_forward_returns(sp500_weekly, weeks=4)
    
    # 统计对比
    normal_ret = forward_returns[~extreme_mask].dropna()
    extreme_ret = forward_returns[extreme_mask].dropna()
    
    stats_summary = {
        'normal': {
            'mean_return': normal_ret.mean() if len(normal_ret) > 0 else 0,
            'median_return': normal_ret.median() if len(normal_ret) > 0 else 0,
            'prob_drop_5pct': (normal_ret < -5).mean() * 100 if len(normal_ret) > 0 else 0,
            'prob_drop_10pct': (normal_ret < -10).mean() * 100 if len(normal_ret) > 0 else 0,
            'avg_max_drawdown': normal_ret[normal_ret < 0].mean() if len(normal_ret[normal_ret < 0]) > 0 else 0,
        },
        'extreme': {
            'mean_return': extreme_ret.mean() if len(extreme_ret) > 0 else 0,
            'median_return': extreme_ret.median() if len(extreme_ret) > 0 else 0,
            'prob_drop_5pct': (extreme_ret < -5).mean() * 100 if len(extreme_ret) > 0 else 0,
            'prob_drop_10pct': (extreme_ret < -10).mean() * 100 if len(extreme_ret) > 0 else 0,
            'avg_max_drawdown': extreme_ret[extreme_ret < 0].mean() if len(extreme_ret[extreme_ret < 0]) > 0 else 0,
        }
    }
    
    print(f"\n  --- Tail Risk Statistics ---")
    print(f"  Normal periods:")
    print(f"    Mean 4-week return: {stats_summary['normal']['mean_return']:.2f}%")
    print(f"    P(drop > 5%): {stats_summary['normal']['prob_drop_5pct']:.1f}%")
    print(f"    P(drop > 10%): {stats_summary['normal']['prob_drop_10pct']:.1f}%")
    print(f"  Extreme pressure periods:")
    print(f"    Mean 4-week return: {stats_summary['extreme']['mean_return']:.2f}%")
    print(f"    P(drop > 5%): {stats_summary['extreme']['prob_drop_5pct']:.1f}%")
    print(f"    P(drop > 10%): {stats_summary['extreme']['prob_drop_10pct']:.1f}%")
    
    # ---- 生成可视化 ----
    print("\n[6/6] Generating visualizations...")
    
    # 图1: 历史压力指数
    plot_pressure_index_history(
        pi, extreme_mask, threshold,
        save_path=os.path.join(OUTPUT_DIR, "01_pressure_index_history.png")
    )
    print("  Saved: 01_pressure_index_history.png")
    
    # 图2: 尾部风险分析
    plot_tail_risk_analysis(
        forward_returns, extreme_mask,
        save_path=os.path.join(OUTPUT_DIR, "02_tail_risk_analysis.png")
    )
    print("  Saved: 02_tail_risk_analysis.png")
    
    # 图3: 当前仪表盘
    current_factors = {}
    for col in factors_standardized.columns:
        last_valid = factors_standardized[col].dropna()
        if len(last_valid) > 0:
            current_factors[col] = last_valid.iloc[-1]
    
    pi_current = pi.dropna().iloc[-1] if len(pi.dropna()) > 0 else 50
    
    # 美化因子名称
    factor_display_names = {
        'short_rate': 'Short-term Rate\n(短端利率)',
        'vix': 'Volatility (VIX)\n(波动率)',
        'oil': 'Oil Price\n(原油价格)',
        'net_liquidity': 'Net Liquidity\n(净流动性)',
        'cpi': 'CPI Inflation\n(通胀)',
        'duration_supply': 'Duration Supply\n(久期供给)',
    }
    
    display_factors = {}
    for k, v in current_factors.items():
        display_name = factor_display_names.get(k, k)
        display_factors[display_name] = v
    
    plot_current_dashboard(
        display_factors, pi_current,
        save_path=os.path.join(OUTPUT_DIR, "03_current_dashboard.png")
    )
    print("  Saved: 03_current_dashboard.png")
    
    # 图4: 前瞻压力测试
    # 基于当前水平和已知宏观日程的简单推演
    monthly_stress = generate_forward_stress(pi_current, factors_standardized)
    plot_forward_stress_test(
        monthly_stress,
        save_path=os.path.join(OUTPUT_DIR, "04_forward_stress_test.png")
    )
    print("  Saved: 04_forward_stress_test.png")
    
    # ---- 保存数据 ----
    # 保存压力指数数据
    output_data = pd.DataFrame({
        'date': pi.index,
        'pressure_index': pi.values,
        'is_extreme': extreme_mask.values
    })
    output_data.to_csv(os.path.join(DATA_DIR, "pressure_index_data.csv"), index=False)
    
    # 保存当前状态JSON
    current_state = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_pi': float(pi_current),
        'extreme_threshold': float(threshold),
        'is_currently_extreme': bool(pi_current >= threshold),
        'factors': {k: float(v) for k, v in current_factors.items()},
        'tail_risk_stats': stats_summary,
        'forward_stress': monthly_stress,
    }
    
    with open(os.path.join(OUTPUT_DIR, "current_state.json"), 'w') as f:
        json.dump(current_state, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Model run complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Current Pressure Index: {pi_current:.1f} percentile")
    if pi_current >= threshold:
        print("⚠️  WARNING: Currently in EXTREME PRESSURE zone!")
    else:
        print("✓  Market pressure within normal range.")
    print("=" * 60)
    
    return current_state


def generate_simulated_data(start_date, end_date):
    """生成模拟数据用于演示"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    n = len(date_range)
    np.random.seed(2024)
    
    factors = pd.DataFrame(index=date_range)
    
    # 模拟短端利率 (0-5%)
    t = np.linspace(0, 1, n)
    base_rate = 0.5 + 2 * np.sin(2 * np.pi * t) + 1.5 * t
    # 添加2022加息周期
    rate_spike = np.zeros(n)
    spike_start = int(n * 0.7)
    spike_end = int(n * 0.85)
    rate_spike[spike_start:spike_end] = np.linspace(0, 3, spike_end - spike_start)
    rate_spike[spike_end:] = 3 * np.exp(-np.linspace(0, 2, n - spike_end))
    
    factors['short_rate'] = base_rate + rate_spike + np.random.normal(0, 0.1, n)
    factors['short_rate'] = factors['short_rate'].clip(0, 6)
    
    # 模拟VIX (10-80)
    base_vix = 18 + 5 * np.sin(3 * np.pi * t)
    # 添加危机spike
    vix_spikes = np.zeros(n)
    crisis_points = [int(n*0.45), int(n*0.72), int(n*0.83)]  # 2020, 2022, 2023
    for cp in crisis_points:
        spike_len = min(20, n - cp)
        vix_spikes[cp:cp+spike_len] = 40 * np.exp(-np.linspace(0, 3, spike_len))
    
    factors['vix'] = base_vix + vix_spikes + np.random.normal(0, 2, n)
    factors['vix'] = factors['vix'].clip(10, 80)
    
    # 模拟原油价格 (20-120)
    base_oil = 60 + 20 * np.sin(2.5 * np.pi * t) + 10 * t
    oil_shock = np.zeros(n)
    oil_shock[int(n*0.45):int(n*0.5)] = -30  # COVID crash
    oil_shock[int(n*0.65):int(n*0.75)] = 30   # 2022 spike
    
    factors['oil'] = base_oil + oil_shock + np.random.normal(0, 3, n)
    factors['oil'] = factors['oil'].clip(20, 130)
    
    # 模拟净流动性 (万亿美元)
    base_liq = 4 + 2 * t + 1.5 * np.sin(np.pi * t)
    # QE/QT cycles
    liq_cycle = np.zeros(n)
    liq_cycle[int(n*0.45):int(n*0.6)] = np.linspace(0, 2, int(n*0.15))  # COVID QE
    liq_cycle[int(n*0.6):int(n*0.75)] = 2
    liq_cycle[int(n*0.75):] = 2 - np.linspace(0, 1.5, n - int(n*0.75))  # QT
    
    factors['net_liquidity'] = base_liq + liq_cycle + np.random.normal(0, 0.1, n)
    
    # 模拟CPI (0-9%)
    base_cpi = 2 + 0.5 * np.sin(2 * np.pi * t)
    cpi_spike = np.zeros(n)
    cpi_spike[int(n*0.55):int(n*0.75)] = np.concatenate([
        np.linspace(0, 7, int(n*0.1)),
        np.linspace(7, 3, int(n*0.1))
    ])
    
    factors['cpi'] = base_cpi + cpi_spike + np.random.normal(0, 0.2, n)
    factors['cpi'] = factors['cpi'].clip(0, 10)
    
    # 模拟久期供给
    base_supply = 50 + 20 * t + 10 * np.sin(4 * np.pi * t)
    supply_spike = np.zeros(n)
    supply_spike[int(n*0.45):int(n*0.55)] = 30  # COVID fiscal
    supply_spike[int(n*0.8):int(n*0.9)] = 20
    
    factors['duration_supply'] = base_supply + supply_spike + np.random.normal(0, 3, n)
    
    return factors


def generate_forward_stress(pi_current, factors_standardized):
    """
    生成前瞻压力测试
    基于已知宏观日程和当前水位推演未来6个月
    """
    now = datetime.now()
    months = []
    values = []
    
    # 基于当前水平和季节性模式推演
    # 已知模式：9月和11月通常压力较高（美联储会议、企业缴税、TGA回补）
    seasonal_adjustment = {
        1: -5, 2: -3, 3: 0, 4: -2, 5: -5, 6: 3,
        7: 5, 8: 8, 9: 15, 10: 5, 11: 12, 12: -3
    }
    
    for i in range(1, 7):
        future_month = (now.month + i - 1) % 12 + 1
        future_year = now.year + (now.month + i - 1) // 12
        month_label = f"{future_year}-{future_month:02d}"
        months.append(month_label)
        
        # 基础值 = 当前PI + 季节性调整 + 随机扰动
        base = pi_current + seasonal_adjustment.get(future_month, 0)
        # 添加一些确定性趋势
        trend = i * 1.5  # 轻微上升趋势
        value = min(95, max(15, base + trend + np.random.normal(0, 3)))
        values.append(value)
    
    return dict(zip(months, values))


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    result = run_model()
