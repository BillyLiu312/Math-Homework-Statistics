import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import matplotlib.dates as mdates
import os

# ==========================================
# 1. 数据获取与预处理 (修复了数据类型问题)
# ==========================================
def get_data():
    file_path = 'data/SP500_log_returns.csv'
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请先确保 data 目录下有 SP500_log_returns.csv 文件。")
        exit()

    # 读取数据
    # header=0 表示第一行是表头。如果 yfinance 保存了多级表头，这里可能需要调整，
    # 但最稳妥的方法是读进来后强制转数字。
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # === 核心修复步骤 ===
    # 强制将 'Close' 和 'Log_Ret' 列转换为数字类型
    # errors='coerce' 会把无法转换的字符（比如多余的表头行）变成 NaN
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Log_Ret'] = pd.to_numeric(df['Log_Ret'], errors='coerce')
    
    # === 数据清洗 ===
    # 1. 替换无穷大值为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. 删除包含空值 (NaN) 的行
    # 这一步会把刚才转换失败的脏数据也一起删掉
    original_len = len(df)
    df.dropna(subset=['Log_Ret', 'Close'], inplace=True)
    
    if len(df) < original_len:
        print(f"数据清洗：删除了 {original_len - len(df)} 行无效数据（含空值或非数字行）。")

    # 提取训练数据
    X = df['Log_Ret'].values.reshape(-1, 1)
    
    return df, X

# ==========================================
# 2. 模型训练 (EM Algorithm)
# ==========================================
def train_hmm(X, n_components=3):
    print(f"正在训练 Gaussian HMM (状态数={n_components})...")
    
    if np.isnan(X).any():
        print("错误：训练数据 X 中仍然包含 NaN 值！")
        return None

    model = hmm.GaussianHMM(n_components=n_components, 
                            covariance_type="diag", 
                            n_iter=1000, 
                            random_state=42)
    try:
        model.fit(X)
    except ValueError as e:
        print(f"训练出错: {e}")
        exit()
    
    print("训练完成。")
    print("状态转移矩阵 A:\n", model.transmat_)
    print("均值 Means:\n", model.means_)
    print("方差 Covars:\n", model.covars_)
    
    return model

# ==========================================
# 3. 结果可视化
# ==========================================
def plot_results(df, model, X):
    if model is None: return

    hidden_states = model.predict(X)
    
    # 按照方差大小排序状态颜色
    variances = np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    order = np.argsort(variances.flatten())
    
    # 0:低波动(绿), 1:中波动(黄), 2:高波动(红)
    color_map = {order[0]: 'green', order[1]: 'gold', order[2]: 'red'}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    dates = df.index
    # 此时 prices 已经是纯浮点数了，不会再报错
    prices = df['Close'].values
    
    # 批量转换日期，解决 matplotlib 版本兼容性问题
    try:
        date_nums = mdates.date2num(dates.to_pydatetime())
    except AttributeError:
        date_nums = mdates.date2num(dates)

    # 绘制背景色带
    # 计算 Y 轴的高度范围
    y_min = prices.min()
    y_max = prices.max()
    y_height = y_max - y_min # 这里的减法之前报错，现在应该正常了

    for i in range(len(hidden_states)):
        # 简单抽样绘制背景以提升速度 (每3个点画一次)
        if i % 3 == 0: 
            start_date = date_nums[i]
            rect_color = color_map[hidden_states[i]]
            
            # 绘制矩形
            ax.add_patch(plt.Rectangle((start_date, y_min), 
                                       1, # 宽度1天
                                       y_height, # 高度
                                       facecolor=rect_color, 
                                       alpha=0.3, 
                                       edgecolor=None))

    # 绘制价格主线
    ax.plot_date(date_nums, prices, '-', color='black', linewidth=1, label='S&P 500 Price')
    
    # 格式化 X 轴
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Low Volatility'),
        Patch(facecolor='gold', alpha=0.3, label='Medium Volatility'),
        Patch(facecolor='red', alpha=0.3, label='High Volatility'),
        plt.Line2D([0], [0], color='black', lw=1, label='Price')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title('S&P 500 Market Regimes Detection (Gaussian HMM)', fontsize=16)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df, X = get_data()
    model = train_hmm(X, n_components=3)
    plot_results(df, model, X)