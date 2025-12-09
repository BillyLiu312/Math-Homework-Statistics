import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import matplotlib.dates as mdates
import os

def get_data():
    file_path = 'data/SP500_log_returns.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请先确保 data 目录下有 SP500_log_returns.csv 文件。")
        exit()

    # 读取数据
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # 数据处理
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Log_Ret'] = pd.to_numeric(df['Log_Ret'], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Log_Ret', 'Close'], inplace=True)

    # 提取训练数据
    X = df['Log_Ret'].values.reshape(-1, 1)
    
    return df, X

def train_hmm(X, n_components=3):
    print(f"正在训练 Gaussian HMM (状态数={n_components})...")
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

def plot_and_save_results(df, model, X):
    if model is None: return

    print("正在生成图像...")
    hidden_states = model.predict(X)
    
    # 按照方差大小排序状态颜色 (绿=稳, 红=乱)
    variances = np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    order = np.argsort(variances.flatten())
    
    color_map = {order[0]: 'green', order[1]: 'gold', order[2]: 'red'}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    dates = df.index
    prices = df['Close'].values
    
    try:
        date_nums = mdates.date2num(dates.to_pydatetime())
    except AttributeError:
        date_nums = mdates.date2num(dates)

    # 计算 Y 轴范围
    y_min = prices.min()
    y_max = prices.max()
    y_height = y_max - y_min

    for i in range(len(hidden_states)):
        if i % 3 == 0: 
            start_date = date_nums[i]
            rect_color = color_map[hidden_states[i]]
            
            ax.add_patch(plt.Rectangle((start_date, y_min), 
                                       1,
                                       y_height,
                                       facecolor=rect_color, 
                                       alpha=0.3, 
                                       edgecolor=None))

    ax.plot_date(date_nums, prices, '-', color='black', linewidth=1, label='S&P 500 Price')
    
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
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
    
    # 保存图片
    save_dir = 'fig'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹: {save_dir}")
        
    save_path = os.path.join(save_dir, 'hmm_sp500_result.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    df, X = get_data()
    model = train_hmm(X, n_components=3)
    plot_and_save_results(df, model, X)