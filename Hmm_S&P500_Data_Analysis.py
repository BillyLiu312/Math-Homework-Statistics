import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
import matplotlib.dates as mdates

# ==========================================
# 1. 数据获取与预处理
# ==========================================
def get_data(symbol='^GSPC', start='2000-01-01', end='2023-12-31'):
    print(f"正在下载 {symbol} 数据...")
    df = yf.download(symbol, start=start, end=end)
    
    # 计算对数收益率
    # log(Pt / Pt-1)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 去除NaN值
    df = df.dropna()
    
    # 提取用于训练的数据 (需转换为二维数组)
    X = df['Log_Ret'].values.reshape(-1, 1)
    return df, X

# ==========================================
# 2. 模型训练 (EM Algorithm)
# ==========================================
def train_hmm(X, n_components=3):
    print(f"正在训练 Gaussian HMM (状态数={n_components})...")
    
    # GaussianHMM 使用 Baum-Welch 算法进行训练
    # covariance_type='diag' 表示对角协方差矩阵（一维数据即方差）
    model = hmm.GaussianHMM(n_components=n_components, 
                            covariance_type="diag", 
                            n_iter=1000, 
                            random_state=42)
    model.fit(X)
    
    print("训练完成。")
    print("状态转移矩阵 A:\n", model.transmat_)
    print("均值 Means:\n", model.means_)
    print("方差 Covars:\n", model.covars_)
    
    return model

# ==========================================
# 3. 结果可视化 (Inference & Plotting)
# ==========================================
def plot_results(df, model, X):
    # 使用 Viterbi 算法预测隐状态序列 (MAP Estimation)
    hidden_states = model.predict(X)
    
    # 为了绘图清晰，我们将不同的状态映射到颜色
    # 我们根据方差大小排序，方差最小的为"稳定"(绿色)，最大的为"动荡"(红色)
    variances = np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    order = np.argsort(variances.flatten())
    
    # 颜色映射: 0(低波) -> 绿色, 1(中波) -> 黄色, 2(高波) -> 红色
    color_map = {order[0]: 'green', order[1]: 'gold', order[2]: 'red'}
    label_map = {order[0]: 'Low Volatility', order[1]: 'Medium Volatility', order[2]: 'High Volatility'}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制价格曲线
    dates = df.index
    prices = df['Close'].values
    
    # 这里的技巧是：为了给曲线分段着色，我们循环绘制
    # 实际应用中为了性能通常会用 Collection，但这里为了代码易读使用循环 scatter/plot
    
    # 创建一个空的背景，用不同颜色填充时间段
    for i, state in enumerate(hidden_states):
        # 这种逐点绘制效率较低，但逻辑最简单。
        # 优化方案是找到状态切换点，分块绘制。
        if i % 100 == 0: continue # 简单稀疏化以加快绘图（仅作背景参考）
        
        start_date = mdates.date2num(dates[i])
        end_date = mdates.date2num(dates[i]) + 1 # 宽度设为1天
        
        rect_color = color_map[state]
        
        # 在背景上画色带
        ax.axvspan(dates[i], dates[i] + pd.Timedelta(days=1), 
                   facecolor=rect_color, alpha=0.3, edgecolor=None)

    # 绘制收盘价主线
    ax.plot(dates, prices, color='black', linewidth=1, label='S&P 500 Price')
    
    # 添加图例（手动创建伪图例）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label=f'State {order[0]} (Low Vol)'),
        Patch(facecolor='gold', alpha=0.3, label=f'State {order[1]} (Med Vol)'),
        Patch(facecolor='red', alpha=0.3, label=f'State {order[2]} (High Vol)'),
        plt.Line2D([0], [0], color='black', lw=1, label='S&P 500 Price')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title('S&P 500 Market Regimes Detection using Gaussian HMM', fontsize=16)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    df, X = get_data()
    
    # 2. 训练模型
    model = train_hmm(X, n_components=3)
    
    # 3. 展示结果
    plot_results(df, model, X)