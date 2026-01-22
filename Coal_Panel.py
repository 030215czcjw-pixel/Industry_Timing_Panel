import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def Backtest(df, buy_condition, sell_condition):
    # Basic Settings
    profit_setted = 0
    observation_periods = 30
    holding_period = 6
    position_strategy = "先快后慢加仓"
    empty_position_mode = "硬清仓"
    
    # 深拷贝输入数据，避免修改原始数据
    df = df.copy()
    
    feature_cols = [
        '全环节库存',
        '需求',
        '库存可用天数',
        '电厂月耗同比',
        '煤矿库存同比',
    ]
    for col in feature_cols:
        df[col] = df[col].shift(1)
    
    #df['股价收益率']
    #df['基准收益率'] We dont have bench here
    df['超额收益率'] = df['秦皇岛5500K动力末煤平仓价'].pct_change()
    df['超额净值'] = (1 + df['超额收益率'].fillna(0)).cumprod()
    df['持有期超额收益率'] = df['超额净值'].shift(-holding_period) / df['超额净值'] - 1
    
    df['胜率触发'] = (df['持有期超额收益率'] > profit_setted).astype(int)
    df['胜率不触发'] = 1 - df['胜率触发']
    df['P(W)'] = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    
    # Buy Strategy 
    df['信号触发'] = np.where(
        eval(buy_condition), 
        1, 
        0
    ).astype(int)
    
    # Sell Strategy
    df['空仓信号'] = np.where(
        eval(sell_condition), 
        1, 
        0
    ).astype(int)
    
    # 计算后验概率 P(W|C)
    df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
    df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
    rolling_w_c = df['W_and_C'].rolling(observation_periods).sum().shift(holding_period + 1)
    rolling_w = df['胜率触发'].rolling(observation_periods).sum().shift(holding_period + 1)
    rolling_notw_c = df['notW_and_C'].rolling(observation_periods).sum().shift(holding_period + 1)
    rolling_notw = df['胜率不触发'].rolling(observation_periods).sum().shift(holding_period + 1)
    p_c_w = rolling_w_c / rolling_w.replace(0, np.nan)
    p_c_notw = rolling_notw_c / rolling_notw.replace(0, np.nan)
    evidence = p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)'])
    df['P(W|C)'] = (p_c_w * df['P(W)']) / evidence.replace(0, np.nan)
    
    prob_condition = (df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1) * 0.9)
    improve_condition = df['P(W|C)'] > df['P(W)']
    
    df['买入信号'] = np.where(
        improve_condition & (df['信号触发'] == 1) & prob_condition, 
        1, 0
    )
    
    if position_strategy == "原始策略逐步加仓":
        # 原始策略逐步加仓：根据概率变化和历史表现动态调整
        df['仓位'] = np.where(
            df['买入信号'] == 1,
            df['信号触发'].shift(1).rolling(holding_period).sum() / holding_period,
            0
        )
    elif position_strategy == "先快后慢加仓":
        # 先快后慢加仓：改进版，更及时响应信号
        # 创建一个仓位累积计数器
        df['仓位'] = np.where(
            df['买入信号'] == 1,
            0.3 + 0.7 * np.sqrt(df['信号触发'].shift(1).rolling(holding_period).sum() / holding_period),
            0
        )
    elif position_strategy == "正金字塔建仓":
        # 正金字塔建仓：底部仓位最重，越涨买得越少
        # 核心思想：在低位时重仓，随着价格上涨逐步减仓，降低风险
        # 计算持有期内的超额净值涨幅（相对于持有期前的最低点）
        df['持有期内最低净值'] = df['超额净值'].shift(1).rolling(holding_period).min()
        df['相对底部涨幅'] = (df['超额净值'].shift(1) - df['持有期内最低净值']) / df['持有期内最低净值'].replace(0, np.nan)
        # 初始化仓位为0
        df['仓位'] = 0.0
        # 只在买入信号触发时计算仓位
        buy_signal_mask = df['买入信号'] == 1
        relative_rise = df.loc[buy_signal_mask, '相对底部涨幅'].fillna(0)
        # 分段仓位分配：
        df.loc[buy_signal_mask, '仓位'] = np.select(
            [
                relative_rise < 0.05,
                relative_rise < 0.10,
                relative_rise < 0.15,
                relative_rise >= 0.15
            ],
            [0.8, 0.6, 0.4, 0.2],
            default=0.8  # 默认使用最大仓位
        )
    elif position_strategy == "时间加权加仓":
        # 时间加权加仓：越近的日期产生的信号权重越大
        # 核心思想：最近的信号更重要，使用指数加权来计算仓位
        # 初始化仓位为0
        df['仓位'] = 0.0
        # 只在买入信号触发时计算仓位
        buy_signal_mask = df['买入信号'] == 1
        # 使用指数加权移动平均(EWM)计算信号的加权和
        # span参数控制衰减速度，span越小，越重视近期信号
        span = max(holding_period // 2, 3)  # 至少为3，最大为持有期的一半
        # 计算信号的指数加权移动平均
        df['信号加权'] = df['信号触发'].shift(1).ewm(span=span, adjust=False).mean()
        # 在买入信号触发时，根据加权信号计算仓位
        # 加权信号范围是0-1，可以直接用作仓位比例
        df.loc[buy_signal_mask, '仓位'] = df.loc[buy_signal_mask, '信号加权']
        # 设置最小仓位阈值，避免仓位过小
        df.loc[buy_signal_mask & (df['仓位'] < 0.2), '仓位'] = 0.2
    # 确保仓位在0-1之间
    df['仓位'] = df['仓位'].clip(0, 1)

    # 应用空仓信号：根据不同的空仓模式处理仓位
    if empty_position_mode == "硬清仓":
        # 模式1：硬清仓 - 触发即归零
        df['仓位'] = np.where(df['空仓信号'] == 1, 0, df['仓位'])
    elif empty_position_mode == "半仓止损":
        # 模式2：半仓止损 - 触发时减至原仓位的50%
        df['仓位'] = np.where(df['空仓信号'] == 1, df['仓位'] * 0.5, df['仓位'])
    elif empty_position_mode == "三分之一仓":
        # 模式3：三分之一仓 - 触发时减至原仓位的33%
        df['仓位'] = np.where(df['空仓信号'] == 1, df['仓位'] * 0.33, df['仓位'])
    elif empty_position_mode == "渐进式减仓":
        # 模式4：渐进式减仓 - 连续触发时逐步减仓
        # 创建一个累计触发计数器
        df['空仓累计'] = (df['空仓信号'] == 1).astype(int)
        # 使用shift和cumsum创建连续触发计数
        # 当空仓信号为0时重置计数
        df['空仓连续触发'] = 0
        current_count = 0
        for idx in df.index:
            if df.loc[idx, '空仓信号'] == 1:
                current_count += 1
            else:
                current_count = 0
            df.loc[idx, '空仓连续触发'] = current_count
        # 根据连续触发次数递减仓位
        # 第1次：减至80%，第2次：减至60%，第3次：减至40%，第4次：减至20%，第5次及以上：清仓
        df['减仓系数'] = np.select(
            [
                df['空仓连续触发'] == 0,
                df['空仓连续触发'] == 1,
                df['空仓连续触发'] == 2,
                df['空仓连续触发'] == 3,
                df['空仓连续触发'] == 4,
                df['空仓连续触发'] == 5
            ],
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            default=1.0
        )
        df['仓位'] = df['仓位'] * df['减仓系数']     
    
    df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    
    return df

# 加载数据
raw_data = pd.read_excel("data/data.xlsx")
raw_data.index = pd.to_datetime(raw_data['日期'])

# 调用回测函数
df_res1 = Backtest(raw_data, 
                  buy_condition = "(df['全环节库存'] - df['需求']) < 0", 
                  sell_condition = "0"
)

df_res2 = Backtest(raw_data, 
                  buy_condition = "(df['全环节库存'] - df['需求']) < 0", 
                  sell_condition = "(df['电厂月耗同比'] < 0) & (df['煤矿库存同比'] > 0)"
)

df_res3 = Backtest(raw_data, 
                  buy_condition = "((df['全环节库存'] - df['需求']) < 0) & ((df['库存可用天数'] - df['库存可用天数'].expanding().quantile(0.75)) < 0)", 
                  sell_condition = "(df['电厂月耗同比'] < 0) & (df['煤矿库存同比'] > 0)"
)

st.title("动力煤面板")
st.divider()

# 找到先验仓位第一次变化的日期
# 先验仓位基于P(W)计算，P(W)在前observation_periods + holding_period + 1个周期内为NaN或0
# 找到P(W)第一次大于0的日期作为起始点
first_change_date = df_res1[df_res1['P(W)'] > 0].index[0]

st.text("策略1：库存小于需求时买入")
st.text("策略2：库存小于需求时买入；电厂月耗同比下降和煤矿库存同比上升时卖出")
st.text("策略3：库存小于需求且库存可用天数小于75%分位数时买入；电厂月耗同比下降和煤矿库存同比上升时卖出")
st.divider()

# 创建策略对比图表
fig = go.Figure()
# 添加先验仓位净值（所有策略共享）
fig.add_trace(go.Scatter(
    x=df_res1.index, 
    y=df_res1['先验仓位净值'],
    name='先验净值',
    line=dict(color='gray', width=2)
))
# 添加策略1的仓位净值
fig.add_trace(go.Scatter(
    x=df_res1.index, 
    y=df_res1['仓位净值'],
    name='策略1',
    line=dict(color='blue', width=2)
))
# 添加策略2的仓位净值
fig.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['仓位净值'],
    name='策略2',
    line=dict(color='orange', width=2)
))
# 添加策略3的仓位净值
fig.add_trace(go.Scatter(
    x=df_res3.index, 
    y=df_res3['仓位净值'],
    name='策略3',
    line=dict(color='red', width=2)
))

# 更新图表布局
fig.update_layout(
    title='不同策略净值对比',
    xaxis_title='日期',
    yaxis_title='净值',
    height=500,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    hovermode='x unified'
)

# 配置x轴，添加range selector和range slider
fig.update_xaxes(
    rangeslider_visible=True,
    tickformat='%Y-%m',
    tickangle=0,
    range=[first_change_date, df_res1.index[-1]]  # 设置x轴范围从第一次变化开始
)

# 显示策略对比图表
st.plotly_chart(fig, width='stretch')
st.text("最优策略：策略2：库存小于需求时买入；电厂月耗同比下降和煤矿库存同比上升时卖出")
# 添加分隔线
st.divider()

# 创建策略2仓位与煤价对比图表
fig2 = go.Figure()
# 添加策略2的仓位（左轴）
fig2.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['仓位'],
    name='仓位',
    line=dict(color='orange', width=2),
    yaxis='y1'
))

# 添加仓位填充区域
fig2.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['仓位'],
    name='仓位填充',
    fill='tozeroy',
    line=dict(width=0),
    fillcolor='rgba(255, 165, 0, 0.15)',
    yaxis='y1',
    showlegend=False,
    hoverinfo='skip'  # 鼠标悬停时不显示此填充区域的信息
))

# 添加煤价（右轴）
fig2.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['秦皇岛5500K动力末煤平仓价'],
    name='秦皇岛5500K动力末煤平仓价',
    line=dict(color='red', width=2),
    yaxis='y2'
))

# 更新图表布局
fig2.update_layout(
    title='最优策略仓位与煤价走势对比',
    xaxis_title='日期',
    height=500,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    hovermode='x unified',
    yaxis=dict(
        title='仓位',
        range=[0, 1.1],
        side='left'
    ),
    yaxis2=dict(
        title='煤价（元/吨）',
        overlaying='y',
        side='right'
    )
)

# 配置x轴，添加range selector和range slider，与第一张图保持一致
fig2.update_xaxes(
    rangeslider_visible=True,
    tickformat='%Y-%m',
    tickangle=0,
    range=[first_change_date, df_res2.index[-1]]  # 与第一张图保持相同的x轴范围
)

# 显示仓位与煤价对比图表
st.plotly_chart(fig2, width='stretch')

# 添加分隔线
st.divider()

# 创建第三张图表：买入信号、空仓信号与煤价走势
fig3 = go.Figure()
# 为策略2创建综合信号列：买入为1，空仓为-1，无信号为0
df_res2['综合信号'] = 0
df_res2.loc[df_res2['买入信号'] == 1, '综合信号'] = 1
df_res2.loc[df_res2['空仓信号'] == 1, '综合信号'] = -1
# 添加灰色水平线y=0
fig3.add_shape(
    type='line',
    x0=df_res2.index[0],
    y0=0,
    x1=df_res2.index[-1],
    y1=0,
    line=dict(
        color='gray',
        width=2,
        dash='solid'
    ),
    name='零线'
)
# 添加买入信号填充区域
fig3.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['买入信号'],
    name='买入信号',
    mode='lines',
    line=dict(color='red', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.2)',
    yaxis='y1',
    hovertemplate='日期: %{x}<br>信号: 买入<extra></extra>'
))
# 添加空仓信号填充区域
fig3.add_trace(go.Scatter(
    x=df_res2.index, 
    y=-df_res2['空仓信号'],
    name='空仓信号',
    mode='lines',
    line=dict(color='green', width=2),
    fill='tozeroy',
    fillcolor='rgba(0, 255, 0, 0.2)',
    yaxis='y1',
    hovertemplate='日期: %{x}<br>信号: 空仓<extra></extra>'
))
# 添加煤价（右轴）
fig3.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['秦皇岛5500K动力末煤平仓价'],
    name='秦皇岛5500K动力末煤平仓价',
    line=dict(color='blue', width=2),
    yaxis='y2'
))
# 更新图表布局
fig3.update_layout(
    title='开空仓信号与煤价走势对比',
    xaxis_title='日期',
    height=500,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    hovermode='x unified',
    yaxis=dict(
        title='信号',
        range=[-1.2, 1.2],
        tickvals=[-1, 0, 1],
        ticktext=['空仓', '无', '买入'],
        side='left'
    ),
    yaxis2=dict(
        title='煤价（元/吨）',
        overlaying='y',
        side='right'
    )
)
# 配置x轴，添加range selector和range slider，与前两张图保持一致
fig3.update_xaxes(
    rangeslider_visible=True,
    tickformat='%Y-%m',
    tickangle=0,
    range=[first_change_date, df_res2.index[-1]]  # 与前两张图保持相同的x轴范围
)
# 显示信号与煤价对比图表
st.plotly_chart(fig3, width='stretch')

# 添加分隔线
st.divider()
# 创建第四张图表：全环节库存、需求与煤价
fig4 = go.Figure()
# 计算库存-需求差值
# 使用原始数据，因为在Backtest函数中已经对这些列进行了shift(1)处理
inventory_demand_diff = df_res2['全环节库存'] - df_res2['需求']
# 添加全环节库存（左轴）
fig4.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['全环节库存'],
    name='全环节库存',
    mode='lines',
    line=dict(color='blue', width=2),
    yaxis='y1',
    hovertemplate='日期: %{x}<br>全环节库存: %{y}<extra></extra>'
))
# 添加需求（左轴）
fig4.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['需求'],
    name='需求',
    mode='lines',
    line=dict(color='red', width=2),
    yaxis='y1',
    hovertemplate='日期: %{x}<br>需求: %{y}<extra></extra>'
))
# 添加库存-需求差值填充区域（左轴）
fig4.add_trace(go.Scatter(
    x=df_res2.index, 
    y=inventory_demand_diff,
    name='库存-需求',
    mode='lines',
    line=dict(color='purple', width=2),
    fill='tozeroy',
    fillcolor='rgba(128, 0, 128, 0.2)',
    yaxis='y1',
    hovertemplate='日期: %{x}<br>库存-需求差值: %{y}<extra></extra>'
))

# 添加煤价（右轴）
fig4.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['秦皇岛5500K动力末煤平仓价'],
    name='秦皇岛5500K动力末煤平仓价',
    mode='lines',
    line=dict(color='orange', width=2),
    yaxis='y2',
    hovertemplate='日期: %{x}<br>煤价: %{y}元/吨<extra></extra>'
))

# 更新图表布局
fig4.update_layout(
    title='全环节库存、需求与煤价走势',
    xaxis_title='日期',
    height=500,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    hovermode='x unified',
    yaxis=dict(
        title='库存/需求',
        side='left'
    ),
    yaxis2=dict(
        title='煤价（元/吨）',
        overlaying='y',
        side='right'
    )
)

# 配置x轴，添加range selector和range slider，与前几张图保持一致
fig4.update_xaxes(
    rangeslider_visible=True,
    tickformat='%Y-%m',
    tickangle=0,
    range=[first_change_date, df_res2.index[-1]]  # 与前几张图保持相同的x轴范围
)

# 显示库存、需求与煤价对比图表
st.plotly_chart(fig4, width='stretch')

# 添加分隔线
st.divider()
# 创建第五张图表：电厂月耗同比、煤矿库存同比与煤价
fig5 = go.Figure()
# 添加电厂月耗同比（左轴）
fig5.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['电厂月耗同比'],
    name='电厂月耗同比',
    mode='lines',
    line=dict(color='blue', width=2),
    yaxis='y1',
    hovertemplate='日期: %{x}<br>电厂月耗同比: %{y:.2%}<extra></extra>'
))
# 添加煤矿库存同比（左轴）
fig5.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['煤矿库存同比'],
    name='煤矿库存同比',
    mode='lines',
    line=dict(color='red', width=2),
    yaxis='y1',
    hovertemplate='日期: %{x}<br>煤矿库存同比: %{y:.2%}<extra></extra>'
))
# 添加煤价（右轴）
fig5.add_trace(go.Scatter(
    x=df_res2.index, 
    y=df_res2['秦皇岛5500K动力末煤平仓价'],
    name='秦皇岛5500K动力末煤平仓价',
    mode='lines',
    line=dict(color='green', width=2),
    yaxis='y2',
    hovertemplate='日期: %{x}<br>煤价: %{y}元/吨<extra></extra>'
))
# 更新图表布局
fig5.update_layout(
    title='电厂月耗同比、煤矿库存同比与煤价走势',
    xaxis_title='日期',
    height=500,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    hovermode='x unified',
    yaxis=dict(
        title='同比增长率（%）',
        side='left',
        tickformat='.2%'  # 显示为百分比格式
    ),
    yaxis2=dict(
        title='煤价（元/吨）',
        overlaying='y',
        side='right'
    )
)
# 配置x轴，添加range selector和range slider，与前几张图保持一致
fig5.update_xaxes(
    rangeslider_visible=True,
    tickformat='%Y-%m',
    tickangle=0,
    range=[first_change_date, df_res2.index[-1]]  # 与前几张图保持相同的x轴范围
)
# 显示电厂月耗同比、煤矿库存同比与煤价对比图表
st.plotly_chart(fig5, width='stretch')








