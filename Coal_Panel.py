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
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = pd.read_excel("data/data.xlsx")
    st.session_state.raw_data.index = pd.to_datetime(st.session_state.raw_data['日期'])

raw_data = st.session_state.raw_data

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

# 计算策略最终净值
final_nav1 = df_res1['仓位净值'].iloc[-1]
final_nav2 = df_res2['仓位净值'].iloc[-1]
final_nav3 = df_res3['仓位净值'].iloc[-1]

# 计算先验仓位最终净值
prior_nav = df_res1['先验仓位净值'].iloc[-1]

# 计算超额净值
final_excess_nav1 = final_nav1 - prior_nav
final_excess_nav2 = final_nav2 - prior_nav
final_excess_nav3 = final_nav3 - prior_nav

# 计算最大回撤函数
def calculate_max_drawdown(df):
    # 提取有效数据（从第一次变化日期开始）
    df = df[first_change_date:]
    roll_max = df['仓位净值'].cummax()
    drawdown = (df['仓位净值'] - roll_max) / roll_max
    return drawdown.min()

# 找到先验仓位第一次变化的日期
# 先验仓位基于P(W)计算，P(W)在前observation_periods + holding_period + 1个周期内为NaN或0
# 找到P(W)第一次大于0的日期作为起始点
first_change_date = df_res1[df_res1['P(W)'] > 0].index[0]

# 计算三个策略的最大回撤
max_dd1 = calculate_max_drawdown(df_res1)
max_dd2 = calculate_max_drawdown(df_res2)
max_dd3 = calculate_max_drawdown(df_res3)

st.title("动力煤面板")
st.divider()

st.markdown("""
*  **策略1**    
    库存小于需求时买入
*  **策略2**    
    库存小于需求时买入；电厂月耗同比小于0和煤矿库存同比大于0卖出
*  **策略3**    
    库存小于需求且库存可用天数小于75%分位数时买入；电厂月耗同比小于0和煤矿库存同比大于0卖出
""")
st.divider()

# 数据显示与更新功能
st.markdown("## 数据概览")

# 显示最新数据日期
latest_date = raw_data['日期'].max()
st.info(f"**当前数据最新日期**: {latest_date.strftime('%Y-%m')}")

# 创建两列布局：左侧显示数据，右侧上传新数据
data_col1, data_col2 = st.columns([2, 1])

with data_col1:
    st.markdown("### 最新数据预览（最近10条）")
    # 显示最近10条数据
    display_df = raw_data.tail(10).copy()
    # 格式化日期列，只显示年月
    display_df['日期'] = pd.to_datetime(display_df['日期']).dt.strftime('%Y-%m')
    display_df = display_df.reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)

with data_col2:
    st.markdown("### 数据更新")
    uploaded_file = st.file_uploader(
        "上传新数据文件（Excel格式）",
        type=['xlsx', 'xls'],
        help="上传与原始数据格式相同的Excel文件来更新数据",
        key="file_uploader_key"
    )

    if uploaded_file is not None:
        try:
            # 读取上传的新数据
            new_data = pd.read_excel(uploaded_file)

            # 验证数据格式
            required_cols = ['日期', '秦皇岛5500K动力末煤平仓价', '全环节库存', '需求',
                           '库存可用天数', '电厂月耗同比', '煤矿库存同比']

            if all(col in new_data.columns for col in required_cols):
                st.success(f"✅ 成功读取新数据，共 {len(new_data)} 行")

                if st.button("🔄 使用新数据重新计算", type="primary"):
                    # 使用新数据替换原始数据
                    st.session_state.raw_data = new_data.copy()
                    st.session_state.raw_data.index = pd.to_datetime(st.session_state.raw_data['日期'])

                    st.success("✅ 数据已更新，回测已重新计算！")
                    st.rerun()
            else:
                missing_cols = [col for col in required_cols if col not in new_data.columns]
                st.error(f"❌ 数据格式错误！缺少以下列: {', '.join(missing_cols)}")

        except Exception as e:
            st.error(f"❌ 读取文件失败: {str(e)}")

st.divider()


left_co, cent_co, last_co = st.columns([1, 3, 1])
with cent_co:
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
        line=dict(color='gold', width=2)
    ))
    # 添加策略2的仓位净值
    fig.add_trace(go.Scatter(
        x=df_res2.index, 
        y=df_res2['仓位净值'],
        name='策略2',
        line=dict(color='red', width=2)
    ))
    # 添加策略3的仓位净值
    fig.add_trace(go.Scatter(
        x=df_res3.index, 
        y=df_res3['仓位净值'],
        name='策略3',
        line=dict(color='pink', width=2)
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
    
    # 创建指标数据框
    metrics_data = {
        '策略': ['策略1', '策略2', '策略3'],
        '净值收益': [final_nav1, final_nav2, final_nav3],
        '超额收益': [final_excess_nav1, final_excess_nav2, final_excess_nav3],
        '最大回撤': [max_dd1, max_dd2, max_dd3]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # 格式化显示
    metrics_df['净值收益'] = metrics_df['净值收益'].apply(lambda x: f"{x:.2%}")
    metrics_df['超额收益'] = metrics_df['超额收益'].apply(lambda x: f"{x:.2%}")
    metrics_df['最大回撤'] = metrics_df['最大回撤'].apply(lambda x: f"{x:.2%}")
    
    # 显示指标表格
    st.markdown("##### 策略绩效指标")
    st.dataframe(metrics_df.set_index('策略'), use_container_width=True)
    
    
    st.markdown("##### 最优策略:")
    st.markdown("""
        * 库存小于需求时买入
        * 电厂月耗同比小于0和煤矿库存同比大于0时卖出
    """)
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
        line=dict(color='black', width=2),
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
        name='卖出信号',
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
        line=dict(color='black', width=2),
        yaxis='y2'
    ))
    # 更新图表布局
    fig3.update_layout(
        title='买入卖出信号与煤价走势对比',
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
            ticktext=['卖出', '无', '买入'],
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
        line=dict(color='gold', width=2),
        yaxis='y1',
        hovertemplate='日期: %{x}<br>全环节库存: %{y}<extra></extra>'
    ))
    # 添加需求（左轴）
    fig4.add_trace(go.Scatter(
        x=df_res2.index, 
        y=df_res2['需求'],
        name='需求',
        mode='lines',
        line=dict(color='orange', width=2),
        yaxis='y1',
        hovertemplate='日期: %{x}<br>需求: %{y}<extra></extra>'
    ))
    # 添加库存-需求差值填充区域（左轴）
    fig4.add_trace(go.Scatter(
        x=df_res2.index, 
        y=inventory_demand_diff,
        name='库存-需求',
        mode='lines',
        line=dict(color='silver', width=2),
        fill='tozeroy',
        fillcolor='rgba(192, 192, 192, 0.2)',
        yaxis='y1',
        hovertemplate='日期: %{x}<br>库存-需求差值: %{y}<extra></extra>'
    ))

    # 添加煤价（右轴）
    fig4.add_trace(go.Scatter(
        x=df_res2.index, 
        y=df_res2['秦皇岛5500K动力末煤平仓价'],
        name='秦皇岛5500K动力末煤平仓价',
        mode='lines',
        line=dict(color='black', width=2),
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
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)',
        yaxis='y1',
        hovertemplate='日期: %{x}<br>电厂月耗同比: %{y:.2%}<extra></extra>'
    ))
    # 添加煤矿库存同比（左轴）
    fig5.add_trace(go.Scatter(
        x=df_res2.index, 
        y=df_res2['煤矿库存同比'],
        name='煤矿库存同比',
        mode='lines',
        line=dict(color='gold', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.2)',
        yaxis='y1',
        hovertemplate='日期: %{x}<br>煤矿库存同比: %{y:.2%}<extra></extra>'
    ))
    # 添加煤价（右轴）
    fig5.add_trace(go.Scatter(
        x=df_res2.index, 
        y=df_res2['秦皇岛5500K动力末煤平仓价'],
        name='秦皇岛5500K动力末煤平仓价',
        mode='lines',
        line=dict(color='black', width=2),
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








