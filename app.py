import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Trade Journal Analysis", layout="wide")
st.title("📈 Trade Journal Analysis Dashboard")
st.markdown("Upload your trades CSV to begin. All analysis is performed in your browser.")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your trades CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")
    st.write(df.head())

    # --- Preprocessing ---
    df['EnteredAt'] = pd.to_datetime(df['EnteredAt'])
    df['ExitedAt'] = pd.to_datetime(df['ExitedAt'])
    df['Was_Win'] = df['PnL'] > 0
    df['Cumulative_PnL'] = df['PnL'].cumsum()
    def get_contract_type(name):
        name = str(name).upper()
        if 'MNQ' in name:
            return 'Micro (MNQ)'
        elif 'NQ' in name:
            return 'Mini (NQ)'
        else:
            return f'Other ({name})'
    df['ContractType'] = df['ContractName'].apply(get_contract_type)

    # --- Outlier Detection for PnL ---
    pnl_mean = df['PnL'].mean()
    pnl_std = df['PnL'].std()
    outliers = df[np.abs(df['PnL'] - pnl_mean) > 5 * pnl_std]
    if not outliers.empty:
        st.warning(f"Warning: {len(outliers)} P&L value(s) are extreme outliers (>|5σ| from mean). Please check your data for errors.")
        st.write(outliers[['EnteredAt', 'PnL', 'ContractName', 'Size']])

    # --- Dashboard Summary ---
    st.header("Dashboard Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"${df['PnL'].sum():,.2f}")
        st.metric("Trade Win %", f"{(df['Was_Win'].mean()*100):.2f}%")
        st.metric("Total Trades", len(df))
    with col2:
        avg_win = df[df['Was_Win']]['PnL'].mean()
        avg_loss = df[~df['Was_Win']]['PnL'].mean()
        st.metric("Avg Win / Avg Loss", f"{abs(avg_win/avg_loss):.2f}")
        st.metric("Avg Winning Trade", f"${avg_win:.2f}")
        st.metric("Avg Losing Trade", f"${avg_loss:.2f}")
    with col3:
        st.metric("Profit Factor", f"{df[df['PnL'] > 0]['PnL'].sum() / abs(df[df['PnL'] < 0]['PnL'].sum()):.2f}")
        st.metric("Total Lots Traded", int(df['Size'].sum()))
        st.metric("Best Trade", f"${df['PnL'].max():.2f}")

    # --- Equity Curve (Plotly with markers and x-axis buffer) ---
    st.header("Equity Curve")
    fig = go.Figure()
    # Line for equity curve
    fig.add_trace(go.Scatter(
        x=df['EnteredAt'],
        y=df['Cumulative_PnL'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='red', width=2)
    ))
    # Circles for each trade
    fig.add_trace(go.Scatter(
        x=df['EnteredAt'],
        y=df['Cumulative_PnL'],
        mode='markers',
        name='Trade',
        marker=dict(color='cyan', size=8, line=dict(color='black', width=1)),
        showlegend=True
    ))
    # Set x-axis range with a small buffer after the last trade
    start_time = df['EnteredAt'].min()
    end_time = df['EnteredAt'].max() + pd.Timedelta(minutes=10)
    fig.update_layout(
        template='plotly_dark',
        title='Equity Curve',
        xaxis_title='Time',
        yaxis_title='Cumulative P&L ($)',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(range=[start_time, end_time])
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Monte Carlo Simulation (matplotlib for now) ---
    st.header("Monte Carlo Simulation vs. Actual")
    n_trades = len(df)
    n_simulations = 100
    trade_outcomes = df['PnL'].values
    your_sim = []
    for _ in range(n_simulations):
        outcomes = np.random.choice(trade_outcomes, size=n_trades, replace=True)
        your_sim.append(np.cumsum(outcomes))
    your_sim = np.array(your_sim)
    plt.style.use('dark_background')
    fig2, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for curve in your_sim:
        ax.plot(curve, color=(1, 1, 1, 0.15), zorder=1)
    ax.plot(np.median(your_sim, axis=0), color='orange', linestyle='--', linewidth=2, label='Projected Median', zorder=2)
    ax.plot(np.cumsum(trade_outcomes), color='red', linewidth=2, label='Your Actual', zorder=3)
    ax.set_title('Monte Carlo Simulations vs. Actual', fontsize=16, color='white')
    ax.set_xlabel('Trade Number', color='white')
    ax.set_ylabel('Cumulative P&L ($)', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig2.patch.set_facecolor('black')
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Position Sizing Stats by Contract Type and Size ---
    st.header("Position Sizing Stats by Contract Type and Size")
    size_stats = df.groupby(['ContractType', 'Size']).agg(
        num_trades=('PnL', 'count'),
        avg_pnl=('PnL', 'mean'),
        win_rate=('Was_Win', 'mean'),
        total_pnl=('PnL', 'sum')
    ).reset_index()
    size_stats['win_rate'] = size_stats['win_rate'] * 100
    st.dataframe(size_stats)

    # --- Bell Curve of Profits (Plotly) ---
    st.header("Distribution of Trade Profits (Bell Curve)")
    fig3 = px.histogram(df, x='PnL', nbins=30, marginal='box', opacity=0.7, color_discrete_sequence=['deepskyblue'], template='plotly_dark')
    fig3.update_layout(title='Distribution of Trade Profits (Bell Curve)', xaxis_title='Trade P&L ($)', yaxis_title='Count')
    st.plotly_chart(fig3, use_container_width=True)

    # --- Position Sizing Consistency by Contract Type ---
    st.header("Position Sizing Consistency by Contract Type")
    consistency_stats = df.groupby('ContractType').agg(
        mean_size=('Size', 'mean'),
        std_size=('Size', 'std'),
        mode_size=('Size', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        min_size=('Size', 'min'),
        max_size=('Size', 'max'),
        num_trades=('Size', 'count')
    ).reset_index()
    st.dataframe(consistency_stats)

    # --- Distribution of Position Sizes by Contract Type ---
    st.header("Distribution of Position Sizes by Contract Type")
    plt.style.use('dark_background')
    fig4, ax = plt.subplots(figsize=(12,7), dpi=120)
    sns.histplot(data=df, x='Size', hue='ContractType', multiple='dodge',
                 bins=range(int(df['Size'].min()), int(df['Size'].max())+2),
                 palette='tab10', edgecolor='white', alpha=0.8, ax=ax)
    ax.set_title('Distribution of Position Sizes by Contract Type', fontsize=16, color='white')
    ax.set_xlabel('Position Size (Contracts)', color='white')
    ax.set_ylabel('Number of Trades', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(title='Contract Type', loc='best')
    fig4.patch.set_facecolor('black')
    plt.tight_layout()
    st.pyplot(fig4)

else:
    st.info("Please upload a CSV file to see your trade analysis dashboard.") 