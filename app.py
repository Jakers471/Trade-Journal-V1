import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # --- Equity Curve ---
    st.header("Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['EnteredAt'], df['Cumulative_PnL'], color='red', linewidth=2, label='Equity Curve')
    ax.scatter(df['EnteredAt'], df['Cumulative_PnL'], color='cyan', s=40, edgecolor='black', zorder=3, label='Trade Segment')
    ax.set_title('Detailed Equity Curve (Your Actual Trades)', fontsize=14, color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Cumulative P&L ($)', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=10)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('black')
    st.pyplot(fig)

    # --- Monte Carlo Simulation ---
    st.header("Monte Carlo Simulation vs. Actual")
    n_trades = len(df)
    n_simulations = 100
    trade_outcomes = df['PnL'].values
    your_sim = []
    for _ in range(n_simulations):
        outcomes = np.random.choice(trade_outcomes, size=n_trades, replace=True)
        your_sim.append(np.cumsum(outcomes))
    your_sim = np.array(your_sim)
    fig, ax = plt.subplots(figsize=(12, 6))
    for curve in your_sim:
        ax.plot(curve, color=(1, 1, 1, 0.15), zorder=1)
    ax.plot(np.median(your_sim, axis=0), color='orange', linestyle='--', linewidth=2, label='Projected Median', zorder=2)
    ax.plot(np.cumsum(trade_outcomes), color='red', linewidth=2, label='Your Actual', zorder=3)
    ax.set_title('Monte Carlo Simulations vs. Actual', fontsize=14, color='white')
    ax.set_xlabel('Trade Number', color='white')
    ax.set_ylabel('Cumulative P&L ($)', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=10)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('black')
    st.pyplot(fig)

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

    # --- Bell Curve of Profits ---
    st.header("Distribution of Trade Profits (Bell Curve)")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(df['PnL'], bins=30, kde=True, color='deepskyblue', edgecolor='white', stat='density', alpha=0.7, ax=ax)
    ax.set_title('Distribution of Trade Profits (Bell Curve)', fontsize=14, color='white')
    ax.set_xlabel('Trade P&L ($)', color='white')
    ax.set_ylabel('Density', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('black')
    st.pyplot(fig)

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
    fig, ax = plt.subplots(figsize=(12,7))
    sns.histplot(data=df, x='Size', hue='ContractType', multiple='dodge',
                 bins=range(int(df['Size'].min()), int(df['Size'].max())+2),
                 palette='tab10', edgecolor='white', alpha=0.8, ax=ax)
    ax.set_title('Distribution of Position Sizes by Contract Type', fontsize=14, color='white')
    ax.set_xlabel('Position Size (Contracts)', color='white')
    ax.set_ylabel('Number of Trades', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(title='Contract Type', loc='best')
    fig.patch.set_facecolor('black')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to see your trade analysis dashboard.") 