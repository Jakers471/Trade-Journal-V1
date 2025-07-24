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
    df = df.sort_values('EnteredAt').reset_index(drop=True)
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

    # --- Outlier Detection for PnL and Big Jumps ---
    pnl_mean = df['PnL'].mean()
    pnl_std = df['PnL'].std()
    outliers = df[np.abs(df['PnL'] - pnl_mean) > 5 * pnl_std]
    if not outliers.empty:
        st.warning(f"Warning: {len(outliers)} P&L value(s) are extreme outliers (>|5σ| from mean). Please check your data for errors.")
        st.write(outliers[['EnteredAt', 'PnL', 'ContractName', 'Size']])
    # Warn for big single-trade jumps
    jump_threshold = 3 * max(abs(pnl_mean), abs(pnl_std))
    big_jumps = df[np.abs(df['PnL']) > jump_threshold]
    if not big_jumps.empty:
        st.warning(f"{len(big_jumps)} trade(s) have unusually large P&L (>|3x mean/std|). Please check for data entry errors.")
        st.write(big_jumps[['EnteredAt', 'PnL', 'Size', 'Cumulative_PnL']])

    # --- Quick PnL and Trade Diagnostics ---
    st.write(f"**Sum of PnL:** {df['PnL'].sum():,.2f}")
    st.write(f"**Number of trades:** {len(df)}")
    st.write('**Top 10 biggest trades:**')
    st.write(df[['EnteredAt', 'PnL', 'Size', 'Cumulative_PnL']].sort_values('PnL', ascending=False).head(10))
    st.write('**Top 10 biggest losses:**')
    st.write(df[['EnteredAt', 'PnL', 'Size', 'Cumulative_PnL']].sort_values('PnL').head(10))

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
        profit_factor = df[df['PnL'] > 0]['PnL'].sum() / abs(df[df['PnL'] < 0]['PnL'].sum())
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.metric("Total Lots Traded", int(df['Size'].sum()))
        st.metric("Best Trade", f"${df['PnL'].max():.2f}")

    # --- Performance Analysis ---
    st.subheader("Performance Analysis")
    perf_text = f"""
- **Win Rate:** {df['Was_Win'].mean()*100:.2f}%
- **Avg Win / Avg Loss:** {abs(avg_win/avg_loss):.2f}
- **Profit Factor:** {profit_factor:.2f}
- **Total P&L:** ${df['PnL'].sum():,.2f}
"""
    if profit_factor > 1.2 and df['Was_Win'].mean() > 0.4:
        perf_text += "\n:green[You have a positive edge! Your winners are larger than your losers and your win rate is solid. Keep focusing on consistency and risk management.]"
    elif profit_factor < 1:
        perf_text += "\n:red[Your system is currently negative expectancy. Consider improving your win rate, increasing your average win, or reducing your average loss.]"
    else:
        perf_text += "\n:orange[You're close to break-even. Small improvements in win rate or average win/loss could make you profitable.]"
    st.markdown(perf_text)

    # --- Equity Curve (Plotly with markers, by trade number) ---
    st.header("Equity Curve")
    st.markdown("""
**What this shows:** Your cumulative profit/loss after each trade. Each point is a trade; the red line is your running total. A smooth upward curve means consistent profits; sharp drops or flat sections indicate drawdowns or periods of no progress.
""")
    trade_numbers = np.arange(1, len(df)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df['Cumulative_PnL'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trade_numbers,
        y=df['Cumulative_PnL'],
        mode='markers',
        name='Trade',
        marker=dict(color='cyan', size=8, line=dict(color='black', width=1)),
        showlegend=True
    ))
    fig.update_layout(
        template='plotly_dark',
        title='Equity Curve (by Trade Number)',
        xaxis_title='Trade Number',
        yaxis_title='Cumulative P&L ($)',
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Monte Carlo Simulation (matplotlib for now) ---
    st.header("Monte Carlo Simulation vs. Actual")
    st.markdown("""
**What this shows:** The white lines are simulated alternate realities of your trading, using your actual win rate and average win/loss. The red line is your real result. If your red line is in the middle of the cloud, you're experiencing typical luck. If it's above, you're outperforming; below, underperforming. The spread shows the role of luck and risk in trading.
""")
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

    # --- Bell Curve of Profits (Plotly) ---
    st.header("Distribution of Trade Profits (Bell Curve)")
    st.markdown("""
**What this shows:** The bell curve (histogram) shows the distribution of your trade profits and losses. Most trades should cluster near the center. A long right tail means you have big winners; a long left tail means big losers. The box plot above shows the spread and outliers. Use this to spot if your results are skewed or have outliers.
""")
    fig3 = px.histogram(df, x='PnL', nbins=30, marginal='box', opacity=0.7, color_discrete_sequence=['deepskyblue'], template='plotly_dark')
    fig3.update_layout(title='Distribution of Trade Profits (Bell Curve)', xaxis_title='Trade P&L ($)', yaxis_title='Count', legend_title_text='Profit/Loss')
    fig3.update_traces(name='PnL', showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)

    # --- Position Sizing Consistency by Contract Type ---
    st.header("Position Sizing Consistency by Contract Type")
    st.markdown("""
**What this shows:** How consistent you are with your position sizes for each contract type (Micro, Mini, Other). Lower standard deviation means more consistent sizing. Mode is your most common size.
""")
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
    st.markdown("""
**What this shows:** The histogram shows how often you use each position size, broken down by contract type. Use this to spot if you are favoring certain sizes or being inconsistent.
""")
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

    # --- Full PnL Diagnostics ---
    st.write(f"**Sum of PnL (raw):** {df['PnL'].sum():,.2f}")
    st.write(f"**PnL dtype:** {df['PnL'].dtype}")
    st.write("**First 5 PnL values:**")
    st.write(df['PnL'].head())
    st.write("**Trades with PnL > 200 or < -200:**")
    st.write(df[(df['PnL'] > 200) | (df['PnL'] < -200)][['EnteredAt', 'PnL', 'Size', 'Cumulative_PnL']])
    # --- Check for and handle duplicate trades by Id ---
    dups = df[df.duplicated(subset=['Id'], keep=False)]
    if not dups.empty:
        st.warning(f"{len(dups)} duplicate trade(s) found by Id. This can cause PnL to be counted twice. Only the first occurrence will be kept in the analysis.")
        st.write(dups[['Id', 'EnteredAt', 'PnL', 'Size', 'Cumulative_PnL']])
        # Drop duplicates, keep first occurrence
        df = df.drop_duplicates(subset=['Id'], keep='first').reset_index(drop=True)
        # Recalculate Cumulative_PnL after removing duplicates
        df = df.sort_values('EnteredAt').reset_index(drop=True)
        df['Cumulative_PnL'] = df['PnL'].cumsum()

else:
    st.info("Please upload a CSV file to see your trade analysis dashboard.") 