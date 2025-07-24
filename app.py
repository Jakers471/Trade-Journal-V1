import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
USER_CSV = r'C:/Users/jakers/Downloads/trades_export.csv'  # Update path if needed
N_TRADES = 100  # Number of trades to simulate for each system
N_SIMULATIONS = 100  # Number of Monte Carlo runs per system
OUTPUT_DIR = 'trade_analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD USER DATA ===
df = pd.read_csv(USER_CSV)
df['Was_Win'] = df['PnL'] > 0
df['Cumulative_PnL'] = df['PnL'].cumsum()
your_stats = {
    'win_rate': df['Was_Win'].mean(),
    'avg_win': df[df['Was_Win']]['PnL'].mean(),
    'avg_loss': abs(df[~df['Was_Win']]['PnL'].mean()),
    'n_trades': len(df),
    'trade_outcomes': df['PnL'].values
}

# === EXAMPLE SYSTEMS ===
systems = {
    'Poor':    {'win_rate': 0.3, 'avg_win': 30,  'avg_loss': 60},
    'Average': {'win_rate': 0.5, 'avg_win': 50,  'avg_loss': 50},
    'Good':    {'win_rate': 0.6, 'avg_win': 80,  'avg_loss': 40}
}

# === MONTE CARLO FUNCTION ===
def monte_carlo_sim(win_rate, avg_win, avg_loss, n_trades, n_simulations=100):
    sim_curves = []
    for _ in range(n_simulations):
        outcomes = np.random.choice(
            [avg_win, -avg_loss],
            size=n_trades,
            p=[win_rate, 1-win_rate]
        )
        sim_curves.append(np.cumsum(outcomes))
    return np.array(sim_curves)

# === RUN SIMULATIONS ===
sim_results = {}
for name, stats in systems.items():
    sim_results[name] = monte_carlo_sim(
        stats['win_rate'], stats['avg_win'], stats['avg_loss'], N_TRADES, N_SIMULATIONS
    )

# User's own Monte Carlo (using real trade distribution)
your_sim = []
for _ in range(N_SIMULATIONS):
    outcomes = np.random.choice(your_stats['trade_outcomes'], size=N_TRADES, replace=True)
    your_sim.append(np.cumsum(outcomes))
your_sim = np.array(your_sim)

# Prepare actual curve for comparison
actual_curve = np.cumsum(np.pad(
    your_stats['trade_outcomes'],
    (0, max(0, N_TRADES-len(your_stats['trade_outcomes']))),
    'constant'
)[:N_TRADES])

# === PLOT ALL SYSTEMS SIDE BY SIDE (DARK THEME) ===
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
system_names = ['Poor', 'Average', 'Good']
colors = {'Poor': '#FF5555', 'Average': '#AAAAAA', 'Good': '#55FF55'}  # Medians
sim_line_color = (1, 1, 1, 0.08)  # RGBA for low-saturation white

for idx, name in enumerate(system_names):
    sims = sim_results[name]
    ax = axes[idx]
    # Set black background for each subplot
    ax.set_facecolor('black')
    # Monte Carlo cloud
    for curve in sims:
        ax.plot(curve, color=sim_line_color)
    # Median
    ax.plot(np.median(sims, axis=0), color=colors[name], linewidth=2, label=f"{name} Median")
    # Your actual
    ax.plot(actual_curve, color='red', linewidth=2, label='Your Actual')
    # Your projected median
    if idx == 2:  # Only show once to avoid legend clutter
        ax.plot(np.median(your_sim, axis=0), color='orange', linestyle='--', linewidth=2, label='Your Projected Median')
    else:
        ax.plot(np.median(your_sim, axis=0), color='orange', linestyle='--', linewidth=2)
    ax.set_title(f"{name} System", fontsize=15, color='white')
    ax.set_xlabel('Trade Number', color='white')
    if idx == 0:
        ax.set_ylabel('Cumulative P&L ($)', color='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.legend(facecolor='black', edgecolor='white', fontsize=10)
    # Set tick params for white color
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

fig.patch.set_facecolor('black')
plt.suptitle('Monte Carlo Simulations: Poor, Average, Good Systems vs. Your Actual & Projected', fontsize=18, color='white')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'monte_carlo_side_by_side.png'), facecolor=fig.get_facecolor())
plt.show()

# === PLOT YOUR PERSONAL SYSTEM ONLY (DARK THEME) ===
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor('black')

# Monte Carlo cloud (your system)
for curve in your_sim:
    ax.plot(curve, color=(1, 1, 1, 0.15), zorder=1)
# Projected median
ax.plot(np.median(your_sim, axis=0), color='orange', linestyle='--', linewidth=2, label='Your Projected Median', zorder=2)
# Your actual (no padding, only real trades)
ax.plot(np.arange(len(your_stats['trade_outcomes'])), np.cumsum(your_stats['trade_outcomes']),
        color='red', linewidth=2, label='Your Actual', zorder=3)

ax.set_title('Your System: Monte Carlo Simulations vs. Actual', fontsize=16, color='white')
ax.set_xlabel('Trade Number', color='white')
ax.set_ylabel('Cumulative P&L ($)', color='white')
ax.set_xlim(0, N_TRADES-1)
ax.grid(True, linestyle='--', alpha=0.3, color='white')
ax.legend(facecolor='black', edgecolor='white', fontsize=10)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# --- Add summary box ---
final_sim_pnls = your_sim[:, -1]
percentiles = np.percentile(final_sim_pnls, [5, 25, 50, 75, 95])
actual_final = np.cumsum(your_stats['trade_outcomes'])[-1]
if percentiles[2] < 0:
    edge_text = "Negative edge\nMost outcomes lose"
elif percentiles[2] > 0 and percentiles[0] < 0:
    edge_text = "Mixed edge\nLuck matters"
else:
    edge_text = "Positive edge\nMost outcomes win"
summary = (
    f"Your Final: ${actual_final:,.0f}\n"
    f"Median Sim: ${percentiles[2]:,.0f}\n"
    f"5-95%: ${percentiles[0]:,.0f} to ${percentiles[4]:,.0f}\n"
    f"{edge_text}"
)
ax.text(
    0.02, 0.98, summary,
    transform=ax.transAxes,
    fontsize=12,
    color='white',
    verticalalignment='top',
    bbox=dict(facecolor='black', alpha=0.7, edgecolor='white')
)

fig.patch.set_facecolor('black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monte_carlo_your_system.png'), facecolor=fig.get_facecolor())
plt.show()

# === EXPLANATION OF YOUR SYSTEM CHART ===
print("="*60)
print("YOUR SYSTEM: MONTE CARLO CHART EXPLANATION")
print("="*60)
print("1. The faint white lines are 100 possible alternate futures for your trading,")
print("   generated by randomly reordering your real trade results (with replacement).")
print("2. The bold red line is your actual cumulative P&L, showing your real journey so far.")
print("3. The orange dashed line is the median of all simulated outcomes—your 'expected' path if you keep trading the same way.")
print("4. The summary box on the chart shows:")
print("   - Your actual final P&L (red line endpoint)")
print("   - The median simulated final P&L (orange dashed endpoint)")
print("   - The 5th–95th percentile range of simulated outcomes (how wide luck can swing)")
print("   - A quick interpretation of your system's edge (positive, negative, or mixed)")
print()
print("How to interpret your chart:")
print("- If your red line is above most white lines, you are outperforming your stats (good luck or skill).")
print("- If your red line is below most white lines, you are underperforming (bad luck or possible issues).")
print("- If the orange dashed line trends up, your system has a positive edge; if flat or down, your edge is weak or negative.")
print("- The spread of white lines shows how much luck and variance can affect your results, even with the same system.")
print()
print("What to do next:")
print("- If your edge is positive, keep focusing on consistency and risk management.")
print("- If your edge is negative or mixed, consider improving your win rate, increasing your average win, or reducing your average loss.")
print("- Use this chart to set realistic expectations and avoid overreacting to short-term swings.")
print("="*60)

# Print summary for each system
for name in system_names:
    sims = sim_results[name]
    final_sim_pnls = sims[:, -1]
    percentiles = np.percentile(final_sim_pnls, [5, 25, 50, 75, 95])
    actual_final = actual_curve[-1]
    print(f"=== {name} System Monte Carlo Breakdown ===")
    print(f"Your Actual Final P&L: ${actual_final:,.2f}")
    print(f"Simulated Final P&L (Median): ${percentiles[2]:,.2f}")
    print(f"Simulated Final P&L (5th - 95th percentile): ${percentiles[0]:,.2f} to ${percentiles[4]:,.2f}")
    print(f"Simulated Final P&L (25th - 75th percentile): ${percentiles[1]:,.2f} to ${percentiles[3]:,.2f}\n")
    if percentiles[2] < 0:
        print("- Most simulated outcomes are negative. This system has a negative edge and is likely to lose money over time.\n")
    elif percentiles[2] > 0 and percentiles[0] < 0:
        print("- Outcomes are mixed. This system is close to break-even; luck and variance play a big role.\n")
    else:
        print("- Most simulated outcomes are positive. This system has a strong positive edge and is likely to make money over time.\n")
    print("- The blue line is your actual result. If it's above most simulations, you outperformed your stats; below means you underperformed.\n")
    print("="*60 + "\n") 

# === ENSURE CONTRACT TYPE COLUMN EXISTS BEFORE ANALYSIS ===
def get_contract_type(name):
    name = str(name).upper()
    if 'MNQ' in name:
        return 'Micro (MNQ)'
    elif 'NQ' in name:
        return 'Mini (NQ)'
    else:
        return f'Other ({name})'

# Always (re)create ContractType before position sizing analysis
if 'ContractType' not in df.columns:
    df['ContractType'] = df['ContractName'].apply(get_contract_type)
else:
    # If it exists, refresh in case df was reloaded
    df['ContractType'] = df['ContractName'].apply(get_contract_type)

# === POSITION SIZING STATS AND BELL CURVE (BY CONTRACT TYPE AND SIZE) ===
pos_dir = os.path.join(OUTPUT_DIR, 'position_sizing_stats')
os.makedirs(pos_dir, exist_ok=True)

# Group by ContractType and Size for detailed stats
size_stats = df.groupby(['ContractType', 'Size']).agg(
    num_trades=('PnL', 'count'),
    avg_pnl=('PnL', 'mean'),
    win_rate=('Was_Win', 'mean'),
    total_pnl=('PnL', 'sum')
).reset_index()
size_stats['win_rate'] = size_stats['win_rate'] * 100
size_stats.to_csv(os.path.join(pos_dir, 'position_sizing_stats_by_contract.csv'), index=False)

print("\nPosition Sizing Stats by Contract Type and Size (saved as CSV):")
for contract in size_stats['ContractType'].unique():
    print(f"\n--- {contract} ---")
    print(size_stats[size_stats['ContractType'] == contract].to_string(index=False))

# 2. Bell Curve (Histogram + KDE) of Profits
import seaborn as sns
plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.histplot(df['PnL'], bins=30, kde=True, color='deepskyblue', edgecolor='white', stat='density', alpha=0.7)
plt.title('Distribution of Trade Profits (Bell Curve)', fontsize=16, color='white')
plt.xlabel('Trade P&L ($)', color='white')
plt.ylabel('Density', color='white')
plt.grid(True, linestyle='--', alpha=0.3, color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.tight_layout()
plt.savefig(os.path.join(pos_dir, 'bell_curve_profits.png'), facecolor='black')
plt.show() 

# === POSITION SIZING CONSISTENCY & CONTRACT TYPE ANALYSIS (ALL TYPES) ===
# Consistency: Standard deviation and mode of position size per contract type
consistency_stats = df.groupby('ContractType').agg(
    mean_size=('Size', 'mean'),
    std_size=('Size', 'std'),
    mode_size=('Size', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    min_size=('Size', 'min'),
    max_size=('Size', 'max'),
    num_trades=('Size', 'count')
).reset_index()

print("\nPosition Sizing Consistency by Contract Type (including Other):")
print(consistency_stats)
consistency_stats.to_csv(os.path.join(pos_dir, 'position_sizing_consistency_by_contract.csv'), index=False)

# Plot distribution of position sizes for each contract type (including Other)
import seaborn as sns
plt.style.use('dark_background')
plt.figure(figsize=(12,7))
sns.histplot(data=df, x='Size', hue='ContractType', multiple='dodge',
             bins=range(int(df['Size'].min()), int(df['Size'].max())+2),
             palette='tab10', edgecolor='white', alpha=0.8)
plt.title('Distribution of Position Sizes by Contract Type', fontsize=16, color='white')
plt.xlabel('Position Size (Contracts)', color='white')
plt.ylabel('Number of Trades', color='white')
plt.grid(True, linestyle='--', alpha=0.3, color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.legend(title='Contract Type', loc='best')
plt.tight_layout()
plt.savefig(os.path.join(pos_dir, 'position_size_distribution_by_contract.png'), facecolor='black')
plt.show() 

# === DETAILED EQUITY CURVE WITH SEGMENT MARKERS ===
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))
actual_pnl = np.cumsum(your_stats['trade_outcomes'])
trade_numbers = np.arange(1, len(actual_pnl) + 1)

ax.plot(trade_numbers, actual_pnl, color='red', linewidth=2, label='Equity Curve')
ax.scatter(trade_numbers, actual_pnl, color='cyan', s=60, edgecolor='black', zorder=3, label='Trade Segment')

ax.set_title('Detailed Equity Curve (Your Actual Trades)', fontsize=16, color='white')
ax.set_xlabel('Trade Number', color='white')
ax.set_ylabel('Cumulative P&L ($)', color='white')
ax.grid(True, linestyle='--', alpha=0.3, color='white')
ax.legend(facecolor='black', edgecolor='white', fontsize=12)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
fig.patch.set_facecolor('black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'detailed_equity_curve.png'), facecolor=fig.get_facecolor())
plt.show() 