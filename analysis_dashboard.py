import streamlit as st
import pandas as pd
import json
import os
import re

st.title("Trade + Analysis Metrics Dashboard")

# --- File Uploaders ---
trade_file = st.file_uploader("Upload Trade CSV", type="csv")

# Try to load the default analysis JSON if not uploaded
analysis_file = st.file_uploader("Upload Analysis JSON", type="json")
default_analysis_path = r"C:\Users\jakers\Desktop\Trade journal\analysis_drawings.json"

if not analysis_file and os.path.exists(default_analysis_path):
    with open(default_analysis_path, 'r') as f:
        analysis = json.load(f)
    analysis_loaded = True
else:
    analysis_loaded = False
    if analysis_file:
        analysis = json.load(analysis_file)
        analysis_loaded = True

if trade_file and analysis_loaded:
    # --- Load Data ---
    trades = pd.read_csv(trade_file)
    stages = pd.DataFrame(analysis)
    
    # Flatten the drawings array to extract individual patterns
    if 'drawings' in stages.columns:
        flattened_drawings = []
        for idx, row in stages.iterrows():
            if isinstance(row['drawings'], list):
                for drawing in row['drawings']:
                    drawing_data = {
                        'symbol': row['symbol'],
                        'timeframe': row['timeframe'],
                        'group': row['group'],
                        'contextKey': row['contextKey']
                    }
                    # Add analysis_time if it exists
                    if 'analysis_time' in row:
                        drawing_data['analysis_time'] = row['analysis_time']
                    # Add drawing fields
                    for key, value in drawing.items():
                        drawing_data[key] = value
                    flattened_drawings.append(drawing_data)
        
        if flattened_drawings:
            stages = pd.DataFrame(flattened_drawings)
            st.write(f"Flattened {len(flattened_drawings)} drawings from analysis data")
    
    # --- Convert Timestamps ---
    entry_col = 'EnteredAt' if 'EnteredAt' in trades.columns else 'entry_time'
    # Set symbol_col to 'ContractName' only if it exists
    if 'ContractName' in trades.columns:
        symbol_col = 'ContractName'
    else:
        st.warning(f"'ContractName' column not found in trades file. Available columns: {list(trades.columns)}")
        st.stop()
    trades[entry_col] = pd.to_datetime(trades[entry_col], errors='coerce')
    if 'timestamp' in stages.columns:
        stages['analysis_time'] = pd.to_datetime(stages['timestamp'], errors='coerce')
    elif 'metrics' in stages.columns and 'startTime' in stages['metrics'].iloc[0]:
        stages['analysis_time'] = pd.to_datetime(stages['metrics'].apply(lambda x: x['startTime']), unit='s', errors='coerce')
    else:
        stages['analysis_time'] = pd.NaT

    # --- Make both columns timezone-naive ---
    trades[entry_col] = trades[entry_col].dt.tz_localize(None)
    stages['analysis_time'] = stages['analysis_time'].dt.tz_localize(None)

    # --- Symbol matching ---
    if 'symbol' not in stages.columns and 'timeframeInfo' in stages.columns:
        stages['symbol'] = stages['timeframeInfo'].apply(lambda x: x.get('symbol') if isinstance(x, dict) else None)

    # Check for symbol column existence
    if 'symbol' not in stages.columns:
        st.warning("Symbol column missing in analysis. Please ensure the analysis file has a symbol column for proper matching.")
    else:
        # Only keep analysis rows with non-null symbol
        stages = stages[stages['symbol'].notnull()]
        trades = trades[trades[symbol_col].notnull()]
        # Check for matching symbols
        trade_syms = set(trades[symbol_col].unique())
        analysis_syms = set(stages['symbol'].unique())
        st.write("Unique ContractName in trades:", trade_syms)
        st.write("Unique symbol in analysis:", analysis_syms)
        matching_syms = trade_syms & analysis_syms
        st.write("Matching symbols:", matching_syms)
        if not matching_syms:
            st.warning(f"No matching symbols found between trades and analysis.\nTrades: {trade_syms}\nAnalysis: {analysis_syms}")
            st.stop()
        else:
            # Filter to only matching symbols
            trades = trades[trades[symbol_col].isin(matching_syms)]
            stages = stages[stages['symbol'].isin(matching_syms)]
            
            # --- Analysis: A1 and A2 patterns per symbol and timeframe ---
            st.header("A and S Patterns")
            if not stages.empty:
                # Get all patterns with "A" in the label (A1, A2, A3, etc.)
                if 'label' in stages.columns:
                    pattern_col = 'label'
                    a_patterns = stages[stages[pattern_col].str.contains('A[0-9]+', na=False)].copy()
                    
                    if not a_patterns.empty:
                        def extract_s_pattern_for_symbol(label, target_symbol):
                            # Only extract S-pattern if the label contains the target symbol
                            if target_symbol in str(label):
                                match = re.search(r'S([1-9][0-9]?)', str(label))
                                if match:
                                    s_num = int(match.group(1))
                                    if 1 <= s_num <= 99:
                                        return f'S{s_num}'
                            return ''
                        
                        def extract_a_pattern(label):
                            match = re.search(r'A([1-9][0-9]?)', str(label))
                            if match:
                                a_num = int(match.group(1))
                                if 1 <= a_num <= 99:
                                    return f'A{a_num}'
                            return ''
                        
                        # Create clean table with A and S patterns (symbol- and timeframe-aware)
                        results = []
                        for (symbol, timeframe), group in a_patterns.groupby(['symbol', 'timeframe']):
                            for _, row in group.iterrows():
                                label = str(row[pattern_col])
                                # Only process if label contains both symbol and timeframe
                                if symbol in label and timeframe in label:
                                    a_match = re.search(r'A([1-9][0-9]?)', label)
                                    s_match = re.search(r'S([1-9][0-9]?)', label)
                                    a_pattern = f"A{a_match.group(1)}" if a_match else ''
                                    s_pattern = f"S{s_match.group(1)}" if s_match else ''
                                    
                                    # Extract chart values if available
                                    start_time = ''
                                    end_time = ''
                                    start_price = ''
                                    end_price = ''
                                    if 'chartValues' in row and isinstance(row['chartValues'], dict):
                                        chart_vals = row['chartValues']
                                        if 'startPoint' in chart_vals and 'endPoint' in chart_vals:
                                            start_point = chart_vals['startPoint']
                                            end_point = chart_vals['endPoint']
                                            start_time = start_point.get('time', '')
                                            end_time = end_point.get('time', '')
                                            start_price = start_point.get('price', '')
                                            end_price = end_point.get('price', '')
                                    
                                    # Extract metrics if available
                                    bar_count = ''
                                    percent_change = ''
                                    raw_percent_change = ''
                                    price_range = ''
                                    price_change = ''
                                    high_price = ''
                                    low_price = ''
                                    bullish = ''
                                    direction = ''
                                    if 'metrics' in row and isinstance(row['metrics'], dict):
                                        metrics = row['metrics']
                                        bar_count = metrics.get('barCount', '')
                                        percent_change = metrics.get('percentChange', '')
                                        raw_percent_change = metrics.get('rawPercentChange', '')
                                        price_range = metrics.get('priceRange', '')
                                        price_change = metrics.get('priceChange', '')
                                        high_price = metrics.get('highPrice', '')
                                        low_price = metrics.get('lowPrice', '')
                                        bullish = metrics.get('bullish', '')
                                        # Only get direction for A1 and A2 patterns
                                        if a_pattern in ['A1', 'A2']:
                                            direction = metrics.get('direction', '')
                                        else:
                                            direction = 'N/A'
                                    
                                    if a_pattern:
                                        results.append({
                                            'Symbol': symbol,
                                            'Timeframe': timeframe,
                                            'A_Pattern': a_pattern,
                                            'S_Pattern': s_pattern,
                                            'Start_Time_Chart': start_time,
                                            'End_Time_Chart': end_time,
                                            'Start_Price': start_price,
                                            'End_Price': end_price,
                                            'Bar_Count': bar_count,
                                            'Percent_Change': percent_change,
                                            'Raw_Percent_Change': raw_percent_change,
                                            'Price_Range': price_range,
                                            'Price_Change': price_change,
                                            'High_Price': high_price,
                                            'Low_Price': low_price,
                                            'Bullish': bullish,
                                            'Direction': direction
                                        })
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                            
                            # Analysis Output Metrics
                            st.header("Analysis Output Metrics")
                            
                            # Filter for A1 and A2 patterns only
                            a1_a2_df = results_df[results_df['A_Pattern'].isin(['A1', 'A2'])]
                            
                            if not a1_a2_df.empty:
                                # Calculate averages for Percent_Change and Bar_Count by symbol, timeframe, and A_pattern
                                st.write("**Averages for A1 and A2 Patterns (by Symbol, Timeframe, and Pattern):**")
                                
                                # Group by symbol, timeframe, and A_pattern, then calculate averages
                                grouped_avg = a1_a2_df.groupby(['Symbol', 'Timeframe', 'A_Pattern']).agg({
                                    'Percent_Change': lambda x: pd.to_numeric(x.str.replace('%', '').str.replace('+', ''), errors='coerce').mean(),
                                    'Bar_Count': lambda x: pd.to_numeric(x, errors='coerce').mean()
                                }).reset_index()
                                
                                # Rename columns for clarity
                                grouped_avg.columns = ['Symbol', 'Timeframe', 'A_Pattern', 'Avg_Percent_Change', 'Avg_Bar_Count']
                                
                                # Format percent change
                                grouped_avg['Avg_Percent_Change'] = grouped_avg['Avg_Percent_Change'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
                                grouped_avg['Avg_Bar_Count'] = grouped_avg['Avg_Bar_Count'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                                
                                st.dataframe(grouped_avg)
                                
                                # Show pattern counts by symbol and timeframe
                                pattern_counts = a1_a2_df.groupby(['Symbol', 'Timeframe', 'A_Pattern']).size().reset_index(name='Count')
                                st.write("**Pattern Counts by Symbol and Timeframe:**")
                                st.dataframe(pattern_counts)
                                
                                # --- Fibonacci Grid Analysis ---
                                st.markdown('<h3 style="color: gold;">Fibonacci Grid Analysis</h3>', unsafe_allow_html=True)
                                st.info("""
                                **Stage 2 (A14/A15) - Full Fibonacci Grid:**
                                A14 = Bottom (1.0), A15 = Top (0.0)
                                Levels: 0.85 (Bullish INV), 0.7 (Red), 0.5 (Grey), 0.3 (Green), 0.15 (Green), 0.0 (Green)
                                **Stage 1 (A12/A13) - Simplified Grid:**
                                A12 = Bottom (1.0), A13 = Top (0.0)
                                Levels: 1.0, 0.5, 0.0
                                **Purpose:** Find where A1 center point falls within the Fibonacci grid based on S-pattern grouping.
                                """)
                                
                                # Find A14/A15 and A12/A13 pairs and analyze A1 positions by S-pattern
                                fib_results = []
                                
                                for (symbol, timeframe), group in results_df.groupby(['Symbol', 'Timeframe']):
                                    # Group by S-pattern within this symbol/timeframe
                                    for s_pattern in group['S_Pattern'].unique():
                                        if s_pattern:  # Only process if S-pattern exists
                                            s_group = group[group['S_Pattern'] == s_pattern]
                                            
                                            # Find A14/A15 pairs (Stage 2) and A12/A13 pairs (Stage 1) for this S-pattern
                                            a14_patterns = s_group[s_group['A_Pattern'] == 'A14']
                                            a15_patterns = s_group[s_group['A_Pattern'] == 'A15']
                                            a12_patterns = s_group[s_group['A_Pattern'] == 'A12']
                                            a13_patterns = s_group[s_group['A_Pattern'] == 'A13']
                                            
                                            # Find A1 and A2 patterns for this S-pattern
                                            a1_patterns = s_group[s_group['A_Pattern'] == 'A1']
                                            a2_patterns = s_group[s_group['A_Pattern'] == 'A2']
                                            
                                            # Establish Fibonacci grid from A14/A15 or A12/A13
                                            fib_bottom = None
                                            fib_top = None
                                            fib_type = None
                                            
                                            # Check for A14/A15 pair first (Stage 2)
                                            if len(a14_patterns) > 0 and len(a15_patterns) > 0:
                                                fib_bottom = float(a15_patterns.iloc[0]['End_Price'])  # A15 = bottom (0.0)
                                                fib_top = float(a14_patterns.iloc[0]['End_Price'])     # A14 = top (1.0)
                                                fib_type = 'Stage2'
                                            # If no A14/A15, check for A12/A13 pair (Stage 1)
                                            elif len(a12_patterns) > 0 and len(a13_patterns) > 0:
                                                fib_bottom = float(a13_patterns.iloc[0]['End_Price'])  # A13 = bottom (0.0)
                                                fib_top = float(a12_patterns.iloc[0]['End_Price'])     # A12 = top (1.0)
                                                fib_type = 'Stage1'
                                            
                                            # If we have a Fibonacci grid, analyze A1 patterns
                                            if fib_bottom is not None and fib_top is not None:
                                                # Sort A1 and A2 by time for sequence analysis
                                                if 'Start_Time_Chart' in a1_patterns.columns:
                                                    a1_patterns = a1_patterns.sort_values('Start_Time_Chart')
                                                if 'Start_Time_Chart' in a2_patterns.columns:
                                                    a2_patterns = a2_patterns.sort_values('Start_Time_Chart')
                                                
                                                # Analyze each A1 that comes after an A2
                                                for idx, a1_row in a1_patterns.iterrows():
                                                    a1_time = a1_row.get('Start_Time_Chart', idx)
                                                    prev_a2 = None
                                                    
                                                    # Find the A2 that comes before this A1
                                                    for a2_idx, a2_row in a2_patterns.iterrows():
                                                        a2_time = a2_row.get('Start_Time_Chart', a2_idx)
                                                        if a2_time < a1_time:
                                                            if prev_a2 is None or a2_time > prev_a2.get('Start_Time_Chart', 0):
                                                                prev_a2 = a2_row
                                                    
                                                    if prev_a2 is not None:
                                                        # Calculate A1 center point
                                                        a1_start_price = float(a1_row.get('Start_Price', 0))
                                                        a1_end_price = float(a1_row.get('End_Price', 0))
                                                        a1_center_price = (a1_start_price + a1_end_price) / 2
                                                        
                                                        # Calculate Fibonacci retracement level
                                                        price_range = fib_top - fib_bottom
                                                        if price_range != 0:
                                                            fib_level = (a1_center_price - fib_bottom) / price_range
                                                            
                                                            # Determine zone based on Fibonacci type
                                                            zone = 'Unknown'
                                                            if fib_type == 'Stage2':
                                                                if fib_level >= 0.85:
                                                                    zone = 'Bullish INV'
                                                                elif fib_level >= 0.7:
                                                                    zone = 'Red'
                                                                elif fib_level >= 0.5:
                                                                    zone = 'Grey/Neutral'
                                                                elif fib_level >= 0.3:
                                                                    zone = 'Green'
                                                                elif fib_level >= 0.15:
                                                                    zone = 'Green'
                                                                else:
                                                                    zone = 'Green'
                                                            elif fib_type == 'Stage1':
                                                                if fib_level >= 0.5:
                                                                    zone = 'Upper Half'
                                                                else:
                                                                    zone = 'Lower Half'
                                                            
                                                            fib_results.append({
                                                                'Symbol': symbol,
                                                                'Timeframe': timeframe,
                                                                'S_Pattern': s_pattern,
                                                                'A1_Start_Time': a1_row.get('Start_Time_Chart', 'N/A'),
                                                                'A1_Center_Price': f"{a1_center_price:.2f}",
                                                                'Fib_Type': fib_type,
                                                                'Fib_Bottom': f"{fib_bottom:.2f}",
                                                                'Fib_Top': f"{fib_top:.2f}",
                                                                'Fib_Level': f"{fib_level:.3f}",
                                                                'Zone': zone
                                                            })
                                
                                if fib_results:
                                    fib_df = pd.DataFrame(fib_results)
                                    st.dataframe(fib_df)
                                else:
                                    st.info("No Fibonacci grid analysis data available.")
                                
                                # --- NEURAL NETWORK: Normalization Section ---
                                st.markdown('<h3 style="color: gold;">🧠 Neural Network Pattern Analysis</h3>', unsafe_allow_html=True)
                                
                                # Calculate normalized_change for each pattern
                                if not a1_a2_df.empty:
                                    # Create a copy to avoid SettingWithCopyWarning
                                    nn_df = a1_a2_df.copy()
                                    # Compute group averages for percent change
                                    nn_df['Percent_Change_Numeric'] = pd.to_numeric(nn_df['Percent_Change'].str.replace('%', '').str.replace('+', ''), errors='coerce')
                                    group_avg = nn_df.groupby(['Symbol', 'Timeframe', 'A_Pattern'])['Percent_Change_Numeric'].transform('mean')
                                    nn_df['Normalized_Change'] = nn_df['Percent_Change_Numeric'] / group_avg
                                    
                                    # Calculate normalized bars
                                    nn_df['Bar_Count_Numeric'] = pd.to_numeric(nn_df['Bar_Count'], errors='coerce')
                                    bar_group_avg = nn_df.groupby(['Symbol', 'Timeframe', 'A_Pattern'])['Bar_Count_Numeric'].transform('mean')
                                    nn_df['Normalized_Bars'] = nn_df['Bar_Count_Numeric'] / bar_group_avg
                                    
                                    # Pattern Strength Scoring
                                    nn_df['Pattern_Strength'] = (
                                        nn_df['Normalized_Change'] * 0.4 +  # 40% weight to size
                                        nn_df['Normalized_Bars'] * 0.3 +    # 30% weight to duration
                                        (nn_df['Percent_Change_Numeric'] / 100) * 0.3  # 30% weight to raw performance
                                    )
                                    
                                    # Volatility Score (inverse of consistency)
                                    volatility_scores = []
                                    for (symbol, timeframe, pattern), group in nn_df.groupby(['Symbol', 'Timeframe', 'A_Pattern']):
                                        if len(group) > 1:
                                            std_dev = group['Percent_Change_Numeric'].std()
                                            mean_val = group['Percent_Change_Numeric'].mean()
                                            volatility = std_dev / mean_val if mean_val != 0 else 0
                                            volatility_scores.extend([volatility] * len(group))
                                        else:
                                            volatility_scores.extend([0] * len(group))
                                    
                                    nn_df['Volatility_Score'] = volatility_scores
                                    
                                    # Pattern Confidence Score (1 = most consistent, 0 = least consistent)
                                    nn_df['Pattern_Confidence'] = 1 / (1 + nn_df['Volatility_Score'])
                                    
                                    # Calculate z-scores for each pattern
                                    z_scores = []
                                    for (symbol, timeframe, pattern), group in nn_df.groupby(['Symbol', 'Timeframe', 'A_Pattern']):
                                        if len(group) > 1:
                                            mean_val = group['Percent_Change_Numeric'].mean()
                                            std_dev = group['Percent_Change_Numeric'].std()
                                            for _, row in group.iterrows():
                                                z_score = (row['Percent_Change_Numeric'] - mean_val) / std_dev if std_dev != 0 else 0
                                                z_scores.append(z_score)
                                        else:
                                            z_scores.extend([0] * len(group))
                                    
                                    nn_df['Z_Score'] = z_scores
                                    
                                    # --- CONSOLIDATED NEURAL NETWORK TABLE ---
                                    st.markdown('<h4 style="color: gold;">🧠 Complete Neural Network Metrics</h4>', unsafe_allow_html=True)
                                    
                                    # Create clean neural network table with key metrics only
                                    neural_table = nn_df[['Symbol', 'Timeframe', 'A_Pattern', 'Percent_Change', 'Bar_Count',
                                                        'Normalized_Change', 'Normalized_Bars', 'Pattern_Strength', 
                                                        'Volatility_Score', 'Pattern_Confidence', 'Z_Score']].copy()
                                    
                                    # Format numeric columns
                                    neural_table['Normalized_Change'] = neural_table['Normalized_Change'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
                                    neural_table['Normalized_Bars'] = neural_table['Normalized_Bars'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
                                    neural_table['Pattern_Strength'] = neural_table['Pattern_Strength'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
                                    neural_table['Volatility_Score'] = neural_table['Volatility_Score'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
                                    neural_table['Pattern_Confidence'] = neural_table['Pattern_Confidence'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
                                    neural_table['Z_Score'] = neural_table['Z_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
                                    
                                    st.info("""
                                    **📊 Complete Neural Network Metrics Explained:**
                                    
                                    | Metric | Formula | Purpose |
                                    |--------|---------|---------|
                                    | **Normalized_Change** | pattern_value / group_average | How big vs typical for this pattern type |
                                    | **Normalized_Bars** | bar_count / group_average | How long vs typical for this pattern type |
                                    | **Pattern_Strength** | Weighted combo of all metrics | Single score for pattern significance |
                                    | **Volatility_Score** | std_dev / mean | How consistent this pattern type is |
                                    | **Pattern_Confidence** | 1 / (1 + volatility) | Reliability score (1 = most reliable) |
                                    | **Z_Score** | (value - mean) / std_dev | How extreme vs group average |
                                    """)
                                    
                                    # Create comprehensive table with ALL data including succession and S-group info
                                    comprehensive_table = neural_table.copy()
                                    
                                    # Add succession info using analysis_time for proper sequence
                                    a1_a2_patterns = nn_df[nn_df['A_Pattern'].isin(['A1', 'A2'])].copy()
                                    if not a1_a2_patterns.empty:
                                        # Use analysis_time for proper pattern sequence
                                        if 'analysis_time' in a1_a2_patterns.columns:
                                            a1_a2_patterns = a1_a2_patterns.sort_values(['Symbol', 'Timeframe', 'analysis_time'])
                                        else:
                                            a1_a2_patterns = a1_a2_patterns.sort_values(['Symbol', 'Timeframe'])
                                        
                                        # Add succession context to each pattern
                                        for (symbol, timeframe), group in a1_a2_patterns.groupby(['Symbol', 'Timeframe']):
                                            patterns = group['A_Pattern'].tolist()
                                            analysis_times = group['analysis_time'].tolist() if 'analysis_time' in group.columns else [None] * len(patterns)
                                            
                                            # Only process if we have multiple patterns in sequence
                                            if len(patterns) > 1:
                                                for i in range(len(patterns)):
                                                    # Find the exact row in comprehensive_table
                                                    pattern_mask = (comprehensive_table['Symbol'] == symbol) & \
                                                                 (comprehensive_table['Timeframe'] == timeframe) & \
                                                                 (comprehensive_table['A_Pattern'] == patterns[i])
                                                    
                                                    # Only update if this pattern exists in comprehensive_table
                                                    if pattern_mask.any():
                                                        has_next = i < len(patterns) - 1
                                                        next_pattern = patterns[i + 1] if has_next else 'None'
                                                        comprehensive_table.loc[pattern_mask, 'Has_Next'] = has_next
                                                        comprehensive_table.loc[pattern_mask, 'Next_Pattern'] = next_pattern
                                                        
                                                        # Calculate time to next using analysis_time
                                                        if has_next and pd.notna(analysis_times[i]) and pd.notna(analysis_times[i + 1]):
                                                            try:
                                                                time1 = pd.to_datetime(analysis_times[i])
                                                                time2 = pd.to_datetime(analysis_times[i + 1])
                                                                time_to_next = (time2 - time1).total_seconds() / 60  # minutes
                                                                comprehensive_table.loc[pattern_mask, 'Time_To_Next_Minutes'] = time_to_next
                                                            except:
                                                                comprehensive_table.loc[pattern_mask, 'Time_To_Next_Minutes'] = None
                                    
                                    # Add S-group info
                                    s_patterns = nn_df[nn_df['S_Pattern'].notna() & (nn_df['S_Pattern'] != '')].copy()
                                    if not s_patterns.empty:
                                        for (symbol, timeframe, s_pattern), group in s_patterns.groupby(['Symbol', 'Timeframe', 'S_Pattern']):
                                            group_size = len(group)
                                            avg_strength = group['Pattern_Strength'].mean()
                                            avg_confidence = group['Pattern_Confidence'].mean()
                                            
                                            for _, row in group.iterrows():
                                                pattern_idx = comprehensive_table[(comprehensive_table['Symbol'] == symbol) & 
                                                                                (comprehensive_table['Timeframe'] == timeframe) & 
                                                                                (comprehensive_table['A_Pattern'] == row['A_Pattern'])].index
                                                if len(pattern_idx) > 0:
                                                    comprehensive_table.loc[pattern_idx, 'S_Pattern'] = s_pattern
                                                    comprehensive_table.loc[pattern_idx, 'S_Group_Size'] = group_size
                                                    comprehensive_table.loc[pattern_idx, 'S_Group_Avg_Strength'] = f"{avg_strength:.3f}" if pd.notna(avg_strength) else 'N/A'
                                                    comprehensive_table.loc[pattern_idx, 'S_Group_Avg_Confidence'] = f"{avg_confidence:.3f}" if pd.notna(avg_confidence) else 'N/A'
                                    
                                    # Fill NaN values
                                    comprehensive_table = comprehensive_table.fillna('N/A')
                                    
                                    st.dataframe(comprehensive_table)
                                    
                                    # Export for ML training
                                    st.markdown('<h4 style="color: gold;">📤 Export for Machine Learning</h4>', unsafe_allow_html=True)
                                    
                                    # Create comprehensive ML-ready dataset with all features
                                    ml_dataset = nn_df[['Symbol', 'Timeframe', 'A_Pattern', 'S_Pattern', 'Normalized_Change', 'Normalized_Bars', 
                                                       'Pattern_Strength', 'Volatility_Score', 'Pattern_Confidence', 'Z_Score',
                                                       'Percent_Change_Numeric', 'Bar_Count_Numeric', 'Start_Time_Chart', 'End_Time_Chart']].copy()
                                    
                                    # Remove rows with NaN values and duplicates
                                    ml_dataset = ml_dataset.dropna()
                                    ml_dataset = ml_dataset.drop_duplicates()
                                    
                                    if not ml_dataset.empty:
                                        # Add one-hot encoding for categorical variables
                                        categorical_cols = ['Symbol', 'Timeframe', 'A_Pattern', 'S_Pattern', 'Next_Pattern', 'S_Group_Consistency']
                                        existing_categorical_cols = [col for col in categorical_cols if col in ml_dataset.columns]
                                        
                                        ml_dataset_encoded = pd.get_dummies(ml_dataset, columns=existing_categorical_cols, 
                                                                           prefix=['Symbol', 'Timeframe', 'Pattern', 'S_Pattern', 'Next_Pattern', 'S_Consistency'])
                                        
                                        # Create download button
                                        csv_data = ml_dataset_encoded.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Complete ML Dataset (CSV)",
                                            data=csv_data,
                                            file_name="complete_neural_network_patterns.csv",
                                            mime="text/csv"
                                        )
                                        
                                        st.info(f"**Complete ML Dataset Summary:** {len(ml_dataset_encoded)} patterns, {len(ml_dataset_encoded.columns)} features")
                                        st.write("**Enhanced Feature Categories:**")
                                        st.write("- **Core Metrics:** Normalized_Change, Normalized_Bars, Pattern_Strength, etc.")
                                        st.write("- **Succession Features:** Has_Next_Pattern, Next_Pattern, Time_To_Next_Minutes")
                                        st.write("- **S-Group Features:** S_Group_Size, S_Group_Avg_Strength, S_Group_Consistency")
                                        st.write("- **Categorical Features:** Symbol, Timeframe, Pattern types (one-hot encoded)")
                                        
                                        # Show sample of enhanced ML data with clean row numbers
                                        st.write("**Sample Enhanced ML Data (first 5 rows):**")
                                        sample_data = ml_dataset_encoded.head().reset_index(drop=True)
                                        st.dataframe(sample_data)
                                    else:
                                        st.warning("No complete data available for ML export (missing values)")
                                else:
                                    st.info("No A1 or A2 patterns found for analysis.")
                            else:
                                st.info("No A1 or A2 patterns found for analysis.")
                        else:
                            st.info("No A patterns found.")
                else:
                    st.info("No 'label' column found in analysis.")
            else:
                st.info("No analysis data available.")
            # Rename symbol column in stages to match trades
            stages = stages.rename(columns={'symbol': symbol_col})
            
            # Sort for merge_asof
            trades = trades.sort_values([symbol_col, entry_col])
            stages = stages.sort_values([symbol_col, 'analysis_time'])
            
            # Final safety check before merge
            if symbol_col not in trades.columns:
                st.error(f"Column '{symbol_col}' missing from trades DataFrame!")
                st.stop()
            if 'symbol' not in stages.columns:
                st.error("Column 'symbol' missing from stages DataFrame!")
                st.stop()
            if len(trades) == 0:
                st.error("Trades DataFrame is empty!")
                st.stop()
            if len(stages) == 0:
                st.error("Stages DataFrame is empty!")
                st.stop()
            
            # --- Merge on both symbol and time ---
            try:
                merged = pd.merge_asof(
                    trades,
                    stages,
                    left_on=entry_col,
                    right_on='analysis_time',
                    by=symbol_col,
                    direction='backward',
                    suffixes=('', '_analysis')
                )
                st.write("Merge successful!")
            except Exception as e:
                st.error(f"Merge failed: {str(e)}")
                st.stop()
            # Attach trade Id to analysis rows
            if 'Id' in trades.columns:
                merged['TradeId'] = merged['Id']
            elif 'trade_id' in trades.columns:
                merged['TradeId'] = merged['trade_id']
            else:
                merged['TradeId'] = merged.index

            # --- Mathematical/statistical outputs ---
            st.header("Mathematical Outputs for Filtered Trades")
            if not merged.empty:
                merged['Was_Win'] = merged['PnL'] > 0
                total_pnl = merged['PnL'].sum()
                win_pct = (merged['Was_Win'].mean() * 100)
                avg_win = merged[merged['Was_Win']]['PnL'].mean()
                avg_loss = merged[~merged['Was_Win']]['PnL'].mean()
                win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('nan')
                profit_factor = merged[merged['PnL'] > 0]['PnL'].sum() / abs(merged[merged['PnL'] < 0]['PnL'].sum()) if (merged[merged['PnL'] < 0]['PnL'].sum()) != 0 else float('nan')
                num_trades = len(merged)
                best_trade = merged.loc[merged['PnL'].idxmax()]
                worst_trade = merged.loc[merged['PnL'].idxmin()]
                st.markdown(f"**Total P&L:** ${total_pnl:,.2f}")
                st.markdown(f"**Trade Win %:** {win_pct:.2f}%")
                st.markdown(f"**Avg Win / Avg Loss:** {win_loss_ratio:.2f}")
                st.markdown(f"**Avg Winning Trade:** ${avg_win:.2f}")
                st.markdown(f"**Avg Losing Trade:** ${avg_loss:.2f}")
                st.markdown(f"**Profit Factor:** {profit_factor:.2f}")
                st.markdown(f"**Total Number of Trades:** {num_trades}")
                st.markdown(f"**Best Trade:** ${best_trade['PnL']:.2f} | {best_trade['Type']} {best_trade['Size']} {best_trade[symbol_col]} @ {best_trade['EntryPrice']} Exited @ {best_trade['ExitPrice']} {best_trade['ExitedAt']}")
                st.markdown(f"**Worst Trade:** ${worst_trade['PnL']:.2f} | {worst_trade['Type']} {worst_trade['Size']} {worst_trade[symbol_col]} @ {worst_trade['EntryPrice']} Exited @ {worst_trade['ExitPrice']} {worst_trade['ExitedAt']}")
            else:
                st.info("No merged trades available for analysis.")
            # --- Filter by Trade Id ---
            st.header("Filter by Trade Id")
            unique_ids = merged['TradeId'].dropna().unique()
            selected_id = st.selectbox("Select Trade Id", options=unique_ids)
            st.dataframe(merged[merged['TradeId'] == selected_id])
else:
    st.info("Please upload a Trade CSV and/or ensure analysis_drawings.json is available.") 