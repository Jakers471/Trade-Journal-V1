import streamlit as st
import pandas as pd
import json
import os
import re
import numpy as np

st.title("Trade + Analysis Metrics Dashboard")

# Add navigation to ML section
if 'beast_ml_ready' in st.session_state and st.session_state['beast_ml_ready']:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Jump to ML Training Section", type="secondary", use_container_width=True):
            st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
    st.markdown("---")

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
    elif 'metrics' in stages.columns:
        # Check if metrics column contains dictionaries
        first_metric = stages['metrics'].iloc[0] if not stages.empty else None
        if isinstance(first_metric, dict) and 'startTime' in first_metric:
            try:
                stages['analysis_time'] = pd.to_datetime(
                    stages['metrics'].apply(lambda x: x['startTime'] if isinstance(x, dict) and 'startTime' in x else None), 
                    unit='s', 
                    errors='coerce'
                )
            except Exception as e:
                st.warning(f"Error converting metrics startTime: {e}")
                stages['analysis_time'] = pd.NaT
        else:
            stages['analysis_time'] = pd.NaT
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
                                
                                # --- Comprehensive A-Pattern Spatial-Temporal Analysis ---
                                st.markdown('<h3 style="color: gold;">📍 A-Pattern Spatial-Temporal Trade Mapping</h3>', unsafe_allow_html=True)
                                st.info("""
                                **A-Pattern Anchor Point Analysis:**
                                - **Lines:** 3 anchor points (start, middle, end)
                                - **Rectangles:** 9 anchor points (top-left, top-center, top-right, middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right)
                                - **VAL/INV Borders:** Validation and invalidation levels for each A-pattern
                                - **Purpose:** Map trade positions relative to ALL A-patterns within each S-group using multiple anchor points.
                                """)
                                
                                # Comprehensive A-pattern spatial-temporal analysis with singular ID grouping
                                a_pattern_spatial_results = []
                                mapped_trade_ids = set()  # Track which trades were mapped
                                
                                # Group by S-pattern to analyze all A-patterns within each group
                                for (symbol, timeframe), group in results_df.groupby(['Symbol', 'Timeframe']):
                                    for s_pattern in group['S_Pattern'].unique():
                                        if s_pattern:  # Only process if S-pattern exists
                                            s_group = group[group['S_Pattern'] == s_pattern]
                                            
                                            # Get all A-patterns in this S-group
                                            a_patterns = s_group[s_group['A_Pattern'].notna()].copy()
                                            
                                            if not a_patterns.empty and 'trades' in locals() and not trades.empty:
                                                # Convert trade times to Unix timestamps
                                                trades['EnteredAt_unix'] = pd.to_datetime(trades['EnteredAt']).astype(np.int64) // 10**9
                                                trades['ExitedAt_unix'] = pd.to_datetime(trades['ExitedAt']).astype(np.int64) // 10**9
                                                
                                                # Find trades that occurred during this S-pattern timeframe
                                                s_start_times = pd.to_numeric(a_patterns['Start_Time_Chart'], errors='coerce')
                                                s_end_times = pd.to_numeric(a_patterns['End_Time_Chart'], errors='coerce')
                                                s_temporal_start = s_start_times.min()
                                                s_temporal_end = s_end_times.max()
                                                
                                                relevant_trades = trades[
                                                    (trades['ContractName'] == symbol) &
                                                    (trades['EnteredAt_unix'] >= s_temporal_start) &
                                                    (trades['ExitedAt_unix'] <= s_temporal_end)
                                                ]
                                                
                                                # For each trade, analyze its relationship to ALL A-patterns in the S-group
                                                for _, trade in relevant_trades.iterrows():
                                                    trade_id = trade['Id']
                                                    entry_price = trade['EntryPrice']
                                                    exit_price = trade['ExitPrice']
                                                    entry_time = trade['EnteredAt_unix']
                                                    exit_time = trade['ExitedAt_unix']
                                                    
                                                    # Determine which A-pattern the trade is INSIDE (if any)
                                                    trade_inside_pattern = None
                                                    for _, a_pattern in a_patterns.iterrows():
                                                        a_start_time = pd.to_numeric(a_pattern['Start_Time_Chart'], errors='coerce')
                                                        a_end_time = pd.to_numeric(a_pattern['End_Time_Chart'], errors='coerce')
                                                        a_start_price = pd.to_numeric(a_pattern['Start_Price'], errors='coerce')
                                                        a_end_price = pd.to_numeric(a_pattern['End_Price'], errors='coerce')
                                                        
                                                        # Check if trade is inside this A-pattern
                                                        if (entry_time >= a_start_time and entry_time <= a_end_time and
                                                            entry_price >= min(a_start_price, a_end_price) and 
                                                            entry_price <= max(a_start_price, a_end_price)):
                                                            trade_inside_pattern = a_pattern['A_Pattern']
                                                            break
                                                    
                                                    # Now analyze trade's relationship to ALL A-patterns in the S-group
                                                    for _, a_pattern in a_patterns.iterrows():
                                                        a_pattern_type = a_pattern['A_Pattern']
                                                        a_start_time = pd.to_numeric(a_pattern['Start_Time_Chart'], errors='coerce')
                                                        a_end_time = pd.to_numeric(a_pattern['End_Time_Chart'], errors='coerce')
                                                        a_start_price = pd.to_numeric(a_pattern['Start_Price'], errors='coerce')
                                                        a_end_price = pd.to_numeric(a_pattern['End_Price'], errors='coerce')
                                                        
                                                        # Get singular_id if available
                                                        singular_id = a_pattern.get('singular_id', 'N/A')
                                                        
                                                        # Determine pattern type (line or rectangle) based on drawing type
                                                        pattern_type = 'rectangle'  # Default assumption
                                                        
                                                        # Calculate VAL/INV borders for this A-pattern
                                                        a_validation_level = max(a_start_price, a_end_price)
                                                        a_invalidation_level = min(a_start_price, a_end_price)
                                                        a_price_range = a_validation_level - a_invalidation_level
                                                        
                                                        # Define anchor points based on pattern type
                                                        if pattern_type == 'line':
                                                            # 3 anchor points for lines
                                                            anchor_points = [
                                                                {'name': 'start', 'time': a_start_time, 'price': a_start_price},
                                                                {'name': 'middle', 'time': (a_start_time + a_end_time) / 2, 'price': (a_start_price + a_end_price) / 2},
                                                                {'name': 'end', 'time': a_end_time, 'price': a_end_price}
                                                            ]
                                                        else:  # rectangle
                                                            # 9 anchor points for rectangles
                                                            anchor_points = [
                                                                {'name': 'top_left', 'time': a_start_time, 'price': a_validation_level},
                                                                {'name': 'top_center', 'time': (a_start_time + a_end_time) / 2, 'price': a_validation_level},
                                                                {'name': 'top_right', 'time': a_end_time, 'price': a_validation_level},
                                                                {'name': 'middle_left', 'time': a_start_time, 'price': (a_validation_level + a_invalidation_level) / 2},
                                                                {'name': 'middle_center', 'time': (a_start_time + a_end_time) / 2, 'price': (a_validation_level + a_invalidation_level) / 2},
                                                                {'name': 'middle_right', 'time': a_end_time, 'price': (a_validation_level + a_invalidation_level) / 2},
                                                                {'name': 'bottom_left', 'time': a_start_time, 'price': a_invalidation_level},
                                                                {'name': 'bottom_center', 'time': (a_start_time + a_end_time) / 2, 'price': a_invalidation_level},
                                                                {'name': 'bottom_right', 'time': a_end_time, 'price': a_invalidation_level}
                                                            ]
                                                        
                                                        # Calculate distances to each anchor point
                                                        anchor_distances = []
                                                        for anchor in anchor_points:
                                                            # Price distance
                                                            price_distance = abs(entry_price - anchor['price'])
                                                            # Time distance (convert to minutes)
                                                            time_distance_minutes = abs(entry_time - anchor['time']) / 60
                                                            
                                                            anchor_distances.append({
                                                                'anchor': anchor['name'],
                                                                'price_distance': price_distance,
                                                                'time_distance_minutes': time_distance_minutes
                                                            })
                                                        
                                                        # Find closest anchor point
                                                        closest_anchor = min(anchor_distances, key=lambda x: x['price_distance'] + x['time_distance_minutes'])
                                                        
                                                        # Calculate retracement within A-pattern
                                                        if a_price_range > 0:
                                                            entry_retracement = (entry_price - a_invalidation_level) / a_price_range
                                                            exit_retracement = (exit_price - a_invalidation_level) / a_price_range
                                                        else:
                                                            entry_retracement = 0.5
                                                            exit_retracement = 0.5
                                                        
                                                        # Calculate temporal positioning within A-pattern
                                                        a_duration = a_end_time - a_start_time
                                                        if a_duration > 0:
                                                            entry_time_percent = (entry_time - a_start_time) / a_duration
                                                            exit_time_percent = (exit_time - a_start_time) / a_duration
                                                        else:
                                                            entry_time_percent = 0.5
                                                            exit_time_percent = 0.5
                                                        
                                                        # Calculate spatial and temporal scores
                                                        spatial_score = 1 - (closest_anchor['price_distance'] / a_price_range) if a_price_range > 0 else 0.5
                                                        temporal_score = 1 - abs(entry_time_percent - 0.5)  # Closer to middle = higher score
                                                        
                                                        # Determine relationship type
                                                        if trade_inside_pattern == a_pattern_type:
                                                            relationship_type = 'INSIDE'
                                                        else:
                                                            relationship_type = 'RELATIVE_TO'
                                                        
                                                        a_pattern_spatial_results.append({
                                                            'Symbol': symbol,
                                                            'Timeframe': timeframe,
                                                            'S_Pattern': s_pattern,
                                                            'Singular_ID': singular_id,
                                                            'Trade_Inside_Pattern': trade_inside_pattern,
                                                            'Analyzed_A_Pattern': a_pattern_type,
                                                            'Relationship_Type': relationship_type,
                                                            'Pattern_Type': pattern_type,
                                                            'Trade_Id': trade_id,
                                                            'Trade_Type': trade['Type'],
                                                            'Trade_PnL': trade['PnL'],
                                                            'A_Validation_Level': f"{a_validation_level:.2f}",
                                                            'A_Invalidation_Level': f"{a_invalidation_level:.2f}",
                                                            'A_Start_Time': a_start_time,
                                                            'A_End_Time': a_end_time,
                                                            'Entry_Price': f"{entry_price:.2f}",
                                                            'Exit_Price': f"{exit_price:.2f}",
                                                            'Closest_Anchor': closest_anchor['anchor'],
                                                            'Anchor_Price_Distance': f"{closest_anchor['price_distance']:.2f}",
                                                            'Anchor_Time_Distance_Minutes': f"{closest_anchor['time_distance_minutes']:.1f}",
                                                            'Entry_Retracement': f"{entry_retracement:.3f}",
                                                            'Exit_Retracement': f"{exit_retracement:.3f}",
                                                            'Entry_Time_Percent': f"{entry_time_percent:.3f}",
                                                            'Exit_Time_Percent': f"{exit_time_percent:.3f}",
                                                            'Spatial_Score': f"{spatial_score:.3f}",
                                                            'Temporal_Score': f"{temporal_score:.3f}",
                                                            'Combined_Score': f"{(spatial_score + temporal_score) / 2:.3f}"
                                                        })
                                                        
                                                        mapped_trade_ids.add(trade_id)
                                
                                # Display A-pattern spatial analysis results
                                if a_pattern_spatial_results:
                                    a_spatial_df = pd.DataFrame(a_pattern_spatial_results)
                                    st.dataframe(a_spatial_df)
                                    
                                    # Add A-pattern spatial data to ML dataset
                                    if 'ml_dataset' in locals():
                                        # Aggregate A-pattern metrics by trade
                                        trade_a_pattern_metrics = a_spatial_df.groupby(['Symbol', 'Timeframe', 'S_Pattern', 'Trade_Id']).agg({
                                            'Spatial_Score': 'mean',
                                            'Temporal_Score': 'mean',
                                            'Combined_Score': 'mean',
                                            'Entry_Retracement': 'mean',
                                            'Exit_Retracement': 'mean'
                                        }).reset_index()
                                        
                                        # Rename columns to avoid conflicts
                                        trade_a_pattern_metrics.columns = ['Symbol', 'Timeframe', 'S_Pattern', 'Trade_Id', 
                                                                          'A_Pattern_Spatial_Score', 'A_Pattern_Temporal_Score', 
                                                                          'A_Pattern_Combined_Score', 'A_Pattern_Entry_Retracement', 
                                                                          'A_Pattern_Exit_Retracement']
                                        
                                        # Merge with ML dataset
                                        ml_dataset = ml_dataset.merge(
                                            trade_a_pattern_metrics,
                                            on=['Symbol', 'Timeframe', 'S_Pattern'],
                                            how='left'
                                        )
                                else:
                                    st.info("No A-pattern spatial analysis data available.")
                                
                                # --- Unmapped Trades Analysis ---
                                st.markdown('<h3 style="color: gold;">🚫 Unmapped Trades (Outside All Drawings)</h3>', unsafe_allow_html=True)
                                
                                if 'trades' in locals() and not trades.empty:
                                    # Find trades that weren't mapped to any A-pattern
                                    all_trade_ids = set(trades['Id'].unique())
                                    unmapped_trade_ids = all_trade_ids - mapped_trade_ids
                                    
                                    if unmapped_trade_ids:
                                        unmapped_trades = trades[trades['Id'].isin(unmapped_trade_ids)]
                                        st.info(f"**Found {len(unmapped_trades)} trades outside all S-pattern drawings:**")
                                        st.dataframe(unmapped_trades[['Id', 'ContractName', 'EnteredAt', 'ExitedAt', 'EntryPrice', 'ExitPrice', 'Type', 'PnL', 'Size']])
                                        
                                        # Summary statistics for unmapped trades
                                        unmapped_pnl = unmapped_trades['PnL'].sum()
                                        unmapped_win_rate = (unmapped_trades['PnL'] > 0).mean() * 100
                                        st.write(f"**Unmapped Trades Summary:** Total PnL: ${unmapped_pnl:.2f}, Win Rate: {unmapped_win_rate:.1f}%")
                                    else:
                                        st.success("✅ All trades were successfully mapped to S-pattern drawings!")
                                else:
                                    st.info("No trade data available for unmapped analysis.")
                                
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
                                    
                                    # Add Fibonacci retracement data to comprehensive table
                                    if fib_results:
                                        fib_df_display = pd.DataFrame(fib_results)
                                        # Merge Fibonacci data with comprehensive table for A1 patterns only
                                        comprehensive_table = comprehensive_table.merge(
                                            fib_df_display[['Symbol', 'Timeframe', 'A1_Start_Time', 'Fib_Type', 'Fib_Level', 'Zone']], 
                                            left_on=['Symbol', 'Timeframe'],
                                            right_on=['Symbol', 'Timeframe'],
                                            how='left'
                                        )
                                        # Only show Fibonacci data for A1 patterns
                                        comprehensive_table['Fib_Type'] = comprehensive_table.apply(
                                            lambda row: row['Fib_Type'] if row['A_Pattern'] == 'A1' else 'N/A', axis=1
                                        )
                                        comprehensive_table['Fib_Level'] = comprehensive_table.apply(
                                            lambda row: row['Fib_Level'] if row['A_Pattern'] == 'A1' else 'N/A', axis=1
                                        )
                                        comprehensive_table['Zone'] = comprehensive_table.apply(
                                            lambda row: row['Zone'] if row['A_Pattern'] == 'A1' else 'N/A', axis=1
                                        )
                                    else:
                                        # Add empty Fibonacci columns if no data available
                                        comprehensive_table['Fib_Type'] = 'N/A'
                                        comprehensive_table['Fib_Level'] = 'N/A'
                                        comprehensive_table['Zone'] = 'N/A'
                                    
                                    # Fill NaN values
                                    comprehensive_table = comprehensive_table.fillna('N/A').infer_objects(copy=False)
                                    
                                    st.dataframe(comprehensive_table)
                                    
                                    # Export for ML training
                                                                            st.markdown('<h4 style="color: gold;">Export for Machine Learning</h4>', unsafe_allow_html=True)
                                    
                                    # --- MASSIVE BEAST ML DATASET CREATION ---
                                    st.markdown('<h3 style="color: red;">MASSIVE BEAST: Complete Analysis + Trade Dimensions ML Dataset</h3>', unsafe_allow_html=True)
                                    
                                    # Start with the comprehensive neural network data
                                    beast_ml_dataset = nn_df[['Symbol', 'Timeframe', 'A_Pattern', 'S_Pattern', 'Normalized_Change', 'Normalized_Bars', 
                                                             'Pattern_Strength', 'Volatility_Score', 'Pattern_Confidence', 'Z_Score',
                                                             'Percent_Change_Numeric', 'Bar_Count_Numeric', 'Start_Time_Chart', 'End_Time_Chart',
                                                             'Start_Price', 'End_Price', 'High_Price', 'Low_Price', 'Price_Range', 'Price_Change',
                                                             'Percent_Change', 'Raw_Percent_Change', 'Bar_Count', 'Bullish', 'Direction']].copy()
                                    
                                    # Add Fibonacci retracement data
                                    if fib_results:
                                        fib_df_ml = pd.DataFrame(fib_results)
                                        beast_ml_dataset = beast_ml_dataset.merge(
                                            fib_df_ml[['Symbol', 'Timeframe', 'A1_Start_Time', 'Fib_Type', 'Fib_Level', 'Zone']], 
                                            left_on=['Symbol', 'Timeframe', 'Start_Time_Chart'],
                                            right_on=['Symbol', 'Timeframe', 'A1_Start_Time'],
                                            how='left'
                                        )
                                        if 'A1_Start_Time' in beast_ml_dataset.columns:
                                            beast_ml_dataset = beast_ml_dataset.drop('A1_Start_Time', axis=1)
                                    else:
                                        beast_ml_dataset['Fib_Type'] = 'N/A'
                                        beast_ml_dataset['Fib_Level'] = 0.0
                                        beast_ml_dataset['Zone'] = 'N/A'
                                    
                                    # Add A-pattern spatial-temporal analysis data
                                    if a_pattern_spatial_results:
                                        spatial_df = pd.DataFrame(a_pattern_spatial_results)
                                        
                                        # Convert string scores to numeric before aggregation
                                        numeric_columns = ['Spatial_Score', 'Temporal_Score', 'Combined_Score', 'Entry_Retracement', 'Exit_Retracement', 'Trade_PnL']
                                        for col in numeric_columns:
                                            if col in spatial_df.columns:
                                                # Check if column is already numeric
                                                if spatial_df[col].dtype == 'object':
                                                    # Handle string values
                                                    spatial_df[col] = pd.to_numeric(spatial_df[col].astype(str).str.replace('N/A', '0'), errors='coerce')
                                                else:
                                                    # Already numeric, just ensure it's float
                                                    spatial_df[col] = pd.to_numeric(spatial_df[col], errors='coerce')
                                        
                                        # Aggregate spatial metrics by pattern
                                        spatial_agg = spatial_df.groupby(['Symbol', 'Timeframe', 'S_Pattern', 'Analyzed_A_Pattern']).agg({
                                            'Spatial_Score': ['mean', 'std', 'min', 'max'],
                                            'Temporal_Score': ['mean', 'std', 'min', 'max'],
                                            'Combined_Score': ['mean', 'std', 'min', 'max'],
                                            'Entry_Retracement': ['mean', 'std'],
                                            'Exit_Retracement': ['mean', 'std'],
                                            'Trade_PnL': ['mean', 'sum', 'count'],
                                            'Trade_Type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                                        }).reset_index()
                                        
                                        # Flatten column names
                                        spatial_agg.columns = ['Symbol', 'Timeframe', 'S_Pattern', 'Analyzed_A_Pattern'] + \
                                                             [f'Spatial_{col[1]}' for col in spatial_agg.columns[4:8]] + \
                                                             [f'Temporal_{col[1]}' for col in spatial_agg.columns[8:12]] + \
                                                             [f'Combined_{col[1]}' for col in spatial_agg.columns[12:16]] + \
                                                             [f'Entry_Retracement_{col[1]}' for col in spatial_agg.columns[16:18]] + \
                                                             [f'Exit_Retracement_{col[1]}' for col in spatial_agg.columns[18:20]] + \
                                                             [f'Trade_PnL_{col[1]}' for col in spatial_agg.columns[20:23]] + \
                                                             ['Trade_Type_Mode']
                                        
                                        # Merge spatial data
                                        beast_ml_dataset = beast_ml_dataset.merge(
                                            spatial_agg,
                                            left_on=['Symbol', 'Timeframe', 'S_Pattern', 'A_Pattern'],
                                            right_on=['Symbol', 'Timeframe', 'S_Pattern', 'Analyzed_A_Pattern'],
                                            how='left'
                                        )
                                        if 'Analyzed_A_Pattern' in beast_ml_dataset.columns:
                                            beast_ml_dataset = beast_ml_dataset.drop('Analyzed_A_Pattern', axis=1)
                                    
                                    # Add trade performance metrics by pattern
                                    if 'trades' in locals() and not trades.empty:
                                        # Convert trade times to Unix timestamps for matching
                                        trades['EnteredAt_unix'] = pd.to_datetime(trades['EnteredAt']).astype(np.int64) // 10**9
                                        trades['ExitedAt_unix'] = pd.to_datetime(trades['ExitedAt']).astype(np.int64) // 10**9
                                        
                                        # Find trades that occurred during each pattern's timeframe
                                        trade_pattern_matches = []
                                        
                                        for _, pattern in beast_ml_dataset.iterrows():
                                            pattern_start = pd.to_numeric(pattern['Start_Time_Chart'], errors='coerce')
                                            pattern_end = pd.to_numeric(pattern['End_Time_Chart'], errors='coerce')
                                            
                                            if pd.notna(pattern_start) and pd.notna(pattern_end):
                                                # Find trades in this symbol during this pattern's timeframe
                                                symbol_trades = trades[trades['ContractName'] == pattern['Symbol']]
                                                relevant_trades = symbol_trades[
                                                    (symbol_trades['EnteredAt_unix'] >= pattern_start) &
                                                    (symbol_trades['ExitedAt_unix'] <= pattern_end)
                                                ]
                                                
                                                if not relevant_trades.empty:
                                                    # Calculate trade metrics for this pattern
                                                    trade_metrics = {
                                                        'Symbol': pattern['Symbol'],
                                                        'Timeframe': pattern['Timeframe'],
                                                        'S_Pattern': pattern['S_Pattern'],
                                                        'A_Pattern': pattern['A_Pattern'],
                                                        'Pattern_Trades_Count': len(relevant_trades),
                                                        'Pattern_Total_PnL': relevant_trades['PnL'].sum(),
                                                        'Pattern_Avg_PnL': relevant_trades['PnL'].mean(),
                                                        'Pattern_Win_Rate': (relevant_trades['PnL'] > 0).mean() * 100,
                                                        'Pattern_Best_Trade': relevant_trades['PnL'].max(),
                                                        'Pattern_Worst_Trade': relevant_trades['PnL'].min(),
                                                        'Pattern_Profit_Factor': relevant_trades[relevant_trades['PnL'] > 0]['PnL'].sum() / abs(relevant_trades[relevant_trades['PnL'] < 0]['PnL'].sum()) if relevant_trades[relevant_trades['PnL'] < 0]['PnL'].sum() != 0 else float('nan'),
                                                        'Pattern_Avg_Win': relevant_trades[relevant_trades['PnL'] > 0]['PnL'].mean() if len(relevant_trades[relevant_trades['PnL'] > 0]) > 0 else 0,
                                                        'Pattern_Avg_Loss': relevant_trades[relevant_trades['PnL'] < 0]['PnL'].mean() if len(relevant_trades[relevant_trades['PnL'] < 0]) > 0 else 0,
                                                        'Pattern_Win_Loss_Ratio': abs(relevant_trades[relevant_trades['PnL'] > 0]['PnL'].mean() / relevant_trades[relevant_trades['PnL'] < 0]['PnL'].mean()) if (len(relevant_trades[relevant_trades['PnL'] > 0]) > 0 and len(relevant_trades[relevant_trades['PnL'] < 0]) > 0 and relevant_trades[relevant_trades['PnL'] < 0]['PnL'].mean() != 0) else float('nan'),
                                                        'Pattern_Total_Volume': relevant_trades['Size'].sum() if 'Size' in relevant_trades.columns else 0,
                                                        'Pattern_Avg_Volume': relevant_trades['Size'].mean() if 'Size' in relevant_trades.columns else 0,
                                                        'Pattern_Long_Trades': len(relevant_trades[relevant_trades['Type'] == 'Long']) if 'Type' in relevant_trades.columns else 0,
                                                        'Pattern_Short_Trades': len(relevant_trades[relevant_trades['Type'] == 'Short']) if 'Type' in relevant_trades.columns else 0,
                                                        'Pattern_Long_Win_Rate': (relevant_trades[(relevant_trades['Type'] == 'Long') & (relevant_trades['PnL'] > 0)]['PnL'].count() / len(relevant_trades[relevant_trades['Type'] == 'Long'])) * 100 if 'Type' in relevant_trades.columns and len(relevant_trades[relevant_trades['Type'] == 'Long']) > 0 else 0,
                                                        'Pattern_Short_Win_Rate': (relevant_trades[(relevant_trades['Type'] == 'Short') & (relevant_trades['PnL'] > 0)]['PnL'].count() / len(relevant_trades[relevant_trades['Type'] == 'Short'])) * 100 if 'Type' in relevant_trades.columns and len(relevant_trades[relevant_trades['Type'] == 'Short']) > 0 else 0
                                                    }
                                                    trade_pattern_matches.append(trade_metrics)
                                        
                                        if trade_pattern_matches:
                                            trade_pattern_df = pd.DataFrame(trade_pattern_matches)
                                            beast_ml_dataset = beast_ml_dataset.merge(
                                                trade_pattern_df,
                                                on=['Symbol', 'Timeframe', 'S_Pattern', 'A_Pattern'],
                                                how='left'
                                            )
                                    
                                    # Add pattern sequence and succession features
                                    beast_ml_dataset['Pattern_Sequence'] = beast_ml_dataset.groupby(['Symbol', 'Timeframe', 'S_Pattern']).cumcount() + 1
                                    beast_ml_dataset['Total_Patterns_In_S_Group'] = beast_ml_dataset.groupby(['Symbol', 'Timeframe', 'S_Pattern'])['A_Pattern'].transform('count')
                                    beast_ml_dataset['Pattern_Position_In_Sequence'] = beast_ml_dataset['Pattern_Sequence'] / beast_ml_dataset['Total_Patterns_In_S_Group']
                                    
                                    # Add pattern type classifications
                                    beast_ml_dataset['Is_Stage1_Pattern'] = beast_ml_dataset['A_Pattern'].isin(['A1', 'A12', 'A13']).astype(int)
                                    beast_ml_dataset['Is_Stage2_Pattern'] = beast_ml_dataset['A_Pattern'].isin(['A2', 'A14', 'A15']).astype(int)
                                    beast_ml_dataset['Is_Support_Resistance'] = beast_ml_dataset['A_Pattern'].isin(['A5', 'A6']).astype(int)
                                    beast_ml_dataset['Is_Demand_Supply'] = beast_ml_dataset['A_Pattern'].isin(['A3', 'A4']).astype(int)
                                    beast_ml_dataset['Is_Fibonacci_Pattern'] = beast_ml_dataset['A_Pattern'].isin(['A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']).astype(int)
                                    
                                    # Add market structure features
                                    beast_ml_dataset['Is_Bullish_Pattern'] = (beast_ml_dataset['Bullish'] == True).astype(int)
                                    beast_ml_dataset['Is_Bearish_Pattern'] = (beast_ml_dataset['Bullish'] == False).astype(int)
                                    beast_ml_dataset['Pattern_Direction_Up'] = (beast_ml_dataset['Direction'] == 'up').astype(int)
                                    beast_ml_dataset['Pattern_Direction_Down'] = (beast_ml_dataset['Direction'] == 'down').astype(int)
                                    
                                    # Add price action features
                                    beast_ml_dataset['Price_Change_Positive'] = (beast_ml_dataset['Price_Change'] > 0).astype(int)
                                    beast_ml_dataset['Percent_Change_Positive'] = (beast_ml_dataset['Percent_Change_Numeric'] > 0).astype(int)
                                    beast_ml_dataset['High_Low_Ratio'] = beast_ml_dataset['High_Price'] / beast_ml_dataset['Low_Price']
                                    beast_ml_dataset['Price_Range_Percent'] = (beast_ml_dataset['Price_Range'] / beast_ml_dataset['Start_Price']) * 100
                                    
                                    # Add timeframe features
                                    timeframe_mapping = {
                                        '1M': 1, '5M': 5, '15M': 15, '30M': 30,
                                        '1H': 60, '4H': 240, '1D': 1440
                                    }
                                    beast_ml_dataset['Timeframe_Minutes'] = beast_ml_dataset['Timeframe'].map(timeframe_mapping)
                                    beast_ml_dataset['Is_Higher_Timeframe'] = (beast_ml_dataset['Timeframe_Minutes'] >= 60).astype(int)
                                    beast_ml_dataset['Is_Lower_Timeframe'] = (beast_ml_dataset['Timeframe_Minutes'] < 60).astype(int)
                                    
                                    # Add S-group consistency features
                                    # Ensure numeric columns for aggregation
                                    numeric_cols_for_agg = ['Pattern_Strength', 'Pattern_Confidence', 'Volatility_Score', 'Z_Score', 'Percent_Change_Numeric', 'Bar_Count_Numeric']
                                    for col in numeric_cols_for_agg:
                                        if col in beast_ml_dataset.columns:
                                            # Handle mixed data types safely
                                            if beast_ml_dataset[col].dtype == 'object':
                                                # Convert string values to numeric
                                                beast_ml_dataset[col] = pd.to_numeric(beast_ml_dataset[col].astype(str).str.replace('N/A', '0'), errors='coerce')
                                            else:
                                                # Already numeric, just ensure it's float
                                                beast_ml_dataset[col] = pd.to_numeric(beast_ml_dataset[col], errors='coerce')
                                    
                                    s_group_metrics = beast_ml_dataset.groupby(['Symbol', 'Timeframe', 'S_Pattern']).agg({
                                        'Pattern_Strength': ['mean', 'std'],
                                        'Pattern_Confidence': ['mean', 'std'],
                                        'Volatility_Score': ['mean', 'std'],
                                        'Z_Score': ['mean', 'std'],
                                        'Percent_Change_Numeric': ['mean', 'std'],
                                        'Bar_Count_Numeric': ['mean', 'std']
                                    }).reset_index()
                                    
                                    # Flatten column names
                                    s_group_metrics.columns = ['Symbol', 'Timeframe', 'S_Pattern'] + \
                                                             [f'S_Group_{col[0]}_{col[1]}' for col in s_group_metrics.columns[3:]]
                                    
                                    beast_ml_dataset = beast_ml_dataset.merge(
                                        s_group_metrics,
                                        on=['Symbol', 'Timeframe', 'S_Pattern'],
                                        how='left'
                                    )
                                    
                                    # Add pattern reliability features
                                    # Ensure numeric values for calculations with safe conversion
                                    def safe_numeric_convert(series):
                                        if series.dtype == 'object':
                                            return pd.to_numeric(series.astype(str).str.replace('N/A', '0'), errors='coerce')
                                        else:
                                            return pd.to_numeric(series, errors='coerce')
                                    
                                    beast_ml_dataset['Pattern_Reliability_Score'] = safe_numeric_convert(beast_ml_dataset['Pattern_Confidence']) * (1 - safe_numeric_convert(beast_ml_dataset['Volatility_Score']))
                                    beast_ml_dataset['Pattern_Performance_Score'] = safe_numeric_convert(beast_ml_dataset['Normalized_Change']) * safe_numeric_convert(beast_ml_dataset['Pattern_Strength'])
                                    beast_ml_dataset['Pattern_Consistency_Score'] = 1 / (1 + safe_numeric_convert(beast_ml_dataset['Z_Score']).abs())
                                    
                                    # Add Fibonacci zone features
                                    if 'Zone' in beast_ml_dataset.columns:
                                        beast_ml_dataset['Is_Bullish_Zone'] = (beast_ml_dataset['Zone'].str.contains('Bullish|Green', na=False)).astype(int)
                                        beast_ml_dataset['Is_Red_Zone'] = (beast_ml_dataset['Zone'].str.contains('Red', na=False)).astype(int)
                                        beast_ml_dataset['Is_Grey_Zone'] = (beast_ml_dataset['Zone'].str.contains('Grey|Neutral', na=False)).astype(int)
                                        beast_ml_dataset['Is_Upper_Half'] = (beast_ml_dataset['Zone'].str.contains('Upper', na=False)).astype(int)
                                        beast_ml_dataset['Is_Lower_Half'] = (beast_ml_dataset['Zone'].str.contains('Lower', na=False)).astype(int)
                                    
                                    # Add temporal features
                                    beast_ml_dataset['Pattern_Duration_Minutes'] = (pd.to_numeric(beast_ml_dataset['End_Time_Chart'], errors='coerce') - 
                                                                                   pd.to_numeric(beast_ml_dataset['Start_Time_Chart'], errors='coerce')) / 60
                                    beast_ml_dataset['Bars_Per_Minute'] = safe_numeric_convert(beast_ml_dataset['Bar_Count_Numeric']) / beast_ml_dataset['Pattern_Duration_Minutes']
                                    beast_ml_dataset['Price_Change_Per_Bar'] = safe_numeric_convert(beast_ml_dataset['Price_Change']) / safe_numeric_convert(beast_ml_dataset['Bar_Count_Numeric'])
                                    
                                    # Add market efficiency features
                                    beast_ml_dataset['Market_Efficiency_Ratio'] = safe_numeric_convert(beast_ml_dataset['Price_Change']) / safe_numeric_convert(beast_ml_dataset['Price_Range'])
                                    beast_ml_dataset['Pattern_Intensity'] = safe_numeric_convert(beast_ml_dataset['Percent_Change_Numeric']) / safe_numeric_convert(beast_ml_dataset['Bar_Count_Numeric'])
                                    
                                    # Fill NaN values with appropriate defaults
                                    numeric_columns = beast_ml_dataset.select_dtypes(include=[np.number]).columns
                                    beast_ml_dataset[numeric_columns] = beast_ml_dataset[numeric_columns].fillna(0)
                                    
                                    categorical_columns = beast_ml_dataset.select_dtypes(include=['object']).columns
                                    beast_ml_dataset[categorical_columns] = beast_ml_dataset[categorical_columns].fillna('N/A')
                                    
                                    # Remove duplicates
                                    beast_ml_dataset = beast_ml_dataset.drop_duplicates()
                                    
                                    if not beast_ml_dataset.empty:
                                        # Create one-hot encoding for categorical variables
                                        categorical_cols = ['Symbol', 'Timeframe', 'A_Pattern', 'S_Pattern', 'Fib_Type', 'Zone', 'Direction', 'Bullish']
                                        existing_categorical_cols = [col for col in categorical_cols if col in beast_ml_dataset.columns]
                                        
                                        # Create dynamic prefixes
                                        prefix_mapping = {
                                            'Symbol': 'Symbol',
                                            'Timeframe': 'Timeframe', 
                                            'A_Pattern': 'Pattern',
                                            'S_Pattern': 'S_Pattern',
                                            'Fib_Type': 'Fib',
                                            'Zone': 'Zone',
                                            'Direction': 'Direction',
                                            'Bullish': 'Bullish'
                                        }
                                        dynamic_prefixes = [prefix_mapping[col] for col in existing_categorical_cols]
                                        
                                        beast_ml_encoded = pd.get_dummies(beast_ml_dataset, columns=existing_categorical_cols, 
                                                                         prefix=dynamic_prefixes)
                                        
                                        # Create download button for the beast dataset
                                        csv_data = beast_ml_encoded.to_csv(index=False)
                                        st.download_button(
                                            label="Download MASSIVE BEAST ML Dataset (CSV)",
                                            data=csv_data,
                                            file_name="massive_beast_ml_dataset.csv",
                                            mime="text/csv"
                                        )
                                        
                                        st.success(f"**MASSIVE BEAST ML Dataset Created!**")
                                        st.info(f"**Dataset Summary:** {len(beast_ml_encoded)} patterns, {len(beast_ml_encoded.columns)} features")
                                        
                                        # Display feature categories
                                        st.markdown("**Complete Feature Categories:**")
                                        st.markdown("""
                                        | Category | Features | Purpose |
                                        |----------|----------|---------|
                                        | **Core Pattern Metrics** | Normalized_Change, Normalized_Bars, Pattern_Strength, Volatility_Score, Pattern_Confidence, Z_Score | Pattern performance and reliability |
                                        | **Price Action** | Start_Price, End_Price, High_Price, Low_Price, Price_Range, Price_Change, Percent_Change | Price movement analysis |
                                        | **Fibonacci Analysis** | Fib_Type, Fib_Level, Zone, Is_Bullish_Zone, Is_Red_Zone, Is_Grey_Zone | Fibonacci retracement positioning |
                                        | **Spatial-Temporal** | Spatial_Score, Temporal_Score, Combined_Score, Entry_Retracement, Exit_Retracement | Trade positioning analysis |
                                        | **Trade Performance** | Pattern_Trades_Count, Pattern_Total_PnL, Pattern_Win_Rate, Pattern_Profit_Factor, Pattern_Avg_Win, Pattern_Avg_Loss | Historical trade performance |
                                        | **Pattern Classification** | Is_Stage1_Pattern, Is_Stage2_Pattern, Is_Support_Resistance, Is_Demand_Supply, Is_Fibonacci_Pattern | Pattern type identification |
                                        | **Market Structure** | Is_Bullish_Pattern, Is_Bearish_Pattern, Pattern_Direction_Up, Pattern_Direction_Down | Market direction analysis |
                                        | **Timeframe Analysis** | Timeframe_Minutes, Is_Higher_Timeframe, Is_Lower_Timeframe | Multi-timeframe context |
                                        | **S-Group Metrics** | S_Group_Pattern_Strength_mean, S_Group_Pattern_Confidence_mean, S_Group_Volatility_Score_mean | Group consistency analysis |
                                        | **Reliability Scores** | Pattern_Reliability_Score, Pattern_Performance_Score, Pattern_Consistency_Score | Pattern predictability |
                                        | **Temporal Features** | Pattern_Duration_Minutes, Bars_Per_Minute, Price_Change_Per_Bar | Time-based analysis |
                                        | **Market Efficiency** | Market_Efficiency_Ratio, Pattern_Intensity | Market behavior metrics |
                                        | **Categorical Features** | Symbol, Timeframe, Pattern types (one-hot encoded) | Classification variables |
                                        """)
                                        
                                        # Show sample of the beast dataset
                                        st.write("**Sample MASSIVE BEAST ML Data (first 5 rows):**")
                                        sample_beast = beast_ml_encoded.head().reset_index(drop=True)
                                        st.dataframe(sample_beast)
                                        
                                        # Show feature importance summary
                                        st.markdown("**Key ML Training Features:**")
                                        st.markdown("""
                                        - **Target Variables:** Pattern_Total_PnL, Pattern_Win_Rate, Pattern_Profit_Factor
                                        - **Primary Predictors:** Pattern_Strength, Pattern_Confidence, Spatial_Score, Temporal_Score
                                        - **Context Features:** Timeframe_Minutes, S_Group metrics, Fibonacci positioning
                                        - **Classification Features:** Pattern types, market direction, zone positioning
                                        """)
                                        
                                        # Export summary statistics
                                        st.markdown("**Dataset Statistics:**")
                                        numeric_summary = beast_ml_encoded.describe()
                                        st.dataframe(numeric_summary)
                                        
                                        # --- MACHINE LEARNING TRAINING SECTION ---
                                        st.markdown('<h3 style="color: purple;">🤖 Machine Learning Training</h3>', unsafe_allow_html=True)
                                        
                                        # Store the beast dataset in session state for ML training
                                        st.session_state['beast_ml_dataset'] = beast_ml_encoded
                                        st.session_state['beast_ml_ready'] = True
                                        
                                        st.success("✅ MASSIVE BEAST ML Dataset ready for training!")
                                        
                                        # Add prominent ML training button
                                        st.markdown("---")
                                        st.markdown('<h3 style="color: purple;">Ready to Train ML Models?</h3>', unsafe_allow_html=True)
                                        
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            if st.button("START MACHINE LEARNING TRAINING", type="primary", use_container_width=True):
                                                st.rerun()  # Refresh the page to show ML section
                                        
                                        st.info("**Scroll down to the bottom of the page to see the ML Training section with all options!**")
                                        
                                    else:
                                        st.warning("No complete data available for MASSIVE BEAST ML export (missing values)")
                                        st.session_state['beast_ml_ready'] = False
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

# --- MACHINE LEARNING TRAINING SECTION ---
st.markdown("---")
st.markdown('<h2 style="color: purple; text-align: center;">MACHINE LEARNING TRAINING</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 18px;">Train predictive models on your fractal trading patterns</p>', unsafe_allow_html=True)
st.markdown("---")

# Check if ML dataset is ready
if 'beast_ml_ready' in st.session_state and st.session_state['beast_ml_ready']:
    st.success("ML Dataset Ready!")
    
    # ML Training Options
    st.markdown("**Select ML Training Options:**")
    
    col1, col2 = st.columns(2)
    with col1:
        target_variable = st.selectbox(
            "Target Variable",
            options=['Pattern_Total_PnL', 'Pattern_Win_Rate', 'Pattern_Profit_Factor'],
            help="Select the variable to predict"
        )
        
        model_type = st.selectbox(
            "Model Type",
            options=['Random Forest', 'XGBoost', 'Neural Network', 'Linear Regression', 'All Models'],
            help="Select the machine learning model to train"
        )
    
    with col2:
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data to use for testing"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Random seed for reproducible results"
        )
    
    # Feature Selection
    st.markdown("**Feature Selection:**")
    
    if 'beast_ml_dataset' in st.session_state:
        dataset = st.session_state['beast_ml_dataset']
        
        # Get numeric columns for features
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target variables from feature options
        target_vars = ['Pattern_Total_PnL', 'Pattern_Win_Rate', 'Pattern_Profit_Factor', 'Pattern_Avg_PnL', 'Pattern_Best_Trade', 'Pattern_Worst_Trade']
        feature_options = [col for col in numeric_cols if col not in target_vars]
        
        selected_features = st.multiselect(
            "Select Features",
            options=feature_options,
            default=feature_options[:20],  # Default to first 20 features
            help="Select features to use for training"
        )
        
        # Show dataset info
        st.info(f"**Dataset Info:** {len(dataset)} samples, {len(selected_features)} features selected")
        
        # ML Training Button
        if st.button("Train Machine Learning Models", type="primary"):
            with st.spinner("Training ML models..."):
                try:
                    # Import ML libraries
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    import xgboost as xgb
                    
                    # Prepare data
                    X = dataset[selected_features].fillna(0)
                    y = dataset[target_variable].fillna(0)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=random_state
                    )
                    
                    st.success(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
                    
                    # Train models
                    models = {}
                    results = {}
                    
                    if model_type in ['Random Forest', 'All Models']:
                        with st.spinner("Training Random Forest..."):
                            rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                            rf_model.fit(X_train, y_train)
                            rf_pred = rf_model.predict(X_test)
                            models['Random Forest'] = rf_model
                            results['Random Forest'] = {
                                'R2': r2_score(y_test, rf_pred),
                                'MSE': mean_squared_error(y_test, rf_pred),
                                'MAE': mean_absolute_error(y_test, rf_pred)
                            }
                    
                    if model_type in ['XGBoost', 'All Models']:
                        with st.spinner("Training XGBoost..."):
                            xgb_model = xgb.XGBRegressor(random_state=random_state)
                            xgb_model.fit(X_train, y_train)
                            xgb_pred = xgb_model.predict(X_test)
                            models['XGBoost'] = xgb_model
                            results['XGBoost'] = {
                                'R2': r2_score(y_test, xgb_pred),
                                'MSE': mean_squared_error(y_test, xgb_pred),
                                'MAE': mean_absolute_error(y_test, xgb_pred)
                            }
                    
                    if model_type in ['Linear Regression', 'All Models']:
                        with st.spinner("Training Linear Regression..."):
                            lr_model = LinearRegression()
                            lr_model.fit(X_train, y_train)
                            lr_pred = lr_model.predict(X_test)
                            models['Linear Regression'] = lr_model
                            results['Linear Regression'] = {
                                'R2': r2_score(y_test, lr_pred),
                                'MSE': mean_squared_error(y_test, lr_pred),
                                'MAE': mean_absolute_error(y_test, lr_pred)
                            }
                    
                    # Display results
                    st.markdown("**Model Performance Results:**")
                    
                    # Create results table
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df)
                    
                    # Find best model
                    best_model = results_df['R2'].idxmax()
                    st.success(f"Best Model: {best_model} (R² = {results_df.loc[best_model, 'R2']:.4f})")
                    
                    # Feature importance for tree-based models
                    if best_model in ['Random Forest', 'XGBoost']:
                        st.markdown("**Feature Importance (Best Model):**")
                        best_model_obj = models[best_model]
                        
                        if hasattr(best_model_obj, 'feature_importances_'):
                            feature_importance = pd.DataFrame({
                                'Feature': selected_features,
                                'Importance': best_model_obj.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            st.dataframe(feature_importance.head(10))
                            
                            # Feature importance chart
                            import plotly.express as px
                            fig = px.bar(
                                feature_importance.head(15),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f'Top 15 Feature Importance - {best_model}'
                            )
                            st.plotly_chart(fig)
                    
                    # Store results in session state
                    st.session_state['ml_results'] = results
                    st.session_state['ml_models'] = models
                    st.session_state['ml_best_model'] = best_model
                    
                    # Model download option
                    st.markdown("**Model Export:**")
                    if st.button("Download Best Model"):
                        import pickle
                        best_model_obj = models[best_model]
                        model_bytes = pickle.dumps(best_model_obj)
                        st.download_button(
                            label=f"Download {best_model} Model",
                            data=model_bytes,
                            file_name=f"best_model_{best_model.lower().replace(' ', '_')}.pkl",
                            mime="application/octet-stream"
                        )
                    
                except Exception as e:
                    st.error(f"ML Training Error: {str(e)}")
                    st.info("Make sure you have scikit-learn and xgboost installed: `pip install scikit-learn xgboost`")
    
    else:
        st.warning("No ML dataset available. Please run the analysis first.")
        
else:
    st.info("Please upload trade data and run analysis to prepare ML dataset first.")
    st.markdown("""
    **Steps to train ML models:**
    1. Upload Trade CSV file
    2. Upload Analysis JSON file (or use default)
    3. Wait for MASSIVE BEAST ML Dataset to be created
    4. Return here to train ML models
    """) 