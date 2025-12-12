"""
Main Streamlit Application
End-to-end demonstration of NL → DSL → AST → Code → Execution pipeline
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import our modules
from nl_parser import NLParser
from dsl_parser import DSLParser
from code_generator import CodeGenerator
from backtest import BacktestSimulator
import indicators


# Page configuration
st.set_page_config(
    page_title="Trading Strategy DSL Demo",
    page_icon="📈",
    layout="wide"
)


def generate_sample_data(days: int = 252, start_price: float = 100.0) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data
    
    Args:
        days: Number of days to generate
        start_price: Starting price
    
    Returns:
        pd.DataFrame: OHLCV data
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price data with trend and volatility
    returns = np.random.randn(days) * 0.02  # 2% daily volatility
    trend = np.linspace(0, 0.3, days)  # 30% upward trend over period
    price = start_price * np.exp((returns + trend / days).cumsum())
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(days) * 0.005),
        'high': price * (1 + abs(np.random.randn(days)) * 0.01),
        'low': price * (1 - abs(np.random.randn(days)) * 0.01),
        'close': price,
        'volume': np.random.randint(800000, 1500000, days)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def main():
    """Main Streamlit application"""
    
    st.title("📈 Trading Strategy DSL Demo")
    st.markdown("""
    This application demonstrates an end-to-end pipeline for converting natural language trading rules 
    into executable code: **NL → DSL → AST → Code → Execution**
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data generation options
    st.sidebar.subheader("Data Settings")
    num_days = st.sidebar.slider("Historical Days", 50, 500, 252)
    start_price = st.sidebar.number_input("Starting Price", 50.0, 200.0, 100.0)
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000.0, 100000.0, 10000.0)
    
    # Generate or upload data
    data_source = st.sidebar.radio("Data Source", ["Generate Synthetic", "Upload CSV"])
    
    if data_source == "Generate Synthetic":
        df = generate_sample_data(num_days, start_price)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        else:
            st.warning("Please upload a CSV file or switch to synthetic data")
            df = generate_sample_data(num_days, start_price)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Natural Language Input",
        "🔤 DSL Output",
        "🌳 AST Visualization",
        "💻 Generated Code",
        "📊 Backtest Results",
        "📋 Final Report",
        "📚 Documentation"
    ])
    
    # Tab 1: Natural Language Input
    with tab1:
        st.header("Step 1: Natural Language Input")
        st.markdown("Enter your trading strategy in plain English:")
        
        # Example strategies
        examples = {
            "Example 1: SMA + Volume": "Buy when the close price is above the 20-day moving average and volume is above 1 million. Exit when RSI(14) is below 30.",
            "Example 2: Price Cross": "Enter when price crosses above yesterday's high. Exit when price crosses below yesterday's low.",
            "Example 3: RSI Strategy": "Buy when RSI(14) is below 30. Sell when RSI(14) is above 70.",
            "Example 4: Multi-condition": "Enter when close is above SMA(close,50) and volume is above 1000000 and RSI(close,14) is above 50. Exit when close is below SMA(close,20).",
            "Custom": "Write your own..."
        }
        
        selected_example = st.selectbox("Choose an example or write custom:", list(examples.keys()))
        
        if selected_example == "Custom":
            nl_input = st.text_area(
                "Natural Language Strategy",
                height=150,
                placeholder="e.g., Buy when close is above 20-day moving average..."
            )
        else:
            nl_input = st.text_area(
                "Natural Language Strategy",
                value=examples[selected_example],
                height=150
            )
        
        # Parse button
        if st.button("🚀 Process Strategy", type="primary"):
            if nl_input.strip():
                # Store in session state
                st.session_state['nl_input'] = nl_input
                st.session_state['process_clicked'] = True
                st.success("✅ Strategy processed! Check other tabs for results.")
            else:
                st.error("Please enter a strategy description.")
    
    # Check if we have input to process
    if 'nl_input' in st.session_state and st.session_state.get('process_clicked'):
        nl_input = st.session_state['nl_input']
        
        try:
            # Step 1: Parse NL to structured JSON
            nl_parser = NLParser()
            structured_json = nl_parser.parse(nl_input)
            
            # Step 2: Convert to DSL
            dsl_text = convert_json_to_dsl(structured_json)
            
            # Step 3: Parse DSL to AST
            dsl_parser = DSLParser()
            ast = dsl_parser.parse(dsl_text)
            
            # Step 4: Generate Python code
            code_gen = CodeGenerator()
            generated_code = code_gen.generate(ast)
            
            # Step 5: Execute and backtest
            signals = code_gen.generate_and_execute(ast, df)
            simulator = BacktestSimulator(initial_capital=initial_capital)
            results = simulator.run(df, signals)
            
            # Store results in session state
            st.session_state['structured_json'] = structured_json
            st.session_state['dsl_text'] = dsl_text
            st.session_state['ast'] = ast
            st.session_state['generated_code'] = generated_code
            st.session_state['signals'] = signals
            st.session_state['results'] = results
            st.session_state['df'] = df
            
        except Exception as e:
            st.error(f"Error processing strategy: {str(e)}")
            st.exception(e)
    
    # Tab 2: DSL Output
    with tab2:
        st.header("Step 2: DSL Representation")
        st.markdown("The natural language is converted to our Domain-Specific Language (DSL):")
        
        if 'structured_json' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Structured JSON")
                st.json(st.session_state['structured_json'])
            
            with col2:
                st.subheader("DSL Text")
                st.code(st.session_state['dsl_text'], language="text")
        else:
            st.info("Process a strategy in the 'Natural Language Input' tab first.")
    
    # Tab 3: AST Visualization
    with tab3:
        st.header("Step 3: Abstract Syntax Tree (AST)")
        st.markdown("The DSL is parsed into an Abstract Syntax Tree:")
        
        if 'ast' in st.session_state:
            st.json(st.session_state['ast'])
        else:
            st.info("Process a strategy in the 'Natural Language Input' tab first.")
    
    # Tab 4: Generated Code
    with tab4:
        st.header("Step 4: Generated Python Code")
        st.markdown("The AST is converted into executable Python code:")
        
        if 'generated_code' in st.session_state:
            st.code(st.session_state['generated_code'], language="python")
        else:
            st.info("Process a strategy in the 'Natural Language Input' tab first.")
    
    # Tab 5: Backtest Results
    with tab5:
        st.header("Step 5: Backtest Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            df = st.session_state['df']
            signals = st.session_state['signals']
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            metrics = results['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:+.2f}%")
                st.metric("Final Capital", f"${metrics['final_capital']:,.2f}")
            with col2:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Number of Trades", metrics['num_trades'])
                st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            with col4:
                st.metric("Avg Win", f"${metrics['avg_win']:+.2f}")
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            
            # Equity curve
            st.subheader("💹 Equity Curve")
            # Equity curve includes initial capital, so it has one extra value
            equity_curve = results['equity_curve']
            if len(equity_curve) == len(df) + 1:
                equity_curve = equity_curve[1:]  # Skip initial capital
            equity_df = pd.DataFrame({
                'Equity': equity_curve
            }, index=df.index)
            st.line_chart(equity_df)
            
            # Price chart with signals
            st.subheader("📈 Price Chart with Signals")
            chart_df = df[['close']].copy()
            chart_df['Entry'] = df['close'].where(signals['entry'], np.nan)
            chart_df['Exit'] = df['close'].where(signals['exit'], np.nan)
            st.line_chart(chart_df)
            
            # Trade log
            st.subheader("📋 Trade Log")
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed with this strategy.")
            
            # Download results
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            with col1:
                trades_csv = pd.DataFrame(results['trades']).to_csv(index=False)
                st.download_button(
                    "Download Trade Log (CSV)",
                    trades_csv,
                    "trades.csv",
                    "text/csv"
                )
            with col2:
                results_json = json.dumps(results['metrics'], indent=2)
                st.download_button(
                    "Download Metrics (JSON)",
                    results_json,
                    "metrics.json",
                    "application/json"
                )
        else:
            st.info("Process a strategy in the 'Natural Language Input' tab first.")
    
    # Tab 6: Final Report (Part 6 format)
    with tab6:
        st.header("Part 6 — End-to-End Demonstration - Final Report")
        
        if 'nl_input' in st.session_state and 'results' in st.session_state:
            nl_input = st.session_state['nl_input']
            dsl_text = st.session_state['dsl_text']
            ast = st.session_state['ast']
            results = st.session_state['results']
            
            st.markdown("---")
            
            # Natural Language Input
            st.subheader("Natural Language Input:")
            st.code(f'"{nl_input}"', language=None)
            
            # Generated DSL
            st.subheader("Generated DSL:")
            st.code(dsl_text, language=None)
            
            # Parsed AST (abbreviated)
            st.subheader("Parsed AST:")
            ast_preview = json.dumps(ast, indent=2)[:500] + "\n  ...\n}"
            st.code(ast_preview, language="json")
            
            # Backtest Result
            st.subheader("Backtest Result:")
            metrics = results['metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:+.1f}%")
            with col2:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
            with col3:
                st.metric("Trades", metrics['num_trades'])
            
            # Entry/Exit Log
            st.subheader("Entry/Exit Log:")
            trades = results['trades']
            if trades:
                log_lines = []
                for i, trade in enumerate(trades[:10], 1):  # Show up to 10 trades
                    entry_date = trade['entry_date'].split()[0] if isinstance(trade['entry_date'], str) else str(trade['entry_date'])
                    exit_date = trade['exit_date'].split()[0] if isinstance(trade['exit_date'], str) else str(trade['exit_date'])
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    
                    log_lines.append(f"- Enter: {entry_date} at {entry_price:.0f}")
                    log_lines.append(f"  Exit:  {exit_date} at {exit_price:.0f}")
                
                if len(trades) > 10:
                    log_lines.append("...")
                
                st.code("\n".join(log_lines), language=None)
            else:
                st.info("No trades executed")
            
            st.markdown("---")
            st.success("✅ Complete End-to-End Pipeline Executed Successfully!")
            
            # Download final report
            st.subheader("📥 Download Final Report")
            report_text = f"""Part 6 — End-to-End Demonstration - Final Report
{'=' * 80}

Natural Language Input:
"{nl_input}"

Generated DSL:
{dsl_text}

Parsed AST:
{json.dumps(ast, indent=2)}

Backtest Result:
Total Return: {metrics['total_return']:+.1f}%
Max Drawdown: {metrics['max_drawdown']:.1f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Win Rate: {metrics['win_rate']:.1f}%
Profit Factor: {metrics['profit_factor']:.2f}
Trades: {metrics['num_trades']}

Entry/Exit Log:
"""
            if trades:
                for trade in trades:
                    entry_date = trade['entry_date']
                    exit_date = trade['exit_date']
                    report_text += f"- Enter: {entry_date} at {trade['entry_price']:.2f}\n"
                    report_text += f"  Exit:  {exit_date} at {trade['exit_price']:.2f}\n"
            
            report_text += f"\n{'=' * 80}\n"
            
            st.download_button(
                "Download Complete Report (TXT)",
                report_text,
                "final_report.txt",
                "text/plain"
            )
            
        else:
            st.info("Process a strategy in the 'Natural Language Input' tab first to see the final report.")
    
    # Tab 7: Documentation
    with tab7:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## Overview
        
        This application implements a complete pipeline for creating trading strategies:
        
        1. **Natural Language Parsing**: Convert plain English descriptions into structured data
        2. **DSL Generation**: Transform structured data into a Domain-Specific Language
        3. **AST Construction**: Parse DSL into an Abstract Syntax Tree
        4. **Code Generation**: Convert AST into executable Python code
        5. **Backtest Execution**: Run the strategy on historical data and calculate metrics
        
        ## Supported Features
        
        ### Fields
        - `close`, `open`, `high`, `low`, `volume`, `price`
        
        ### Indicators
        - **SMA**: Simple Moving Average - `sma(close, 20)`
        - **EMA**: Exponential Moving Average - `ema(close, 20)`
        - **RSI**: Relative Strength Index - `rsi(close, 14)`
        - **MACD**: Moving Average Convergence Divergence - `macd(close, 12, 26, 9)`
        - **Bollinger Bands**: `bollinger(close, 20, 2)`
        - **ATR**: Average True Range - `atr(high, low, close, 14)`
        - **Stochastic**: `stochastic(high, low, close, 14, 3, 3)`
        - **VWAP**: Volume Weighted Average Price
        - **OBV**: On Balance Volume
        
        ### Operators
        - Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
        - Logical: `AND`, `OR`
        - Cross: `crosses above`, `crosses below`
        
        ### Time References
        - `yesterday's high/low/close`
        - `last week`
        
        ## Example Strategies
        
        ### Trend Following
        ```
        Buy when close is above 50-day moving average and volume is above 1 million.
        Sell when close is below 20-day moving average.
        ```
        
        ### Mean Reversion
        ```
        Enter when RSI(14) is below 30.
        Exit when RSI(14) is above 70.
        ```
        
        ### Breakout
        ```
        Buy when price crosses above yesterday's high and volume is above 1 million.
        Sell when price crosses below yesterday's low.
        ```
        
        ### Multi-Indicator
        ```
        Enter when close is above SMA(50) and RSI is above 50 and volume is above average.
        Exit when close is below SMA(20) or RSI is below 30.
        ```
        
        ## DSL Grammar
        
        ```
        ENTRY:
            condition1 AND condition2 OR condition3
        EXIT:
            condition1
        ```
        
        Where each condition follows the format:
        ```
        field|indicator operator value|field|indicator
        ```
        
        ## Performance Metrics
        
        - **Total Return**: Overall percentage return
        - **Max Drawdown**: Largest peak-to-trough decline
        - **Sharpe Ratio**: Risk-adjusted return measure
        - **Win Rate**: Percentage of profitable trades
        - **Profit Factor**: Ratio of gross profit to gross loss
        - **Average Win/Loss**: Mean profit/loss per trade
        
        ## Tips for Writing Strategies
        
        1. Be specific with indicators and their parameters
        2. Use clear entry and exit conditions
        3. Combine multiple conditions with AND/OR
        4. Test different timeframes and parameters
        5. Consider both trend and momentum indicators
        
        """)


def convert_json_to_dsl(structured_json: Dict) -> str:
    """
    Convert structured JSON to DSL text format
    
    Args:
        structured_json: Structured representation from NL parser
    
    Returns:
        str: DSL text
    """
    dsl_lines = []
    
    # Entry conditions
    if 'entry' in structured_json and structured_json['entry']:
        dsl_lines.append("ENTRY:")
        conditions = []
        for i, cond in enumerate(structured_json['entry']):
            cond_str = f"{cond['left']} {cond['operator']} {cond['right']}"
            if i > 0 and cond.get('logic'):
                cond_str = f" {cond['logic']} " + cond_str
            conditions.append(cond_str)
        dsl_lines.append("    " + "".join(conditions))
    
    # Exit conditions
    if 'exit' in structured_json and structured_json['exit']:
        dsl_lines.append("\nEXIT:")
        conditions = []
        for i, cond in enumerate(structured_json['exit']):
            cond_str = f"{cond['left']} {cond['operator']} {cond['right']}"
            if i > 0 and cond.get('logic'):
                cond_str = f" {cond['logic']} " + cond_str
            conditions.append(cond_str)
        dsl_lines.append("    " + "".join(conditions))
    
    return "\n".join(dsl_lines)


if __name__ == "__main__":
    main()
