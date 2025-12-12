import pandas as pd
import numpy as np
import json
from datetime import datetime

from nl_parser import NLParser
from dsl_parser import DSLParser
from code_generator import CodeGenerator
from backtest import BacktestSimulator


def generate_sample_data(days=252):
    """Generate sample OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    returns = np.random.randn(days) * 0.02
    trend = np.linspace(0, 0.3, days)
    price = 100 * np.exp((returns + trend / days).cumsum())
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(days) * 0.005),
        'high': price * (1 + abs(np.random.randn(days)) * 0.01),
        'low': price * (1 - abs(np.random.randn(days)) * 0.01),
        'close': price,
        'volume': np.random.randint(800000, 1500000, days)
    }, index=dates)
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def convert_json_to_dsl(structured_json):
    """Convert structured JSON to DSL text"""
    dsl_lines = []
    
    if 'entry' in structured_json and structured_json['entry']:
        dsl_lines.append("ENTRY:")
        conditions = []
        for i, cond in enumerate(structured_json['entry']):
            cond_str = f"{cond['left']} {cond['operator']} {cond['right']}"
            if i > 0 and cond.get('logic'):
                cond_str = f" {cond['logic']} " + cond_str
            conditions.append(cond_str)
        dsl_lines.append("    " + "".join(conditions))
    
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


def main():
    print("=" * 80)
    print("DEMO TRADING STRATEGY DSL:")
    print("=" * 80)
    print()

    # Natural Language Input
    nl_input = """
    Buy when the close price is above the 20-day moving average and volume is above 1 million.
    Exit when RSI(14) is below 30.
    """

    print("Part 1: Natural Language Parsing")
    print(nl_input.strip())
    print()

    print("Parse Natural Language → Structured JSON")
    nl_parser = NLParser()
    structured_json = nl_parser.parse(nl_input)
    print(json.dumps(structured_json, indent=2))
    print()

    # Part 2
    print("-" * 80)
    print("Part 2: Domain Specific Language (DSL) Design")
    dsl_text = convert_json_to_dsl(structured_json)
    print(dsl_text)
    print()


    #Part 3
    print("-" * 80)
    print("Part 3: DSL Parser → Abstract Syntax Tree (AST) Construction")
    dsl_parser = DSLParser()
    ast = dsl_parser.parse(dsl_text)
    print(json.dumps(ast, indent=2))
    print()

    #Part 4
    print("-" * 80)
    print("Part 4: From AST to Python Code Generation")
    code_gen = CodeGenerator()
    generated_code = code_gen.generate(ast)
    print(generated_code)
    print()

    #Part 5
    print("-" * 80)
    print("Part 5: Simple Backtest Simulation")
    df = generate_sample_data(252)
    print(f"Generated {len(df)} days of OHLCV data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print()

    # Generate signals
    signals = code_gen.generate_and_execute(ast, df)
    print(f"Entry signals: {signals['entry'].sum()}")
    print(f"Exit signals: {signals['exit'].sum()}")
    print()

    # Run backtest
    simulator = BacktestSimulator(initial_capital=10000)
    results = simulator.run(df, signals)

    # Display Results
    simulator.print_results(results)
    print()
    
    # Final Report
    print()
    print("=" * 80)
    print("Part 6: Final Report")
    print("=" * 80)
    print()
    
    print("Natural Language Input:")
    print(f'"{nl_input.strip()}"')
    print()
    
    print("Generated DSL:")
    print(dsl_text)
    print()
    
    print("Parsed AST:", json.dumps(ast, indent=2)[:200] + "...}")
    print()
    
    print("Backtest Result:")
    metrics = results['metrics']
    print(f"Total Return: {metrics['total_return']:+.1f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
    print(f"Trades: {metrics['num_trades']}")
    print()
    
    print("Entry/Exit Log:")
    trades = results['trades']
    if trades:
        for i, trade in enumerate(trades[:5], 1):  # Show first 5 trades
            entry_date = trade['entry_date'].split()[0] if isinstance(trade['entry_date'], str) else str(trade['entry_date'])
            exit_date = trade['exit_date'].split()[0] if isinstance(trade['exit_date'], str) else str(trade['exit_date'])
            print(f"- Enter: {entry_date} at {trade['entry_price']:.0f}")
            print(f"  Exit:  {exit_date} at {trade['exit_price']:.0f}")
        if len(trades) > 5:
            print("...")
    print()
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    
    print("You can also:")
    print("Run 'streamlit run main.py' to launch the interactive web app")
    print()


if __name__ == "__main__":
    main()
