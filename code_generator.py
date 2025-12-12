"""
Code Generator Module
Converts AST into executable Python code for strategy evaluation
"""
import pandas as pd
from typing import Dict, Any, List
import indicators


class CodeGenerator:
    """
    Generates Python code from AST to evaluate trading strategies
    """
    
    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "
    
    def generate(self, ast: Dict[str, Any]) -> str:
        """
        Generate Python function from AST
        
        Args:
            ast: Abstract Syntax Tree
        
        Returns:
            str: Generated Python code
        """
        code_lines = []
        
        # Function definition
        code_lines.append("def evaluate_strategy(df):")
        code_lines.append("    \"\"\"")
        code_lines.append("    Evaluate trading strategy on OHLCV data")
        code_lines.append("    ")
        code_lines.append("    Args:")
        code_lines.append("        df: DataFrame with columns: open, high, low, close, volume")
        code_lines.append("    ")
        code_lines.append("    Returns:")
        code_lines.append("        dict: {'entry': Series, 'exit': Series}")
        code_lines.append("    \"\"\"")
        code_lines.append("    import pandas as pd")
        code_lines.append("    import numpy as np")
        code_lines.append("    import indicators")
        code_lines.append("")
        code_lines.append("    # Initialize signals")
        code_lines.append("    signals = pd.DataFrame(index=df.index)")
        code_lines.append("")
        
        # Generate entry conditions
        if 'entry' in ast and ast['entry']:
            code_lines.append("    # Entry conditions")
            entry_code = self._generate_conditions(ast['entry'], "entry")
            code_lines.append(f"    signals['entry'] = {entry_code}")
        else:
            code_lines.append("    signals['entry'] = False")
        
        code_lines.append("")
        
        # Generate exit conditions
        if 'exit' in ast and ast['exit']:
            code_lines.append("    # Exit conditions")
            exit_code = self._generate_conditions(ast['exit'], "exit")
            code_lines.append(f"    signals['exit'] = {exit_code}")
        else:
            code_lines.append("    signals['exit'] = False")
        
        code_lines.append("")
        code_lines.append("    return signals")
        
        return "\n".join(code_lines)
    
    def _generate_conditions(self, conditions: List[Dict[str, Any]], signal_type: str) -> str:
        """
        Generate code for a list of conditions
        
        Args:
            conditions: List of condition dictionaries
            signal_type: 'entry' or 'exit'
        
        Returns:
            str: Python expression for combined conditions
        """
        if not conditions:
            return "False"
        
        # Group conditions by logic operator
        and_conditions = []
        or_conditions = []
        
        for i, condition in enumerate(conditions):
            logic = condition.get('logic', 'AND' if i > 0 else None)
            condition_code = self._generate_single_condition(condition)
            
            if logic == 'OR':
                or_conditions.append(condition_code)
            else:
                and_conditions.append(condition_code)
        
        # Combine conditions
        if or_conditions and and_conditions:
            and_expr = " & ".join([f"({c})" for c in and_conditions])
            or_expr = " | ".join([f"({c})" for c in or_conditions])
            return f"(({and_expr}) | ({or_expr}))"
        elif or_conditions:
            return " | ".join([f"({c})" for c in or_conditions])
        else:
            return " & ".join([f"({c})" for c in and_conditions])
    
    def _generate_single_condition(self, condition: Dict[str, Any]) -> str:
        """
        Generate code for a single condition
        
        Args:
            condition: Condition dictionary from AST
        
        Returns:
            str: Python expression for the condition
        """
        if condition.get('type') == 'binary_op':
            left = self._generate_expression(condition['left'])
            op = condition['op']
            right = self._generate_expression(condition['right'])
            
            # Handle cross operations specially
            if op == 'cross_above':
                # Check if right is a constant (no shift needed) or a series
                if condition['right'].get('type') == 'constant':
                    # Right is constant, no shift needed
                    return f"({left} > {right}) & ({left}.shift(1) <= {right})"
                else:
                    # Both are series, shift both
                    return f"({left} > {right}) & ({left}.shift(1) <= {right}.shift(1))"
            elif op == 'cross_below':
                # Check if right is a constant (no shift needed) or a series
                if condition['right'].get('type') == 'constant':
                    # Right is constant, no shift needed
                    return f"({left} < {right}) & ({left}.shift(1) >= {right})"
                else:
                    # Both are series, shift both
                    return f"({left} < {right}) & ({left}.shift(1) >= {right}.shift(1))"
            else:
                return f"({left} {op} {right})"
        
        return "False"
    
    def _generate_expression(self, expr: Dict[str, Any]) -> str:
        """
        Generate code for an expression
        
        Args:
            expr: Expression dictionary from AST
        
        Returns:
            str: Python expression
        """
        expr_type = expr.get('type')
        
        if expr_type == 'series':
            # Direct series reference
            return f"df['{expr['value']}']"
        
        elif expr_type == 'constant':
            # Constant value
            return str(expr['value'])
        
        elif expr_type == 'indicator':
            # Indicator function call
            return self._generate_indicator(expr)
        
        elif expr_type == 'time_reference':
            # Time-based reference (e.g., yesterday)
            field = expr['field']
            offset = expr.get('offset', 1)
            return f"df['{field}'].shift({offset})"
        
        elif expr_type == 'binary_op':
            # Nested binary operation
            return self._generate_single_condition(expr)
        
        return "0"
    
    def _generate_indicator(self, indicator: Dict[str, Any]) -> str:
        """
        Generate code for indicator calculation
        
        Args:
            indicator: Indicator dictionary from AST
        
        Returns:
            str: Python code to calculate indicator
        """
        name = indicator['name']
        params = indicator['params']
        
        # Map parameter names
        param_strs = []
        for param in params:
            if isinstance(param, str) and param in ['close', 'open', 'high', 'low', 'volume']:
                param_strs.append(f"df['{param}']")
            else:
                param_strs.append(str(param))
        
        params_code = ", ".join(param_strs)
        
        # Special handling for different indicators
        if name in ['sma', 'ema', 'rsi']:
            # Ensure first param is a series for these indicators
            if len(param_strs) > 0 and "df[" not in param_strs[0]:
                # First param is not a series, default to close
                param_strs = [f"df['close']"] + param_strs
                params_code = ", ".join(param_strs)
            return f"indicators.{name}({params_code})"
        elif name == 'macd':
            return f"indicators.{name}({params_code})[0]"  # Return MACD line
        elif name == 'bollinger':
            return f"indicators.bollinger_bands({params_code})[1]"  # Return middle band
        elif name == 'atr':
            if len(param_strs) < 3:
                param_strs = [f"df['high']", f"df['low']", f"df['close']"] + param_strs
            return f"indicators.atr({', '.join(param_strs)})"
        elif name == 'stochastic':
            if len(param_strs) < 3:
                param_strs = [f"df['high']", f"df['low']", f"df['close']"] + param_strs
            return f"indicators.stochastic({', '.join(param_strs)})[0]"  # Return %K
        elif name == 'vwap':
            return f"indicators.vwap(df['high'], df['low'], df['close'], df['volume'])"
        elif name == 'obv':
            return f"indicators.obv(df['close'], df['volume'])"
        
        return f"indicators.{name}({params_code})"
    
    def generate_and_execute(self, ast: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate code from AST and execute it on data
        
        Args:
            ast: Abstract Syntax Tree
            df: DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: Signals with entry and exit columns
        """
        # Generate code
        code = self.generate(ast)
        
        # Create execution environment
        exec_globals = {
            'pd': pd,
            'indicators': indicators,
            'df': df
        }
        
        # Execute generated code
        exec(code, exec_globals)
        
        # Get the generated function
        evaluate_strategy = exec_globals['evaluate_strategy']
        
        # Execute strategy
        signals = evaluate_strategy(df)
        
        return signals


if __name__ == "__main__":
    # Test the code generator
    import json
    
    # Sample AST
    ast = {
        "entry": [
            {
                "type": "binary_op",
                "left": {"type": "series", "value": "close"},
                "op": ">",
                "right": {"type": "indicator", "name": "sma", "params": ["close", 20]}
            },
            {
                "type": "binary_op",
                "left": {"type": "series", "value": "volume"},
                "op": ">",
                "right": {"type": "constant", "value": 1000000},
                "logic": "AND"
            }
        ],
        "exit": [
            {
                "type": "binary_op",
                "left": {"type": "indicator", "name": "rsi", "params": ["close", 14]},
                "op": "<",
                "right": {"type": "constant", "value": 30}
            }
        ]
    }
    
    print("Testing Code Generator")
    print("=" * 50)
    print("Input AST:")
    print(json.dumps(ast, indent=2))
    print("\n" + "=" * 50)
    print("Generated Code:")
    print("=" * 50)
    
    generator = CodeGenerator()
    code = generator.generate(ast)
    print(code)
