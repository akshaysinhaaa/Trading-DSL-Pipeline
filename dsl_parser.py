"""
Parses DSL text into Abstract Syntax Tree (AST)
"""
import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ASTNode:
    """Base class for AST nodes"""
    type: str


@dataclass
class SeriesNode(ASTNode):
    """Represents a data series (e.g., 'close', 'volume')"""
    value: str
    
    def __init__(self, value: str):
        super().__init__(type="series")
        self.value = value


@dataclass
class ConstantNode(ASTNode):
    """Represents a constant value"""
    value: Union[int, float]
    
    def __init__(self, value: Union[int, float]):
        super().__init__(type="constant")
        self.value = value


@dataclass
class IndicatorNode(ASTNode):
    """Represents a technical indicator"""
    name: str
    params: List[Any]
    
    def __init__(self, name: str, params: List[Any]):
        super().__init__(type="indicator")
        self.name = name
        self.params = params


@dataclass
class BinaryOpNode(ASTNode):
    """Represents a binary operation"""
    left: ASTNode
    op: str
    right: ASTNode
    
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        super().__init__(type="binary_op")
        self.left = left
        self.op = op
        self.right = right


@dataclass
class LogicalOpNode(ASTNode):
    """Represents a logical operation (AND/OR)"""
    op: str
    conditions: List[ASTNode]
    
    def __init__(self, op: str, conditions: List[ASTNode]):
        super().__init__(type="logical_op")
        self.op = op
        self.conditions = conditions


@dataclass
class TimeReferenceNode(ASTNode):
    """Represents a time-based reference (e.g., yesterday)"""
    reference: str
    field: str
    offset: int = 1
    
    def __init__(self, reference: str, field: str, offset: int = 1):
        super().__init__(type="time_reference")
        self.reference = reference
        self.field = field
        self.offset = offset


class DSLParser:
    """
    Parses DSL text into AST
    
    DSL Grammar:
    ------------
    strategy    : ENTRY ':' conditions EXIT ':' conditions
    conditions  : condition (('AND' | 'OR') condition)*
    condition   : expression operator expression
    expression  : field | indicator | constant | time_reference
    field       : 'close' | 'open' | 'high' | 'low' | 'volume'
    indicator   : name '(' params ')'
    operator    : '>' | '<' | '>=' | '<=' | '==' | '!='
    """
    
    def __init__(self):
        self.valid_fields = ['close', 'open', 'high', 'low', 'volume']
        self.valid_operators = ['>', '<', '>=', '<=', '==', '!=', 'cross_above', 'cross_below']
        self.valid_indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 
                                'stochastic', 'vwap', 'obv']
    
    def parse(self, dsl_text: str) -> Dict[str, Any]:
        """
        Parse DSL text into AST
        
        Args:
            dsl_text: DSL text to parse
        
        Returns:
            dict: AST representation
        """
        # Normalize text
        dsl_text = dsl_text.strip()
        
        # Split into ENTRY and EXIT sections
        entry_section, exit_section = self._split_sections(dsl_text)
        
        # Parse each section
        ast = {
            'entry': self._parse_conditions(entry_section) if entry_section else [],
            'exit': self._parse_conditions(exit_section) if exit_section else []
        }
        
        return ast
    
    def _split_sections(self, text: str) -> tuple:
        """Split DSL text into ENTRY and EXIT sections"""
        entry_match = re.search(r'ENTRY\s*:\s*(.*?)(?=EXIT\s*:|$)', text, re.IGNORECASE | re.DOTALL)
        exit_match = re.search(r'EXIT\s*:\s*(.*?)$', text, re.IGNORECASE | re.DOTALL)
        
        entry_section = entry_match.group(1).strip() if entry_match else ""
        exit_section = exit_match.group(1).strip() if exit_match else ""
        
        return entry_section, exit_section
    
    def _parse_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Parse conditions from a section"""
        if not text:
            return []
        
        conditions = []
        
        # Split by AND/OR at the top level (not inside parentheses)
        parts = self._split_logical_operators(text)
        
        for part, logic_op in parts:
            condition = self._parse_single_condition(part.strip())
            if condition:
                condition_dict = self._ast_to_dict(condition)
                if logic_op:
                    condition_dict['logic'] = logic_op
                conditions.append(condition_dict)
        
        return conditions
    
    def _split_logical_operators(self, text: str) -> List[tuple]:
        """Split text by AND/OR operators, respecting parentheses"""
        parts = []
        current_logic = None
        
        # Split by AND
        and_parts = re.split(r'\s+AND\s+', text, flags=re.IGNORECASE)
        
        if len(and_parts) > 1:
            for i, part in enumerate(and_parts):
                parts.append((part, 'AND' if i > 0 else None))
        else:
            # Try OR
            or_parts = re.split(r'\s+OR\s+', text, flags=re.IGNORECASE)
            if len(or_parts) > 1:
                for i, part in enumerate(or_parts):
                    parts.append((part, 'OR' if i > 0 else None))
            else:
                parts.append((text, None))
        
        return parts
    
    def _parse_single_condition(self, text: str) -> Optional[ASTNode]:
        """Parse a single condition into an AST node"""
        # Extract operator
        operator = None
        for op in self.valid_operators:
            if op in text:
                operator = op
                break
        
        if not operator:
            raise ValueError(f"No valid operator found in condition: {text}")
        
        # Split by operator
        parts = text.split(operator, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid condition format: {text}")
        
        left_text = parts[0].strip()
        right_text = parts[1].strip()
        
        # Parse left and right expressions
        left_node = self._parse_expression(left_text)
        right_node = self._parse_expression(right_text)
        
        return BinaryOpNode(left_node, operator, right_node)
    
    def _parse_expression(self, text: str) -> ASTNode:
        """Parse an expression into an AST node"""
        text = text.strip()
        
        # Check for indicator pattern: name(params)
        indicator_match = re.match(r'(\w+)\s*\((.*?)\)', text)
        if indicator_match:
            name = indicator_match.group(1).lower()
            params_text = indicator_match.group(2)
            
            if name not in self.valid_indicators:
                raise ValueError(f"Unknown indicator: {name}")
            
            # Parse parameters
            params = [p.strip() for p in params_text.split(',')]
            params = [self._parse_param(p) for p in params if p]
            
            return IndicatorNode(name, params)
        
        # Check for time reference: yesterday(field) or lastweek(field)
        time_match = re.match(r'(yesterday|lastweek)\s*\((\w+)\)', text, re.IGNORECASE)
        if time_match:
            reference = time_match.group(1).lower()
            field = time_match.group(2).lower()
            offset = 1 if reference == 'yesterday' else 5
            return TimeReferenceNode(reference, field, offset)
        
        # Check for numeric constant
        try:
            value = float(text)
            return ConstantNode(value)
        except ValueError:
            pass
        
        # Check for field
        if text.lower() in self.valid_fields:
            return SeriesNode(text.lower())
        
        raise ValueError(f"Unable to parse expression: {text}")
    
    def _parse_param(self, param: str) -> Union[str, int, float]:
        """Parse a parameter value"""
        param = param.strip()

        try:
            if '.' in param:
                return float(param)
            return int(param)
        except ValueError:
            pass
        
        # Return as string (field name)
        return param
    
    def _ast_to_dict(self, node: ASTNode) -> Dict[str, Any]:
        """Convert AST node to dictionary"""
        if isinstance(node, SeriesNode):
            return {'type': 'series', 'value': node.value}
        elif isinstance(node, ConstantNode):
            return {'type': 'constant', 'value': node.value}
        elif isinstance(node, IndicatorNode):
            return {'type': 'indicator', 'name': node.name, 'params': node.params}
        elif isinstance(node, BinaryOpNode):
            return {
                'type': 'binary_op',
                'left': self._ast_to_dict(node.left),
                'op': node.op,
                'right': self._ast_to_dict(node.right)
            }
        elif isinstance(node, TimeReferenceNode):
            return {
                'type': 'time_reference',
                'reference': node.reference,
                'field': node.field,
                'offset': node.offset
            }
        else:
            return {}
    
    def validate_ast(self, ast: Dict[str, Any]) -> bool:
        """Validate the AST structure"""
        # Check for required sections
        if 'entry' not in ast and 'exit' not in ast:
            raise ValueError("AST must contain at least 'entry' or 'exit' section")
        
        return True
    
    def to_json(self, ast: Dict[str, Any], indent: int = 2) -> str:
        """Convert AST to JSON string"""
        return json.dumps(ast, indent=indent)


if __name__ == "__main__":
    parser = DSLParser()
    
    test_dsl = """
    ENTRY:
        close > sma(close,20) AND volume > 1000000
    EXIT:
        rsi(close,14) < 30
    """
    
    print("Testing DSL Parser")
    print("=" * 50)
    print(f"Input DSL:\n{test_dsl}")
    print("\nParsed AST:")
    ast = parser.parse(test_dsl)
    print(json.dumps(ast, indent=2))
