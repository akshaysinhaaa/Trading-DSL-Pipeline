import re
import json
from typing import Dict, List, Any, Optional


class NLParser:
    """
    Parses natural language descriptions into json
    """
    
    def __init__(self):
        # Define supported fields with variations
        self.fields = ['close', 'open', 'high', 'low', 'volume', 'price']
        self.field_aliases = {
            # Close price variations
            'close': 'close',
            'closing': 'close',
            'closing price': 'close',
            'close price': 'close',
            'last': 'close',
            'last price': 'close',
            
            # Open price variations
            'open': 'open',
            'opening': 'open',
            'opening price': 'open',
            'open price': 'open',
            
            # High price variations
            'high': 'high',
            'highest': 'high',
            'high price': 'high',
            'highest price': 'high',
            'top': 'high',
            'peak': 'high',
            
            # Low price variations
            'low': 'low',
            'lowest': 'low',
            'low price': 'low',
            'lowest price': 'low',
            'bottom': 'low',
            
            # Volume variations
            'volume': 'volume',
            'vol': 'volume',
            'trading volume': 'volume',
            'trade volume': 'volume',
            
            # Price (generic)
            'price': 'close',
            'stock price': 'close',
            'share price': 'close',
            'current price': 'close'
        }
        
        # Define supported indicators with expanded patterns
        # Order matters! More specific patterns first to avoid false matches
        self.indicators = {
            'macd': r'\bmacd\b|moving\s+average\s+convergence\s+divergence',
            'vwap': r'\bvwap\b|volume\s+weighted\s+average\s+price|volume\s+weighted\s+price',
            'obv': r'\bobv\b|on\s+balance\s+volume|on-balance\s+volume',
            'ema': r'\bema\b|exponential\s+moving\s+average|exp\s+moving\s+average|exponential\s+ma',
            'rsi': r'\brsi\b|relative\s+strength\s+index|relative\s+strength',
            'bollinger': r'bollinger\s+bands?|\bbb\b|bollinger',
            'atr': r'\batr\b|average\s+true\s+range|true\s+range',
            'stochastic': r'stochastic|\bstoch\b|stochastic\s+oscillator',
            'sma': r'\bsma\b|(?:simple\s+)?moving\s+average|simple\s+ma|moving\s+avg|\bma\b',
        }
        
        # Expanded comparison operators with many variations
        self.operators = {
            # Greater than variations
            'above': '>',
            'is above': '>',
            'goes above': '>',
            'greater than': '>',
            'is greater than': '>',
            'more than': '>',
            'is more than': '>',
            'over': '>',
            'is over': '>',
            'exceeds': '>',
            'higher than': '>',
            'is higher than': '>',
            'bigger than': '>',
            'is bigger than': '>',
            'larger than': '>',
            'is larger than': '>',
            
            # Less than variations
            'below': '<',
            'is below': '<',
            'goes below': '<',
            'less than': '<',
            'is less than': '<',
            'fewer than': '<',
            'under': '<',
            'is under': '<',
            'lower than': '<',
            'is lower than': '<',
            'smaller than': '<',
            'is smaller than': '<',
            
            # Greater than or equal
            'at least': '>=',
            'is at least': '>=',
            'greater than or equal to': '>=',
            'greater than or equal': '>=',
            'at or above': '>=',
            'is at or above': '>=',
            
            # Less than or equal
            'at most': '<=',
            'is at most': '<=',
            'less than or equal to': '<=',
            'less than or equal': '<=',
            'at or below': '<=',
            'is at or below': '<=',
            
            # Equals variations
            'equals': '==',
            'is equal to': '==',
            'is equals to': '==',
            'equal to': '==',
            'is': '==',
            'becomes': '==',
            
            # Cross operations
            'crosses above': 'cross_above',
            'cross above': 'cross_above',
            'breaks above': 'cross_above',
            'breaks through': 'cross_above',
            'moves above': 'cross_above',
            'goes above': 'cross_above',
            'rises above': 'cross_above',
            'crosses over': 'cross_above',
            
            'crosses below': 'cross_below',
            'cross below': 'cross_below',
            'breaks below': 'cross_below',
            'falls below': 'cross_below',
            'moves below': 'cross_below',
            'goes below': 'cross_below',
            'drops below': 'cross_below',
            'crosses under': 'cross_below',
            
            # Default cross is above
            'crosses': 'cross_above'
        }
        
    def parse(self, nl_text: str) -> Dict[str, Any]:
        """
        Parse natural language text into structured JSON
        
        Args:
            nl_text: Natural language strategy description
        
        Returns:
            dict: Structured representation with entry and exit rules
        """
        result = {
            'entry': [],
            'exit': []
        }
        
        # Normalize text
        nl_text = nl_text.lower().strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s*|\n', nl_text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Determine if this is an entry or exit rule
            rule_type = self._identify_rule_type(sentence)
            
            # Extract conditions from the sentence
            conditions = self._extract_conditions(sentence)
            
            if conditions and rule_type:
                result[rule_type].extend(conditions)
        
        return result
    
    def _identify_rule_type(self, sentence: str) -> Optional[str]:
        """
        Identify if sentence describes entry or exit rules
        
        Args:
            sentence: Sentence to analyze
        
        Returns:
            str: 'entry' or 'exit' or None
        """
        # Expanded entry keywords with synonyms and variations
        entry_keywords = [
            'buy', 'buys', 'buying', 'purchase', 'purchases', 'purchasing',
            'enter', 'enters', 'entering', 'entry', 
            'long', 'go long', 'take long', 'open long',
            'open', 'open position', 'open trade',
            'acquire', 'acquires', 'acquiring',
            'trigger entry', 'signal entry', 'entry signal',
            'initiate', 'initiates', 'initiating',
            'start', 'starts', 'starting'
        ]
        
        # Expanded exit keywords with synonyms and variations
        exit_keywords = [
            'sell', 'sells', 'selling',
            'exit', 'exits', 'exiting',
            'close', 'closes', 'closing', 'close position', 'close trade',
            'liquidate', 'liquidates', 'liquidating',
            'stop', 'stop loss', 'stop out',
            'take profit', 'book profit', 'realize profit',
            'trigger exit', 'signal exit', 'exit signal',
            'leave', 'leaves', 'leaving',
            'terminate', 'terminates', 'terminating',
            'finish', 'finishes', 'finishing',
            'end', 'ends', 'ending'
        ]
        
        sentence_lower = sentence.lower()
        
        # Check for entry keywords
        for keyword in entry_keywords:
            if keyword in sentence_lower:
                return 'entry'
        
        # Check for exit keywords
        for keyword in exit_keywords:
            if keyword in sentence_lower:
                return 'exit'
        
        return None
    
    def _extract_conditions(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Extract trading conditions from a sentence
        
        Args:
            sentence: Sentence containing conditions
        
        Returns:
            list: List of condition dictionaries
        """
        conditions = []
        
        # Split by AND/OR to handle multiple conditions
        # First, let's identify all conditions separated by 'and'/'or'
        parts = re.split(r'\s+and\s+|\s+or\s+', sentence, flags=re.IGNORECASE)
        
        for part in parts:
            condition = self._parse_single_condition(part.strip())
            if condition:
                conditions.append(condition)
        
        # Identify the logical operator
        if len(conditions) > 1:
            if ' or ' in sentence.lower():
                # Mark conditions as OR-connected
                for cond in conditions:
                    cond['logic'] = 'OR'
            else:
                # Default to AND
                for cond in conditions:
                    cond['logic'] = 'AND'
        
        return conditions
    
    def _parse_single_condition(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single condition from text
        
        Args:
            text: Text containing a single condition
        
        Returns:
            dict: Condition dictionary or None
        """
        # Pattern: field/indicator operator value
        # Examples:
        # - "close is above the 20-day moving average"
        # - "volume is above 1 million"
        # - "RSI(14) is below 30"
        # - "MACD crosses above zero"
        # - "price crosses above yesterday's high"
        
        # Extract operator first
        operator = self._extract_operator(text)
        if not operator:
            return None
        
        # Split text by operator phrase to get left and right
        # Sort by length (longest first) to split by "crosses above" not "above"
        left_text = None
        right_text = None
        
        sorted_operators = sorted(self.operators.items(), key=lambda x: -len(x[0]))
        for phrase, op in sorted_operators:
            if op == operator and phrase in text:
                parts = text.split(phrase, 1)
                if len(parts) == 2:
                    left_text = parts[0].strip()
                    right_text = parts[1].strip()
                    break
        
        if not left_text or not right_text:
            # Fallback: couldn't split properly
            return None
        
        # Extract left side - check for indicator first, then field
        left_indicator = self._extract_indicator(left_text)
        if left_indicator:
            left = left_indicator
        else:
            left_field = self._extract_field(left_text)
            if left_field:
                left = left_field
            else:
                return None
        
        # Extract right side (value or another field/indicator)
        right = self._extract_right_side(right_text, operator)
        if right is None:
            return None
        
        return {
            'left': left,
            'operator': operator,
            'right': right
        }
    
    def _extract_field(self, text: str) -> Optional[str]:
        """Extract field name from text"""
        text_lower = text.lower()
        
        # Check aliases first (longer phrases first)
        for alias, field in sorted(self.field_aliases.items(), key=lambda x: -len(x[0])):
            if alias in text_lower:
                return field
        
        # Check direct field names
        for field in self.fields:
            if re.search(r'\b' + field + r'\b', text_lower):
                return field
        
        return None
    
    def _extract_indicator(self, text: str) -> Optional[str]:
        """Extract indicator expression from text"""
        # Look for patterns like "20-day moving average", "SMA(close,20)", "RSI(14)", "MACD"
        
        # Pattern: indicator(field, period) - with both params
        indicator_func_pattern = r'(sma|ema|rsi|macd|bollinger|atr|stochastic|vwap|obv)\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)'
        match = re.search(indicator_func_pattern, text, re.IGNORECASE)
        if match:
            indicator_name = match.group(1).lower()
            field = match.group(2)
            period = match.group(3)
            return f"{indicator_name}({field},{period})"
        
        # Pattern: indicator(period) - only period given, default to close
        indicator_period_pattern = r'(sma|ema|rsi|macd|bollinger|atr|stochastic|vwap|obv)\s*\(\s*(\d+)\s*\)'
        match = re.search(indicator_period_pattern, text, re.IGNORECASE)
        if match:
            indicator_name = match.group(1).lower()
            period = match.group(2)
            return f"{indicator_name}(close,{period})"
        
        # Pattern: Just indicator name (like "MACD", "RSI") without parameters
        # This handles cases like "MACD crosses above zero"
        for indicator, pattern in self.indicators.items():
            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                # Check if there's already a function call pattern (avoid duplicates)
                if '(' not in text or indicator not in text.lower():
                    # Use default parameters based on indicator
                    if indicator == 'macd':
                        return f"macd(close,12,26,9)"
                    elif indicator == 'rsi':
                        return f"rsi(close,14)"
                    elif indicator in ['sma', 'ema', 'bollinger']:
                        # Extract period if mentioned
                        period_match = re.search(r'(\d+)[-\s]?(?:day|period|bar)?', text)
                        period = period_match.group(1) if period_match else '20'
                        field = 'close'
                        for f in self.fields:
                            if f in text.lower():
                                field = f
                                break
                        return f"{indicator}({field},{period})"
                    elif indicator == 'stochastic':
                        return f"stochastic(high,low,close,14,3,3)"
                    elif indicator == 'atr':
                        return f"atr(high,low,close,14)"
                    elif indicator == 'vwap':
                        return f"vwap(high,low,close,volume)"
                    elif indicator == 'obv':
                        return f"obv(close,volume)"
        
        return None
    
    def _extract_operator(self, text: str) -> Optional[str]:
        """Extract comparison operator from text"""
        # Sort by length (longest first) to match "crosses above" before "above"
        sorted_operators = sorted(self.operators.items(), key=lambda x: -len(x[0]))
        
        for phrase, op in sorted_operators:
            if phrase in text:
                return op
        
        # Check for symbol operators
        for op in ['>=', '<=', '>', '<', '==', '!=']:
            if op in text:
                return op
        
        return None
    
    def _extract_right_side(self, text: str, operator: str) -> Any:
        """Extract the right side of a comparison"""
        text = text.strip()
        
        # Check for indicator first
        indicator = self._extract_indicator(text)
        if indicator:
            return indicator
        
        # Check for time-based references
        if 'yesterday' in text:
            time_match = re.search(r"yesterday'?s?\s+(\w+)", text)
            if time_match:
                field = time_match.group(1)
                return f"yesterday({field})"
        
        if 'last week' in text:
            field_match = re.search(r'(\w+)', text)
            field = field_match.group(1) if field_match else 'close'
            return f"lastweek({field})"
        
        # Check for field
        for field in self.fields:
            if re.search(r'\b' + field + r'\b', text):
                return field
        
        # Check for special keywords like "zero" (common in indicator comparisons)
        if text.lower() in ['zero', '0']:
            return 0
        
        # Extract numeric value
        # Handle patterns like "1 million", "1M", "30 percent", "30%"
        value_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:million|m)\b', lambda x: float(x) * 1000000),
            (r'(\d+(?:\.\d+)?)\s*(?:thousand|k)\b', lambda x: float(x) * 1000),
            (r'(\d+(?:\.\d+)?)\s*(?:percent|%)', lambda x: float(x)),
            (r'\b(\d+(?:\.\d+)?)\b', lambda x: float(x))
        ]
        
        for pattern, converter in value_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return converter(match.group(1))
        
        return None
    
    def to_json(self, structured_data: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert structured data to JSON string
        
        Args:
            structured_data: Structured rule dictionary
            indent: JSON indentation level
        
        Returns:
            str: JSON string
        """
        return json.dumps(structured_data, indent=indent)
    
    def parse_to_json(self, nl_text: str) -> str:
        """
        Parse natural language text directly to JSON string
        
        Args:
            nl_text: Natural language strategy description
        
        Returns:
            str: JSON string representation
        """
        structured = self.parse(nl_text)
        return self.to_json(structured)


if __name__ == "__main__":
    # Test the parser
    parser = NLParser()
    
    test_cases = [
        "Buy when the close price is above the 20-day moving average and volume is above 1 million.",
        "Enter when price crosses above yesterday's high.",
        "Exit when RSI(14) is below 30.",
        "Trigger entry when volume increases by more than 30 percent compared to last week."
    ]
    
    for test in test_cases:
        print(f"\nInput: {test}")
        result = parser.parse(test)
        print(f"Output: {json.dumps(result, indent=2)}")
