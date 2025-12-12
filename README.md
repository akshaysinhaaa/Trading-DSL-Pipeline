# Trading Strategy DSL - NLP to Execution Pipeline

A comprehensive implementation of a natural language to executable code pipeline for trading strategies. This project converts plain English trading rules into a Domain-Specific Language (DSL), parses them into an Abstract Syntax Tree (AST), generates executable Python code, and runs backtests with full performance metrics.


### Pipeline Flow

1. **Natural Language Input** → Parse English descriptions
2. **Structured JSON** → Convert to intermediate representation
3. **DSL Text** → Generate domain-specific language
4. **Abstract Syntax Tree** → Parse DSL into AST
5. **Python Code** → Generate executable strategy code
6. **Backtest Execution** → Run simulation and calculate metrics

## 📁 Project Structure

```
.
├── indicators.py          # Technical indicators (SMA, RSI, MACD, etc.)
├── nl_parser.py           # Natural language to structured JSON parser
├── dsl_parser.py          # DSL parser and AST builder
├── code_generator.py      # AST to Python code generator
├── backtest.py            # Backtest simulator with performance metrics
├── main.py                # Streamlit web application
├── GRAMMAR_SUPPORT.md     # Detailed DSL grammar documentation
└── README.md              # This file
```

## 🚀 Quick Start

### Clone

```bash
git clone https://github.com/akshaysinhaaa/Trading-DSL-Pipeline/tree/main
```

### Prerequisites

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy
```

### Option 1: Interactive Web Application (Recommended)

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`


### Option 2: Command-Line Demo

```bash
python demo.py
```

**Shows complete pipeline:**
1. Natural Language Input
2. Structured JSON
3. DSL Text Generation
4. AST Construction
5. Python Code Generation
6. Strategy Execution
7. Backtest Results with Final Report

### Option 3: Test Individual Components

```bash
# Test technical indicators
python indicators.py

# Test natural language parser
python nl_parser.py

# Test DSL parser
python dsl_parser.py

# Test code generator
python code_generator.py

# Test backtest simulator
python backtest.py
```
