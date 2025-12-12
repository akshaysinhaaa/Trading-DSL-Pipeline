# Enhanced Grammar Support

## Overview

The Natural Language Parser now supports a **wide variety of phrasings** and **synonyms** to make it more flexible and user-friendly. You can describe your trading strategies in many different ways!

---

## ✅ What's Supported

### 1. **Action Words (Entry)**

All these words trigger **ENTRY** conditions:

| Category | Words |
|----------|-------|
| **Buy** | buy, buys, buying |
| **Purchase** | purchase, purchases, purchasing |
| **Acquire** | acquire, acquires, acquiring |
| **Enter** | enter, enters, entering, entry |
| **Long** | long, go long, take long, open long |
| **Open** | open, open position, open trade |
| **Start** | start, starts, starting |
| **Initiate** | initiate, initiates, initiating |
| **Trigger** | trigger entry, signal entry, entry signal |

**Examples:**
- ✅ "**Buy** when close is above SMA(20)"
- ✅ "**Purchase** when volume exceeds 1M"
- ✅ "**Acquire** when price breaks above yesterday's high"
- ✅ "**Enter** when RSI is below 30"
- ✅ "**Initiate** long when close is higher than 100"

---

### 2. **Action Words (Exit)**

All these words trigger **EXIT** conditions:

| Category | Words |
|----------|-------|
| **Sell** | sell, sells, selling |
| **Exit** | exit, exits, exiting |
| **Close** | close, closes, closing, close position, close trade |
| **Liquidate** | liquidate, liquidates, liquidating |
| **Stop** | stop, stop loss, stop out |
| **Profit** | take profit, book profit, realize profit |
| **Leave** | leave, leaves, leaving |
| **Terminate** | terminate, terminates, terminating |
| **Finish** | finish, finishes, finishing |
| **End** | end, ends, ending |
| **Trigger** | trigger exit, signal exit, exit signal |

**Examples:**
- ✅ "**Sell** when RSI is above 70"
- ✅ "**Exit** when price crosses below SMA(50)"
- ✅ "**Close** position when volume drops below 500K"
- ✅ "**Liquidate** when price falls below yesterday's low"
- ✅ "**Take profit** when close is 10% above entry"

---

### 3. **Field Names**

Multiple ways to refer to price fields:

| Standard Field | Variations Supported |
|----------------|---------------------|
| **close** | close, closing, closing price, close price, last, last price, price, stock price, share price, current price |
| **open** | open, opening, opening price, open price |
| **high** | high, highest, high price, highest price, top, peak |
| **low** | low, lowest, low price, lowest price, bottom |
| **volume** | volume, vol, trading volume, trade volume |

**Examples:**
- ✅ "when **close** is above 100"
- ✅ "when **closing price** exceeds SMA"
- ✅ "when **last price** breaks above"
- ✅ "when **stock price** is higher than"
- ✅ "when **trading volume** is more than 1M"

---

### 4. **Indicators**

Multiple ways to specify indicators:

| Indicator | Variations |
|-----------|-----------|
| **SMA** | SMA, simple moving average, moving average, MA, simple MA, moving avg |
| **EMA** | EMA, exponential moving average, exp moving average, exponential MA |
| **RSI** | RSI, relative strength index, relative strength |
| **MACD** | MACD, moving average convergence divergence |
| **Bollinger** | Bollinger, Bollinger Bands, BB |
| **ATR** | ATR, average true range, true range |
| **Stochastic** | Stochastic, stoch, stochastic oscillator |
| **VWAP** | VWAP, volume weighted average price, volume weighted price |
| **OBV** | OBV, on balance volume, on-balance volume |

**Examples:**
- ✅ "above the **20-day moving average**"
- ✅ "above **SMA(20)**"
- ✅ "above the **20-day MA**"
- ✅ "when **RSI(14)** is below 30"
- ✅ "when **relative strength index** is oversold"

---

### 5. **Comparison Operators**

Many ways to express comparisons:

| Operator | Variations | Result |
|----------|-----------|--------|
| **>** | above, is above, greater than, is greater than, more than, is more than, over, is over, exceeds, higher than, is higher than, bigger than, is bigger than, larger than, is larger than | Greater than |
| **<** | below, is below, less than, is less than, fewer than, under, is under, lower than, is lower than, smaller than, is smaller than | Less than |
| **>=** | at least, is at least, greater than or equal to, greater than or equal, at or above, is at or above | Greater than or equal |
| **<=** | at most, is at most, less than or equal to, less than or equal, at or below, is at or below | Less than or equal |
| **==** | equals, is equal to, equal to, is, becomes | Equals |

**Examples:**
- ✅ "price is **above** 100"
- ✅ "price is **higher than** 100"
- ✅ "price **exceeds** 100"
- ✅ "volume is **more than** 1M"
- ✅ "RSI is **at least** 50"
- ✅ "volume is **at most** 500K"

---

### 6. **Cross Operators**

Special operators for crossover detection:

| Type | Variations |
|------|-----------|
| **Cross Above** | crosses above, cross above, breaks above, breaks through, moves above, goes above, rises above, crosses over |
| **Cross Below** | crosses below, cross below, breaks below, falls below, moves below, goes below, drops below, crosses under |

**Examples:**
- ✅ "when price **crosses above** SMA(50)"
- ✅ "when close **breaks above** yesterday's high"
- ✅ "when price **rises above** the 20-day MA"
- ✅ "when close **crosses below** SMA(20)"
- ✅ "when price **falls below** yesterday's low"

---

### 7. **Case Insensitive**

All inputs are case-insensitive:

- ✅ "Buy when close is above SMA(20)"
- ✅ "buy when close is above sma(20)"
- ✅ "BUY WHEN CLOSE IS ABOVE SMA(20)"
- ✅ "BuY wHeN cLoSe Is AbOvE sMa(20)"

**All work the same!**

---


