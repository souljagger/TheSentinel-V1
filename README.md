# TheSentinel-V1
TheSentinel-V1 is an advanced yet highly secure trading Expert Advisor (EA) developed specifically for MetaTrader 5 (MT5).
## 🚀 Trading Strategy (Trend-Following & Multi-Indicator Confirmation)

### Indicators used:
- **EMA 50 & EMA 200** (Trend identification)
- **RSI (14 periods)** (Trend strength confirmation)
- **MACD (12,26,9)** (Momentum confirmation)

### Entry Conditions:
- **BUY:**
  - Price above EMA 50 and EMA 200.
  - EMA 50 above EMA 200.
  - RSI > 50 and < 70.
  - MACD positive and increasing histogram.

- **SELL:**
  - Price below EMA 50 and EMA 200.
  - EMA 50 below EMA 200.
  - RSI < 50 and > 30.
  - MACD negative and decreasing histogram.

### Risk Management:
- Risk per trade: **Max 1%** of capital.
- Stop-Loss: **fixed at 40 pips**
- Take-Profit: **fixed at 80 pips (Ratio 1:2)**
- Trailing-Stop activated at **+25 pips**.

### Capital Protection:
- Daily Equity protection at **-3%** loss maximum.
