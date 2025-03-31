#include <Trade/Trade.mqh>
#include "Classes/RiskManager.mqh"
#include "Classes/TrailingStop.mqh"
#include "Strategies/RSIHybridStrategy.mqh"

// === External Parameters (Optimizable) ===
input double RiskPerTrade = 1.0;             // % of capital to risk per trade
input double DailyLossLimit = 3.0;           // Max daily loss in %
input int MaxTradesPerDay = 5;               // Max trades allowed per day

input int RSI_Period = 14;                   // RSI Period
input double RSI_Upper = 70;                 // RSI overbought level
input double RSI_Lower = 30;                 // RSI oversold level

input int EMA_Fast = 9;                      // Fast EMA period
input int EMA_Slow = 21;                     // Slow EMA period

input double SL_ATR_Multiplier = 1.5;        // SL distance in ATR multipliers
input double TP_Multiplier = 2.0;            // TP distance relative to SL
input ENUM_TIMEFRAMES ATR_Timeframe = PERIOD_H1; // ATR timeframe

// === Global Variables ===
CTrade trade;
RiskManager riskManager;
TrailingStop trailing;
RSIHybridStrategy strategy;

int tradeCountToday = 0;
double equityStartOfDay = 0;
datetime dayStart = 0;

int OnInit() {
    Print("[TheSentinel] EA initialized.");
    strategy.Init(RSI_Period, RSI_Upper, RSI_Lower, EMA_Fast, EMA_Slow, ATR_Timeframe, SL_ATR_Multiplier, TP_Multiplier);
    riskManager.Init(RiskPerTrade, DailyLossLimit);
    trailing.SetTradePointer(&trade);
    dayStart = iTime(_Symbol, PERIOD_D1, 0);
    equityStartOfDay = AccountInfoDouble(ACCOUNT_EQUITY);
    return INIT_SUCCEEDED;
}

void OnTick() {
    // Reset daily counters if new day started
    if (TimeCurrent() >= dayStart + 86400) {
        tradeCountToday = 0;
        dayStart = iTime(_Symbol, PERIOD_D1, 0);
        equityStartOfDay = AccountInfoDouble(ACCOUNT_EQUITY);
    }

    // Capital protection: Daily loss limit
    if (!riskManager.CheckDailyLossLimit(equityStartOfDay)) return;

    // Limit number of trades per day
    if (tradeCountToday >= MaxTradesPerDay) return;

    // Avoid opening new position if one is already active
    if (PositionSelect(_Symbol)) {
        trailing.Update(); // Apply trailing stop if position is open
        return;
    }

    // Generate trade signal
    if (strategy.GenerateSignal()) {
        double slPoints = strategy.GetSLPoints();
        double lotSize = riskManager.CalculateLotSize(slPoints);
        double sl = strategy.GetSLPrice();
        double tp = strategy.GetTPPrice();
        ENUM_ORDER_TYPE orderType = strategy.GetOrderType();

        bool opened = false;
        if (orderType == ORDER_TYPE_BUY)
            opened = trade.Buy(lotSize, _Symbol, 0, sl, tp);
        else if (orderType == ORDER_TYPE_SELL)
            opened = trade.Sell(lotSize, _Symbol, 0, sl, tp);

        if (opened) {
            tradeCountToday++;
            Print("[TheSentinel] Trade opened successfully.");
        } else {
            Print("[TheSentinel] Failed to open trade.");
        }
    }
}
