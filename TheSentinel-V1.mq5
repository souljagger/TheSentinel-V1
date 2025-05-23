// === TheSentinel-V1.mq5 — Ultra-Protégé avec anti-range, pause après pertes & journal debug ===
#include <Trade/Trade.mqh>

// === Input Parameters ===
input double RiskPerTrade = 1.0;
input double DailyLossLimit = 3.0;
input int MaxTradesPerDay = 5;

input int RSI_Period = 14;
input double RSI_Upper = 70;
input double RSI_Lower = 30;
input int EMA_Fast = 9;
input int EMA_Slow = 21;
input double SL_ATR_Multiplier = 1.0;
input double TP_Multiplier_Input_Input = 2.5;
double TP_Multiplier_Input;
input ENUM_TIMEFRAMES ATR_Timeframe = PERIOD_H1;
input double TrailingStart = 20;
input double TrailingStep = 10;
input double MaxLotSize = 1.0;
input double MinVolatilityATR = 1.5;
input double MinSLPoints = 100;
input int LossStreakLimit = 5;
input int PauseAfterLosses_Minutes = 180;

// === Global Variables ===
CTrade trade;
int emaFastHandle, emaSlowHandle, rsiHandle, macdHandle, atrHandle, adxHandle;
bool handlesReady = false;
datetime lastCalcTime = 0;
double slPoints, tpPoints;
ENUM_ORDER_TYPE direction;
int lossStreak = 0;
datetime lastLossTime = 0;

int tradeCountToday = 0;
double equityStartOfDay = 0;
datetime dayStart = 0;

int OnInit() {
    emaFastHandle = iMA(_Symbol, _Period, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
    emaSlowHandle = iMA(_Symbol, _Period, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
    rsiHandle     = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
    macdHandle    = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
    atrHandle     = iATR(_Symbol, ATR_Timeframe, 14);
    adxHandle     = iADX(_Symbol, _Period, 14);

    handlesReady = (emaFastHandle != INVALID_HANDLE && emaSlowHandle != INVALID_HANDLE &&
                    rsiHandle != INVALID_HANDLE && macdHandle != INVALID_HANDLE &&
                    atrHandle != INVALID_HANDLE && adxHandle != INVALID_HANDLE);

    dayStart = iTime(_Symbol, PERIOD_D1, 0);
    equityStartOfDay = AccountInfoDouble(ACCOUNT_EQUITY);
    return INIT_SUCCEEDED;
}

void OnTick() {
    if (!handlesReady) return;

    if (TimeCurrent() >= dayStart + 86400) {
        tradeCountToday = 0;
        dayStart = iTime(_Symbol, PERIOD_D1, 0);
        equityStartOfDay = AccountInfoDouble(ACCOUNT_EQUITY);
    }

    if (AccountInfoDouble(ACCOUNT_EQUITY) < equityStartOfDay * (1.0 - DailyLossLimit / 100.0)) return;
    if (tradeCountToday >= MaxTradesPerDay) return;

    if (lossStreak >= LossStreakLimit && TimeCurrent() < lastLossTime + PauseAfterLosses_Minutes * 60) {
        Print("🛑 Pause auto après ", lossStreak, " pertes — Reprise à ", TimeToString(lastLossTime + PauseAfterLosses_Minutes * 60));
        return;
    }

    if (PositionSelect(_Symbol)) {
        ApplyTrailingStop();
        if (ShouldExitEarly()) {
            trade.PositionClose(_Symbol);
            Print("[Sentinel] Exit anticipée déclenchée");
        }
        return;
    }

    if (!GenerateSignal()) return;

    double entryPrice = (direction == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double slPrice = (direction == ORDER_TYPE_BUY) ? entryPrice - slPoints * _Point : entryPrice + slPoints * _Point;
    double tp1x = (direction == ORDER_TYPE_BUY) ? entryPrice + slPoints * _Point : entryPrice - slPoints * _Point;
    double tpExtended = (direction == ORDER_TYPE_BUY) ? entryPrice + slPoints * 3.5 * _Point : entryPrice - slPoints * 3.5 * _Point;

    double riskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * (RiskPerTrade / 100.0);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = NormalizeDouble(riskAmount / (slPoints * _Point * tickValue), 2);
    lotSize = MathMin(lotSize, MaxLotSize);

    double marginRequired;
    if (!OrderCalcMargin(direction, _Symbol, lotSize, entryPrice, marginRequired)) return;
    if (AccountInfoDouble(ACCOUNT_FREEMARGIN) < marginRequired) return;

    bool opened = false;
    double halfLot = NormalizeDouble(lotSize / 2.0, 2);

    if (direction == ORDER_TYPE_BUY) {
        opened = trade.Buy(halfLot, _Symbol, 0, slPrice, tp1x);
        opened &= trade.Buy(halfLot, _Symbol, 0, slPrice, tpExtended);
    } else if (direction == ORDER_TYPE_SELL) {
        opened = trade.Sell(halfLot, _Symbol, 0, slPrice, tp1x);
        opened &= trade.Sell(halfLot, _Symbol, 0, slPrice, tpExtended);
    }

    if (opened) {
        tradeCountToday++;
        lossStreak = 0; // reset on success
        Print("[TheSentinel] Split entry: ", DoubleToString(halfLot, 2), "×2 | SL: ", slPrice, " | TP: 1x=", tp1x, " | 3.5x=", tpExtended);
    } else {
        lossStreak++;
        lastLossTime = TimeCurrent();
        Print("⚠️ Trade échoué ou refusé | Pertes consécutives : ", lossStreak);
    }
}

bool GenerateSignal() {
    if (Bars(_Symbol, _Period) < 50) return false;
    datetime currentBarTime = iTime(_Symbol, _Period, 0);
    if (currentBarTime == lastCalcTime) return false;
    lastCalcTime = currentBarTime;

    double emaFast[1], emaSlow[1], rsi[1], macd[1], atr[1], adx[1];
    if (CopyBuffer(emaFastHandle, 0, 0, 1, emaFast) < 1) return false;
    if (CopyBuffer(emaSlowHandle, 0, 0, 1, emaSlow) < 1) return false;
    if (CopyBuffer(rsiHandle, 0, 0, 1, rsi) < 1) return false;
    if (CopyBuffer(macdHandle, 0, 0, 1, macd) < 1) return false;
    if (CopyBuffer(atrHandle, 0, 0, 1, atr) < 1) return false;
    if (CopyBuffer(adxHandle, 0, 0, 1, adx) < 1) return false;

    if (atr[0] < MinVolatilityATR * _Point) return false;
    if (adx[0] < 20.0) {
        Print("⛔ Marché en range détecté (ADX = ", adx[0], ") — pas de trade");
        return false;
    }

    TP_Multiplier_Input = (rsi[0] > RSI_Upper + 10 && macd[0] > 0 && emaFast[0] > emaSlow[0]) ? 3.5 : 2.5;

    if (emaFast[0] > emaSlow[0] && rsi[0] > RSI_Upper && macd[0] > 0) {
        direction = ORDER_TYPE_BUY;
    } else if (emaFast[0] < emaSlow[0] && rsi[0] < RSI_Lower && macd[0] < 0) {
        direction = ORDER_TYPE_SELL;
    } else {
        return false;
    }

    slPoints = MathMax(atr[0] * SL_ATR_Multiplier / _Point, MinSLPoints);
    tpPoints = slPoints * TP_Multiplier_Input;
    return true;
}
