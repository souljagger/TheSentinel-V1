// === Strategies/RSIHybridStrategy.mqh — Optimized & Debug-Light by Quantum Emperor 👑🧠 ===
class RSIHybridStrategy {
private:
    int rsiPeriod;
    double rsiUpper, rsiLower;
    int emaFastPeriod, emaSlowPeriod;
    ENUM_TIMEFRAMES atrTimeframe;
    double slATRMultiplier, tpMultiplier;
    ENUM_ORDER_TYPE signalDirection;
    double slPoints, tpPoints;

    int emaFastHandle;
    int emaSlowHandle;
    int rsiHandle;
    int macdHandle;
    int atrHandle;
    datetime lastCalcTime;
    bool handlesReady;

public:
    void Init(int rsiP, double upper, double lower, int emaFast, int emaSlow,
              ENUM_TIMEFRAMES atrTF, double slMult, double tpMult) {
        rsiPeriod = rsiP;
        rsiUpper = upper;
        rsiLower = lower;
        emaFastPeriod = emaFast;
        emaSlowPeriod = emaSlow;
        atrTimeframe = atrTF;
        slATRMultiplier = slMult;
        tpMultiplier = tpMult;
        lastCalcTime = 0;

        emaFastHandle = iMA(_Symbol, PERIOD_M15, emaFastPeriod, 0, MODE_EMA, PRICE_CLOSE);
        emaSlowHandle = iMA(_Symbol, PERIOD_M15, emaSlowPeriod, 0, MODE_EMA, PRICE_CLOSE);
        rsiHandle     = iRSI(_Symbol, PERIOD_M15, rsiPeriod, PRICE_CLOSE);
        macdHandle    = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
        atrHandle     = iATR(_Symbol, atrTimeframe, 14);

        handlesReady = (emaFastHandle != INVALID_HANDLE &&
                        emaSlowHandle != INVALID_HANDLE &&
                        rsiHandle != INVALID_HANDLE &&
                        macdHandle != INVALID_HANDLE &&
                        atrHandle != INVALID_HANDLE);
    }

    bool GenerateSignal() {
        if (!handlesReady) return false;
        if (Bars(_Symbol, PERIOD_M15) < 50) return false;

        datetime currentBarTime = iTime(_Symbol, PERIOD_M15, 0);
        if (currentBarTime == lastCalcTime) return false; // Avoid recalculating on same candle
        lastCalcTime = currentBarTime;

        double emaFast[1], emaSlow[1], rsi[1], macdMain[1];
        if (CopyBuffer(emaFastHandle, 0, 0, 1, emaFast) < 1) return false;
        if (CopyBuffer(emaSlowHandle, 0, 0, 1, emaSlow) < 1) return false;
        if (CopyBuffer(rsiHandle, 0, 0, 1, rsi) < 1) return false;
        if (CopyBuffer(macdHandle, 0, 0, 1, macdMain) < 1) return false;

        if (emaFast[0] > emaSlow[0] && rsi[0] > rsiUpper && macdMain[0] > 0) {
            signalDirection = ORDER_TYPE_BUY;
            CalculateSLTP();
            return true;
        }
        if (emaFast[0] < emaSlow[0] && rsi[0] < rsiLower && macdMain[0] < 0) {
            signalDirection = ORDER_TYPE_SELL;
            CalculateSLTP();
            return true;
        }

        return false;
    }

    void CalculateSLTP() {
        double atr[1];
        if (CopyBuffer(atrHandle, 0, 0, 1, atr) < 1) return;
        slPoints = atr[0] * slATRMultiplier / _Point;
        tpPoints = slPoints * tpMultiplier;
    }

    double GetSLPoints() { return slPoints; }

    double GetSLPrice() {
        if (signalDirection == ORDER_TYPE_BUY)
            return SymbolInfoDouble(_Symbol, SYMBOL_BID) - slPoints * _Point;
        else
            return SymbolInfoDouble(_Symbol, SYMBOL_ASK) + slPoints * _Point;
    }

    double GetTPPrice() {
        if (signalDirection == ORDER_TYPE_BUY)
            return SymbolInfoDouble(_Symbol, SYMBOL_BID) + tpPoints * _Point;
        else
            return SymbolInfoDouble(_Symbol, SYMBOL_ASK) - tpPoints * _Point;
    }

    ENUM_ORDER_TYPE GetOrderType() { return signalDirection; }
};