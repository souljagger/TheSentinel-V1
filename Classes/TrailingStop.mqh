#include <Trade/Trade.mqh>

// === Classes/TrailingStop.mqh — Compatible & Refined by Quantum Emperor 👑🔧 ===
class TrailingStop {
private:
    double trailStart; // Minimum profit in points before trailing starts
    double trailStep;  // Distance of trailing step in points
    CTrade *trade;     // Pointer to global trade object (set externally)

public:
    // Initialization method
    void Init(double start, double step) {
        trailStart = start;
        trailStep = step;
    }

    // Link the global trade object
    void SetTradePointer(CTrade *t) {
        trade = t;
    }

    // Update the stop loss according to trailing logic
    void Update() {
        if (!PositionSelect(_Symbol)) return;
        if (trade == NULL) return;

        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);
        long positionType = PositionGetInteger(POSITION_TYPE);

        double marketPrice = (positionType == POSITION_TYPE_BUY)
                             ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                             : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        double profitPoints = (positionType == POSITION_TYPE_BUY)
                              ? (marketPrice - openPrice) / _Point
                              : (openPrice - marketPrice) / _Point;

        if (profitPoints < trailStart) return;

        double newSL = (positionType == POSITION_TYPE_BUY)
                       ? marketPrice - trailStep * _Point
                       : marketPrice + trailStep * _Point;

        bool shouldModify = (positionType == POSITION_TYPE_BUY && newSL > currentSL) ||
                            (positionType == POSITION_TYPE_SELL && (newSL < currentSL || currentSL == 0.0));

        if (shouldModify) {
            if (trade.PositionModify(_Symbol, newSL, currentTP)) {
                Print("[TrailingStop] SL updated to ", DoubleToString(newSL, _Digits));
            } else {
                Print("[TrailingStop] Failed to update SL: ", GetLastError());
            }
        }
    }
};