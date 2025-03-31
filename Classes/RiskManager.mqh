// === Classes/RiskManager.mqh — Final Form by Quantum Emperor 👑🧮 ===
class RiskManager {
private:
    double riskPercent;
    double maxDailyLossPercent;

public:
    void Init(double risk, double dailyLoss) {
        riskPercent = risk;
        maxDailyLossPercent = dailyLoss;
    }

    // Calculate lot size based on stop loss in points
    double CalculateLotSize(double slPoints) {
        if (slPoints <= 0.0) return 0.01;

        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double riskAmount = balance * (riskPercent / 100.0);

        double tickValue;
        if (!SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE, tickValue)) return 0.01;

        double lot = (riskAmount / (slPoints * _Point * tickValue));
        double minLot, maxLot, lotStep;
        SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN, minLot);
        SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX, maxLot);
        SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP, lotStep);

        long volumeDigits;
        SymbolInfoInteger(_Symbol, SYMBOL_DIGITS, volumeDigits);

        lot = MathMax(minLot, MathMin(maxLot, lot));
        lot = NormalizeDouble(lot, (int)volumeDigits);
        lot = MathFloor(lot / lotStep) * lotStep;

        return lot;
    }

    // Check if current drawdown exceeds allowed daily limit
    bool CheckDailyLossLimit(double startingEquity) {
        double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        double loss = startingEquity - currentEquity;
        double lossPercent = 100.0 * loss / startingEquity;

        if (lossPercent >= maxDailyLossPercent) {
            Print("[RiskManager] Daily loss limit exceeded: ", DoubleToString(lossPercent, 2), "%");
            return false;
        }
        return true;
    }
};
