#ifndef __RISKMANAGER_MQH__
#define __RISKMANAGER_MQH__

class RiskManager
{
private:
    double riskPercent;

public:
    void SetRiskPercent(double percent)
    {
        riskPercent = percent;
    }

    double CalculateLot(double stopLossInPoints)
    {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double riskAmount = balance * (riskPercent / 100.0);

        double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        double pointValue = (tickValue / tickSize);

        double lotSize = riskAmount / (stopLossInPoints * pointValue);
        lotSize = MathFloor(lotSize / lotStep) * lotStep;

        return NormalizeDouble(lotSize, 2);
    }
};

#endif
