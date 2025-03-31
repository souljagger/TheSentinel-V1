#ifndef __TRAILINGSTOP_MQH__
#define __TRAILINGSTOP_MQH__

#include <Trade/Trade.mqh>

class TrailingStop
{
private:
    double activationDistance;
    CTrade trade;

public:
    void SetActivation(double pips)
    {
        activationDistance = pips * _Point;
    }

    void Update()
    {
        for (int i = PositionsTotal() - 1; i >= 0; i--)
        {
            if (!PositionGetTicket(i)) continue;

            string symbol = PositionGetString(POSITION_SYMBOL);
            if (symbol != _Symbol) continue;

            long type = PositionGetInteger(POSITION_TYPE);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl = PositionGetDouble(POSITION_SL);
            double tp = PositionGetDouble(POSITION_TP);
            double currentPrice = (type == POSITION_TYPE_BUY)
                ? SymbolInfoDouble(symbol, SYMBOL_BID)
                : SymbolInfoDouble(symbol, SYMBOL_ASK);

            double newSL;
            if (type == POSITION_TYPE_BUY && (currentPrice - openPrice) > activationDistance)
            {
                newSL = NormalizeDouble(currentPrice - activationDistance, _Digits);
                if (sl < newSL || sl == 0.0)
                    trade.PositionModify(symbol, newSL, tp);
            }
            else if (type == POSITION_TYPE_SELL && (openPrice - currentPrice) > activationDistance)
            {
                newSL = NormalizeDouble(currentPrice + activationDistance, _Digits);
                if (sl > newSL || sl == 0.0)
                    trade.PositionModify(symbol, newSL, tp);
            }
        }
    }
};

#endif
