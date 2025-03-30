//+------------------------------------------------------------------+
//| TrailingStop.mqh                                                 |
//| Gestion simple du trailing stop                                 |
//+------------------------------------------------------------------+
#pragma once
#include <Trade/Trade.mqh>

class TrailingStop
{
private:
    double activationDistance;  // Distance en pips (convertie en points)
    CTrade trade;               // Gestionnaire d'ordres

public:
    // Définir la distance d’activation du trailing (en pips)
    void SetActivation(double pips)
    {
        activationDistance = pips * _Point;
    }

    // Met à jour le trailing stop sur toutes les positions de l'instrument courant
    void Update()
    {
        for (int i = PositionsTotal() - 1; i >= 0; i--)
        {
            if (!PositionGetTicket(i)) continue;

            string symbol = PositionGetString(POSITION_SYMBOL);
            if (symbol != _Symbol) continue;

            long   type       = PositionGetInteger(POSITION_TYPE);
            double openPrice  = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl         = PositionGetDouble(POSITION_SL);
            double tp         = PositionGetDouble(POSITION_TP);
            double currentPrice = (type == POSITION_TYPE_BUY)
                ? SymbolInfoDouble(symbol, SYMBOL_BID)
                : SymbolInfoDouble(symbol, SYMBOL_ASK);

            double newSL;

            if (type == POSITION_TYPE_BUY)
            {
                if ((currentPrice - openPrice) > activationDistance)
                {
                    newSL = currentPrice - activationDistance;
                    if (sl < newSL || sl == 0.0)
                        trade.PositionModify(symbol, NormalizeDouble(newSL, _Digits), tp);
                }
            }
            else if (type == POSITION_TYPE_SELL)
            {
                if ((openPrice - currentPrice) > activationDistance)
                {
                    newSL = currentPrice + activationDistance;
                    if (sl > newSL || sl == 0.0)
                        trade.PositionModify(symbol, NormalizeDouble(newSL, _Digits), tp);
                }
            }
        }
    }
};
