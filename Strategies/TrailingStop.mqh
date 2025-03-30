//+------------------------------------------------------------------+
//| TrailingStop.mqh                                                 |
//| Gestion simple du trailing stop                                 |
//+------------------------------------------------------------------+
class TrailingStop
{
private:
    double activationDistance; // en pips

public:
    void SetActivation(double pips)
    {
        activationDistance = pips * _Point;
    }

    void Update()
    {
        for (int i = PositionsTotal() - 1; i >= 0; i--)
        {
            if (PositionGetTicket(i) < 0) continue;
            string symbol = PositionGetString(POSITION_SYMBOL);
            if (symbol != _Symbol) continue;

            double price_open = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl = PositionGetDouble(POSITION_SL);
            double volume = PositionGetDouble(POSITION_VOLUME);
            double current_price = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                ? SymbolInfoDouble(symbol, SYMBOL_BID)
                : SymbolInfoDouble(symbol, SYMBOL_ASK);

            double new_sl;
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
            {
                if ((current_price - price_open) > activationDistance)
                {
                    new_sl = current_price - activationDistance;
                    if (sl < new_sl)
                        trade.PositionModify(symbol, new_sl, PositionGetDouble(POSITION_TP));
                }
            }
            else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
            {
                if ((price_open - current_price) > activationDistance)
                {
                    new_sl = current_price + activationDistance;
                    if (sl > new_sl || sl == 0.0)
                        trade.PositionModify(symbol, new_sl, PositionGetDouble(POSITION_TP));
                }
            }
        }
    }
};
