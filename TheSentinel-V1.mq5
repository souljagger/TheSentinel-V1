//+------------------------------------------------------------------+
//|                                                      TheSentinel-V1.mq5 |
//| Expert Advisor principal                                          |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
#include "Strategies/RSIHybridStrategy.mqh"

input ENUM_TIMEFRAMES Timeframe = PERIOD_H1;

RSIHybridStrategy strategy;
CTrade trade;

int OnInit()
{
    strategy.Init();
    return INIT_SUCCEEDED;
}

void OnTick()
{
    if (!strategy.OnNewBar())
        return;

    // Si aucune position ouverte
    if (!PositionSelect(_Symbol))
    {
        strategy.ExecuteTrade();
    }
    else
    {
        strategy.ManageTrailing();
    }
}
