//+------------------------------------------------------------------+
//| RSIHybridStrategy.mqh                                            |
//| Stratégie avancée RSI + EMA + ATR + Risk + SL/TP + Trailing     |
//+------------------------------------------------------------------+
#include <Classes/RiskManager.mqh>
#include <Classes/TrailingStop.mqh>

class RSIHybridStrategy
{
private:
    double rsi;
    double emaFast;
    double emaSlow;
    double atr;
    datetime lastBarTime;

public:
    RiskManager risk;
    TrailingStop trail;

    // Initialisation
    void Init()
    {
        lastBarTime = 0;
        risk.SetRiskPercent(1.0); // 1% de risque par trade
        trail.SetActivation(25); // Trailing actif à +25 pips
    }

    // Mise à jour sur nouvelle bougie
    bool OnNewBar()
    {
        datetime currentBar = iTime(_Symbol, _Period, 0);
        if (currentBar == lastBarTime)
            return false;

        lastBarTime = currentBar;
        rsi     = iRSI(_Symbol, _Period, 14, PRICE_CLOSE, 0);
        emaFast = iMA(_Symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE, 0);
        emaSlow = iMA(_Symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE, 0);
        atr     = iATR(_Symbol, _Period, 14, 0);
        return true;
    }

    // Condition d'achat
    bool CheckBuy()
    {
        return (rsi > 55 && emaFast > emaSlow);
    }

    // Condition de vente
    bool CheckSell()
    {
        return (rsi < 45 && emaFast < emaSlow);
    }

    // Exécution des ordres avec SL/TP
    void ExecuteTrade()
    {
        double lot = risk.CalculateLot(atr * 4); // SL = 4xATR
        double sl = atr * 4;
        double tp = atr * 8;

        if (CheckBuy())
        {
            double sl_price = NormalizeDouble(Bid - sl * _Point, _Digits);
            double tp_price = NormalizeDouble(Bid + tp * _Point, _Digits);
            trade.Buy(lot, _Symbol, Ask, sl_price, tp_price, "RSIBuy");
        }
        else if (CheckSell())
        {
            double sl_price = NormalizeDouble(Ask + sl * _Point, _Digits);
            double tp_price = NormalizeDouble(Ask - tp * _Point, _Digits);
            trade.Sell(lot, _Symbol, Bid, sl_price, tp_price, "RSISell");
        }
    }

    // Gestion du trailing
    void ManageTrailing()
    {
        trail.Update();
    }
};
