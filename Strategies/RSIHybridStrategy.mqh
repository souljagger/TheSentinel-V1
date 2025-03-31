//+------------------------------------------------------------------+
//| RSIHybridStrategy.mqh                                            |
//| Stratégie avancée RSI + EMA + ATR + Risk + SL/TP + Trailing     |
//+------------------------------------------------------------------+
#pragma once
#include "../Classes/RiskManager.mqh"
#include "../Classes/TrailingStop.mqh"
#include <Trade/Trade.mqh>

class RSIHybridStrategy
{
private:
    double rsi;
    double emaFast;
    double emaSlow;
    double atr;
    datetime lastBarTime;

    int handle_rsi;
    int handle_ema_fast;
    int handle_ema_slow;
    int handle_atr;

    CTrade trade;

public:
    RiskManager risk;
    TrailingStop trail;

    void Init()
    {
        lastBarTime = 0;
        risk.SetRiskPercent(1.0); // 1% de risque
        trail.SetActivation(25);  // Trailing actif à +25 pips

        handle_rsi      = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
        handle_ema_fast = iMA(_Symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE);
        handle_ema_slow = iMA(_Symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE);
        handle_atr      = iATR(_Symbol, _Period, 14);
    }

    bool OnNewBar()
    {
        datetime currentBar = iTime(_Symbol, _Period, 0);
        if (currentBar == lastBarTime)
            return false;

        lastBarTime = currentBar;

        double rsiBuffer[1], emaFastBuffer[1], emaSlowBuffer[1], atrBuffer[1];
        if (
            CopyBuffer(handle_rsi, 0, 0, 1, rsiBuffer) < 0 ||
            CopyBuffer(handle_ema_fast, 0, 0, 1, emaFastBuffer) < 0 ||
            CopyBuffer(handle_ema_slow, 0, 0, 1, emaSlowBuffer) < 0 ||
            CopyBuffer(handle_atr, 0, 0, 1, atrBuffer) < 0)
        {
            Print("[ERROR] Failed to copy indicator buffer");
            return false;
        }

        rsi     = rsiBuffer[0];
        emaFast = emaFastBuffer[0];
        emaSlow = emaSlowBuffer[0];
        atr     = atrBuffer[0];

        return true;
    }

    bool CheckBuy()
    {
        return (rsi > 55 && emaFast > emaSlow);
    }

    bool CheckSell()
    {
        return (rsi < 45 && emaFast < emaSlow);
    }

    void ExecuteTrade()
    {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double lot = risk.CalculateLot(atr * 4);
        double sl = atr * 4;
        double tp = atr * 8;

        if (CheckBuy())
        {
            double sl_price = NormalizeDouble(bid - sl * _Point, _Digits);
            double tp_price = NormalizeDouble(bid + tp * _Point, _Digits);
            trade.Buy(lot, _Symbol, ask, sl_price, tp_price, "RSIBuy");
        }
        else if (CheckSell())
        {
            double sl_price = NormalizeDouble(ask + sl * _Point, _Digits);
            double tp_price = NormalizeDouble(ask - tp * _Point, _Digits);
            trade.Sell(lot, _Symbol, bid, sl_price, tp_price, "RSISell");
        }
    }

    void ManageTrailing()
    {
        trail.Update();
    }
};