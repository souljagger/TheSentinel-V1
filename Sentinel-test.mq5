//+------------------------------------------------------------------+
//| Sentinel RSI Hybrid EA - Optimisation Ready                     |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>
CTrade trade;

// === INPUTS STRATÉGIE ===
input double   RiskPercent        = 1.0;
input int      RSI_Period         = 14;
input double   RSI_Overbought     = 70.0;
input double   RSI_Oversold       = 30.0;
input int      MA_Fast_Period     = 50;
input int      MA_Slow_Period     = 200;
input int      ATR_Period         = 14;
input double   SL_ATR_Multiplier  = 1.5;
input double   TP_Multiplier      = 2.0;
input double   MinVolumeMultiplier= 1.2;
input ENUM_TIMEFRAMES SignalTF    = PERIOD_H1;

// === INPUTS GESTION POSITIONS ===
input bool     EnableBreakEven    = true;
input double   BreakEvenTrigger   = 1.0;
input double   BreakEvenOffset    = 5.0;
input bool     EnableTrailing     = true;
input double   TrailingStopATR    = 1.0;

// === INPUTS SÉCURITÉ CAPITAL ===
input double   MaxDrawdownPercent = 10.0;
input double   DailyLossLimit     = 5.0;
input int      MaxTradesPerDay    = 3;

// === VARIABLES DE SUIVI ===
double initialEquity;
double dailyLoss = 0;
int tradesToday = 0;
datetime lastTradeDate = 0;

int OnInit()
{
   initialEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   lastTradeDate = TimeCurrent();
   return INIT_SUCCEEDED;
}

void OnTick()
{
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double ddPercent = 100.0 * (1.0 - (currentEquity / initialEquity));
   if(ddPercent >= MaxDrawdownPercent)
      return;

   MqlDateTime currentTimeStruct, lastTradeTimeStruct;
   TimeToStruct(TimeCurrent(), currentTimeStruct);
   TimeToStruct(lastTradeDate, lastTradeTimeStruct);

   if(currentTimeStruct.day != lastTradeTimeStruct.day)
   {
      dailyLoss = 0;
      tradesToday = 0;
      lastTradeDate = TimeCurrent();
   }

   if(dailyLoss >= DailyLossLimit || tradesToday >= MaxTradesPerDay)
      return;

   static datetime lastSignalTime = 0;
   datetime currentBarTime = iTime(_Symbol, SignalTF, 0);
   if(currentBarTime == lastSignalTime) return;
   lastSignalTime = currentBarTime;

   if(PositionsTotal() > 0){ ManageOpenPositions(); return; }

   int handleRSI = iRSI(_Symbol, SignalTF, RSI_Period, PRICE_CLOSE);
   int handleMAFast = iMA(_Symbol, SignalTF, MA_Fast_Period, 0, MODE_EMA, PRICE_CLOSE);
   int handleMASlow = iMA(_Symbol, SignalTF, MA_Slow_Period, 0, MODE_EMA, PRICE_CLOSE);
   int handleATR = iATR(_Symbol, SignalTF, ATR_Period);
   if(handleRSI==INVALID_HANDLE || handleMAFast==INVALID_HANDLE || handleMASlow==INVALID_HANDLE || handleATR==INVALID_HANDLE)
      return;

   double rsi[], maFast[], maSlow[], atr[];
   if(CopyBuffer(handleRSI, 0, 0, 1, rsi)<=0) return;
   if(CopyBuffer(handleMAFast, 0, 0, 1, maFast)<=0) return;
   if(CopyBuffer(handleMASlow, 0, 0, 1, maSlow)<=0) return;
   if(CopyBuffer(handleATR, 0, 0, 1, atr)<=0) return;

   double volume = iVolume(_Symbol, SignalTF, 1);
   double avgVol = 0.0;
   for(int i=1; i<=20; i++) avgVol += iVolume(_Symbol, SignalTF, i);
   avgVol /= 20.0;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl, tp;
   double lot = CalcLotSize(atr[0] * SL_ATR_Multiplier);

   if(maFast[0] > maSlow[0] && rsi[0] > 55 && volume > avgVol * MinVolumeMultiplier)
   {
      sl = ask - (atr[0] * SL_ATR_Multiplier);
      tp = ask + (atr[0] * SL_ATR_Multiplier * TP_Multiplier);
      if(trade.Buy(lot,_Symbol,ask,sl,tp,"Trend Buy")) tradesToday++;
      return;
   }
   if(rsi[0] > RSI_Overbought && maFast[0] < maSlow[0])
   {
      sl = bid + atr[0] * SL_ATR_Multiplier;
      tp = bid - atr[0] * SL_ATR_Multiplier * TP_Multiplier;
      if(trade.Sell(lot,_Symbol,bid,sl,tp,"RSI Sell")) tradesToday++;
      return;
   }
   if(rsi[0] < RSI_Oversold && maFast[0] > maSlow[0])
   {
      sl = ask - atr[0] * SL_ATR_Multiplier;
      tp = ask + atr[0] * SL_ATR_Multiplier * TP_Multiplier;
      if(trade.Buy(lot,_Symbol,ask,sl,tp,"RSI Buy")) tradesToday++;
      return;
   }
}

void ManageOpenPositions()
{
   // Gestion BreakEven & Trailing Stop à ajouter ici si besoin
}

double CalcLotSize(double slPips)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = equity * RiskPercent / 100.0;
   double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   return(NormalizeDouble(riskAmount / (slPips/_Point * pipValue), 2));
}
