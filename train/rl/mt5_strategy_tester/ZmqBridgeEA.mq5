//+------------------------------------------------------------------+
//|                                              ZmqBridgeEA.mq5     |
//|                                    RL Trading Bot - ZMQ Bridge    |
//|                    Connects to Python PPO Model via ZeroMQ       |
//+------------------------------------------------------------------+
#property copyright "SmarfRobotTrader"
#property version "2.00"
#property strict

// Force Strategy Tester to copy libsodium.dll to the Agent
#import "libsodium.dll"
#import

#include <Trade\Trade.mqh>
#include <Zmq\Zmq.mqh>

//--- Input Parameters
input string ZmqHost = "127.0.0.1";
input int ZmqPort = 5555;
input int MagicNumber = 12345;
input double RiskPercent = 1.0; // Risk % per trade
input int SL_Pips_Input = 50;   // SL in pips (for auto lot calculation)

//--- Global Variables
Context context("PPO_ZMQ_Client");
Socket *reqSocket = NULL;
CTrade trade;
datetime lastBarTime = 0;
double lastBid = 0.0;
double lastAsk = 0.0;
int accumDeltaTick = 0;
double accumDeltaPrice = 0.0;
double currentLot = 0.0; // Auto-calculated lot size

//+------------------------------------------------------------------+
double CalculateLotSize() {
  double balance = AccountInfoDouble(ACCOUNT_BALANCE);
  double riskAmount = balance * RiskPercent / 100.0;

  // Get pip value per 1 standard lot
  double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
  double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
  double pipSize = (_Digits == 5 || _Digits == 3) ? tickSize * 10 : tickSize;
  double pipValuePerLot = tickValue * (pipSize / tickSize);

  // Lot = Risk$ / (SL_pips × pip_value_per_lot)
  double lotCalc = riskAmount / (SL_Pips_Input * pipValuePerLot);

  // Normalize to broker limits
  double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
  double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
  double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

  lotCalc = MathFloor(lotCalc / lotStep) * lotStep;
  lotCalc = MathMax(minLot, MathMin(maxLot, lotCalc));

  return NormalizeDouble(lotCalc, 2);
}

//+------------------------------------------------------------------+
int OnInit() {
  trade.SetExpertMagicNumber(MagicNumber);
  trade.SetDeviationInPoints(10);

  Print("🚀 ZmqBridgeEA v2.0 | SL/TP managed by Python server");

  reqSocket = new Socket(context, ZMQ_REQ);
  if (reqSocket == NULL) {
    Print("❌ Failed to create ZMQ socket");
    return INIT_FAILED;
  }

  string endpoint = "tcp://" + ZmqHost + ":" + IntegerToString(ZmqPort);
  Print("📡 Connecting to ZMQ: ", endpoint);

  if (!reqSocket.connect(endpoint)) {
    Print("❌ Failed to connect to ZMQ endpoint: ", endpoint);
    return INIT_FAILED;
  }

  Print("✅ ZMQ Connection established");
  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
  if (reqSocket != NULL) {
    delete reqSocket;
    Print("🔌 ZMQ Socket closed");
  }
}

//+------------------------------------------------------------------+
int GetPPOAction(string data) {
  if (reqSocket == NULL)
    return -1;

  ZmqMsg request(data);
  if (!reqSocket.send(request)) {
    Print("❌ ZMQ send failed");
    return -1;
  }

  ZmqMsg reply;
  if (!reqSocket.recv(reply)) {
    Print("❌ ZMQ receive failed");
    return -1;
  }

  string response = reply.getData();
  StringTrimRight(response);
  StringTrimLeft(response);
  return (int)StringToInteger(response);
}

//+------------------------------------------------------------------+
string BuildDataString(int dTick, double dPrice) {
  MqlRates rates[];
  int copied = CopyRates(_Symbol, PERIOD_H1, 1, 200, rates);

  if (copied < 200) {
    Print("⚠️ Only got ", copied, " bars (need 200)");
    return "";
  }

  string data = "";
  for (int i = 0; i < copied; i++) {
    if (i > 0)
      data += "|";
    int barTs = (int)rates[i].time;
    data += IntegerToString(barTs) + "," +
            DoubleToString(rates[i].open, _Digits) + "," +
            DoubleToString(rates[i].high, _Digits) + "," +
            DoubleToString(rates[i].low, _Digits) + "," +
            DoubleToString(rates[i].close, _Digits);
  }

  // Send MT5 position info (for sync detection)
  int position = 0;
  double unrealized_pnl = 0;
  double entry_price = 0;
  double position_volume = 0.0;

  if (PositionSelect(_Symbol)) {
    long posType = PositionGetInteger(POSITION_TYPE);
    if (posType == POSITION_TYPE_BUY)
      position = 1;
    else if (posType == POSITION_TYPE_SELL)
      position = -1;
    unrealized_pnl = PositionGetDouble(POSITION_PROFIT);
    entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
    position_volume = PositionGetDouble(POSITION_VOLUME);
  }

  double equity = AccountInfoDouble(ACCOUNT_EQUITY);

  data += ";" + IntegerToString(position) + "," + DoubleToString(equity, 2) +
          "," + DoubleToString(unrealized_pnl, 2) + ",0," +
          DoubleToString(entry_price, _Digits) + "," + IntegerToString(dTick) +
          "," + DoubleToString(dPrice, _Digits) + "," +
          DoubleToString(currentLot, 2) + "," +
          DoubleToString(position_volume, 2);

  return data;
}

//+------------------------------------------------------------------+
void ExecuteAction(int action) {
  double price;

  bool hasPosition = PositionSelect(_Symbol);
  long posType = -1;
  if (hasPosition)
    posType = PositionGetInteger(POSITION_TYPE);

  switch (action) {
  case 0: // HOLD
    break;

  case 1: // BUY
    if (hasPosition && posType == POSITION_TYPE_SELL) {
      trade.PositionClose(_Symbol);
      Sleep(100);
      hasPosition = PositionSelect(_Symbol);
      posType = hasPosition ? PositionGetInteger(POSITION_TYPE) : -1;
    }
    if (!hasPosition || posType == POSITION_TYPE_BUY) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      // No SL/TP on order — Python server manages SL/TP
      trade.Buy(currentLot, _Symbol, price, 0, 0, "PPO_BUY");
      Print("📈 BUY @ ", price, " | Lot: ", currentLot, " (auto-sized)");
    }
    break;

  case 2: // SELL
    if (hasPosition && posType == POSITION_TYPE_BUY) {
      trade.PositionClose(_Symbol);
      Sleep(100);
      hasPosition = PositionSelect(_Symbol);
      posType = hasPosition ? PositionGetInteger(POSITION_TYPE) : -1;
    }
    if (!hasPosition || posType == POSITION_TYPE_SELL) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      trade.Sell(currentLot, _Symbol, price, 0, 0, "PPO_SELL");
      Print("📉 SELL @ ", price, " | Lot: ", currentLot, " (auto-sized)");
    }
    break;

  case 3: // CLOSE_ONE (net-position mode => close all)
  case 4: // CLOSE_ALL
    if (hasPosition) {
      trade.PositionClose(_Symbol);
      Print("🔒 CLOSE position");
    }
    break;
  }
}

//+------------------------------------------------------------------+
void OnTick() {
  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
  double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

  if (lastBid > 0.0 && lastAsk > 0.0) {
    if (bid > lastBid || (bid == lastBid && ask > lastAsk))
      accumDeltaTick++;
    else if (bid < lastBid || (bid == lastBid && ask < lastAsk))
      accumDeltaTick--;
    accumDeltaPrice += (bid - lastBid) + (ask - lastAsk);
  }
  lastBid = bid;
  lastAsk = ask;

  datetime currentBarTime = iTime(_Symbol, PERIOD_H1, 0);
  if (currentBarTime == lastBarTime || lastBarTime == 0) {
    if (lastBarTime == 0)
      lastBarTime = currentBarTime;
    return;
  }
  lastBarTime = currentBarTime;

  // Recalculate lot size each bar (adapts to balance changes)
  currentLot = CalculateLotSize();

  int deltaTickToSend = accumDeltaTick;
  double deltaPriceToSend = accumDeltaPrice;
  accumDeltaTick = 0;
  accumDeltaPrice = 0.0;

  string data = BuildDataString(deltaTickToSend, deltaPriceToSend);
  if (data == "")
    return;

  int action = GetPPOAction(data);

  if (action < 0 || action > 4) {
    Print("⚠️ Invalid action: ", action);
    return;
  }

  string actionNames[] = {"HOLD", "BUY", "SELL", "CLOSE_ONE", "CLOSE_ALL"};
  Print("🤖 PPO Action: ", actionNames[action], " (", action, ")");

  ExecuteAction(action);
}
//+------------------------------------------------------------------+
