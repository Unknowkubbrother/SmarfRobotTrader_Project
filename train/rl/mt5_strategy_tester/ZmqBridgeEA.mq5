//+------------------------------------------------------------------+
//|                                              ZmqBridgeEA.mq5     |
//|                                    RL Trading Bot - ZMQ Bridge    |
//|                    Connects to Python PPO Model via ZeroMQ       |
//+------------------------------------------------------------------+
#property copyright "SmarfRobotTrader"
#property version "1.00"
#property strict

// 🚨 REQUIREMENTS:
// You must install the MetaTrader-ZeroMQ library:
// Download DLLs and MQH files from: https://github.com/dingmaotu/mql-zmq
// Place BOTH libzmq.dll and libsodium.dll in MT5/MQL5/Libraries
// Place Zmq directory in MT5/MQL5/Include

// Force Strategy Tester to copy libsodium.dll to the Agent
#import "libsodium.dll"
#import

#include <Trade\Trade.mqh>
#include <Zmq\Zmq.mqh>

//--- Input Parameters
input string ZmqHost = "127.0.0.1"; // Python ZMQ Server IP
input int ZmqPort = 5555;           // Python ZMQ Server Port
input double LotSize = 0.1;         // Lot Size
input int SL_Pips = 30;             // Stop Loss (pips)
input int TP_Pips = 60;             // Take Profit (pips)
input int MagicNumber = 12345;      // Magic Number

//--- Global Variables
Context context("PPO_ZMQ_Client");
Socket *reqSocket = NULL;
CTrade trade;
datetime lastBarTime = 0;
double lastBid = 0.0;
double lastAsk = 0.0;
int accumDeltaTick = 0;
double accumDeltaPrice = 0.0;
double pipSize = 0.0; // Actual pip size (auto-detected)

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit() {
  trade.SetExpertMagicNumber(MagicNumber);
  trade.SetDeviationInPoints(10);

  // Auto-detect pip size: 5-digit broker → point*10, 4-digit → point
  // EURUSD 5-digit: point=0.00001, pipSize=0.0001
  // USDJPY 3-digit: point=0.001,   pipSize=0.01
  pipSize = _Point * ((_Digits == 3 || _Digits == 5) ? 10 : 1);
  Print("🚀 ZmqBridgeEA initialized | pipSize=", pipSize, " point=", _Point,
        " digits=", _Digits);

  // Create REQ socket for Request-Reply pattern
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
  Print("🎯 SL=", SL_Pips, " pips (", SL_Pips * pipSize, ") | TP=", TP_Pips,
        " pips (", TP_Pips * pipSize, ")");

  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
  if (reqSocket != NULL) {
    delete reqSocket;
    Print("🔌 ZMQ Socket closed");
  }
}

//+------------------------------------------------------------------+
//| Send data and receive action via ZMQ                             |
//+------------------------------------------------------------------+
int GetPPOAction(string data) {
  if (reqSocket == NULL)
    return -1;

  // Send request to Python
  ZmqMsg request(data);
  if (!reqSocket.send(request)) {
    Print("❌ ZMQ send failed");
    return -1;
  }

  // Receive reply from Python
  ZmqMsg reply;
  if (!reqSocket.recv(reply)) {
    Print("❌ ZMQ receive failed");
    return -1;
  }

  string response = reply.getData();
  StringTrimRight(response);
  StringTrimLeft(response);

  int action = (int)StringToInteger(response);
  return action;
}

//+------------------------------------------------------------------+
//| Build OHLC data string for Python                                |
//+------------------------------------------------------------------+
string BuildDataString(int dTick, double dPrice) {
  MqlRates rates[];
  int copied = CopyRates(_Symbol, PERIOD_H1, 1, 80, rates);

  if (copied < 80) {
    Print("⚠️ Only got ", copied, " bars (need 80)");
    return "";
  }

  string data = "";
  for (int i = 0; i < copied; i++) {
    if (i > 0)
      data += "|";
    data += DoubleToString(rates[i].open, _Digits) + "," +
            DoubleToString(rates[i].high, _Digits) + "," +
            DoubleToString(rates[i].low, _Digits) + "," +
            DoubleToString(rates[i].close, _Digits);
  }

  int position = 0;
  double unrealized_pnl = 0;
  double entry_price = 0;

  if (PositionSelect(_Symbol)) {
    long posType = PositionGetInteger(POSITION_TYPE);
    if (posType == POSITION_TYPE_BUY)
      position = 1;
    else if (posType == POSITION_TYPE_SELL)
      position = -1;
    unrealized_pnl = PositionGetDouble(POSITION_PROFIT);
    entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
  }

  double equity = AccountInfoDouble(ACCOUNT_EQUITY);

  static int holdSteps = 0;
  static int lastPos = 0;
  if (position == lastPos && position != 0)
    holdSteps++;
  else
    holdSteps = 0;
  lastPos = position;

  data += ";" + IntegerToString(position) + "," + DoubleToString(equity, 2) +
          "," + DoubleToString(unrealized_pnl, 2) + "," +
          IntegerToString(holdSteps) + "," +
          DoubleToString(entry_price, _Digits) + "," + IntegerToString(dTick) +
          "," + DoubleToString(dPrice, _Digits);

  return data;
}

//+------------------------------------------------------------------+
//| Execute trading action                                           |
//+------------------------------------------------------------------+
void ExecuteAction(int action) {
  double price, sl, tp;

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
    }
    if (!hasPosition || posType == POSITION_TYPE_SELL) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = NormalizeDouble(price - SL_Pips * pipSize, _Digits);
      tp = NormalizeDouble(price + TP_Pips * pipSize, _Digits);
      trade.Buy(LotSize, _Symbol, price, sl, tp, "PPO_BUY");
      Print("📈 BUY @ ", price, " SL:", sl, " TP:", tp);
    }
    break;

  case 2: // SELL
    if (hasPosition && posType == POSITION_TYPE_BUY) {
      trade.PositionClose(_Symbol);
      Sleep(100);
    }
    if (!hasPosition || posType == POSITION_TYPE_BUY) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = NormalizeDouble(price + SL_Pips * pipSize, _Digits);
      tp = NormalizeDouble(price - TP_Pips * pipSize, _Digits);
      trade.Sell(LotSize, _Symbol, price, sl, tp, "PPO_SELL");
      Print("📉 SELL @ ", price, " SL:", sl, " TP:", tp);
    }
    break;

  case 3: // CLOSE
    if (hasPosition) {
      trade.PositionClose(_Symbol);
      Print("🔒 CLOSE position");
    }
    break;
  }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
  double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

  if (lastBid > 0.0 && lastAsk > 0.0) {
    if (bid > lastBid || (bid == lastBid && ask > lastAsk)) {
      accumDeltaTick++;
    } else if (bid < lastBid || (bid == lastBid && ask < lastAsk)) {
      accumDeltaTick--;
    }
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

  int deltaTickToSend = accumDeltaTick;
  double deltaPriceToSend = accumDeltaPrice;

  accumDeltaTick = 0;
  accumDeltaPrice = 0.0;

  string data = BuildDataString(deltaTickToSend, deltaPriceToSend);
  if (data == "")
    return;

  int action = GetPPOAction(data);

  if (action < 0 || action > 3) {
    Print("⚠️ Invalid action: ", action, " → defaulting to HOLD");
    return;
  }

  string actionNames[] = {"HOLD", "BUY", "SELL", "CLOSE"};
  Print("🤖 PPO Action (ZMQ+Delta): ", actionNames[action], " (", action, ")");

  ExecuteAction(action);
}
//+------------------------------------------------------------------+
