//+------------------------------------------------------------------+
//|                                                  ZmqBridgeEA.mq5 |
//|                                 Copyright 2025, unknowkubbrother |
//|                                                 https://mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, unknowkubbrother"
#property link      "https://mql5.com"
#property version   "1.02"
#property strict

#include <Zmq/Zmq.mqh>

Context context;        // ZMQ Context
Socket *pub;            // สำหรับ PUB → Python SUB
Socket *rep;            // สำหรับ REP ← Python REQ

//--- Input parameters สำหรับกำหนด port จากหน้าต่าง EA
input int PUB_PORT = 5555;
input int REP_PORT = 6000;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Create sockets
    pub = new Socket(context, ZMQ_PUB);
    rep = new Socket(context, ZMQ_REP);
    
    // Bind PUB / REP
    if(!pub.bind("tcp://*:" + IntegerToString(PUB_PORT)))
    {
        Print("Failed to bind PUB socket on port ", PUB_PORT);
        return(INIT_FAILED);
    }
    
    if(!rep.bind("tcp://*:" + IntegerToString(REP_PORT)))
    {
        Print("Failed to bind REP socket on port ", REP_PORT);
        return(INIT_FAILED);
    }

    Print("ZMQ Bridge Started on PUB:", PUB_PORT, " REP:", REP_PORT);

    // ใช้ timer every 1 second เพื่อ publish ราคา
    EventSetTimer(1);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    if(CheckPointer(pub) == POINTER_DYNAMIC)
        delete pub;
    if(CheckPointer(rep) == POINTER_DYNAMIC)
        delete rep;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // ส่งราคาแบบ realtime ทุก tick
    string msg = "PRICE|" + _Symbol + "|" + DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),5);
    pub.send(msg);

    // รับคำสั่งจาก Python
    ZmqMsg request;
    if(rep.recv(request, true))  // true = nowait (non-blocking)
    {
        string req = request.getData();
        Print("Received:", req);

        if(StringFind(req,"BUY")==0)
        {
            rep.send("BUY_OK");
        }
        else if(StringFind(req,"SELL")==0)
        {
            rep.send("SELL_OK");
        }
        else
        {
            rep.send("UNKNOWN");
        }
    }
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    // ทุก 1 วินาที publish ราคาปัจจุบันอีกครั้ง
    string msg = "PRICE|" + _Symbol + "|" + DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),5);
    pub.send(msg);
}

//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
    // สำหรับอนาคต หากต้องการส่งสถานะคำสั่งกลับ Python
}

//+------------------------------------------------------------------+
