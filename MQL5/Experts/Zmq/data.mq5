//+------------------------------------------------------------------+
//|                                                Auto_Snapshot.mq5 |
//|                                   Script for automated capturing |
//+------------------------------------------------------------------+
#property script_show_inputs

//--- Input parameters
input datetime StartDate = D'2015.01.01 00:00'; // วันที่เริ่มแคป
input datetime EndDate   = D'2015.01.02 00:00'; // วันที่สิ้นสุด (ลองเทสช่วงสั้นๆ ก่อน)
input int      Width     = 1920;                // ความกว้างรูป
input int      Height    = 1080;                // ความสูงรูป
input int      SleepTime = 200;                 // เวลาพักรอให้กราฟโหลด (มิลลิวินาที)

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   // 1. ปิด Auto Scroll ก่อน ไม่งั้นกราฟดีดกลับ
   ChartSetInteger(0, CHART_AUTOSCROLL, false);
   
   // 2. ลูปตามเวลา (เพิ่มทีละ 1 ชั่วโมง = 3600 วินาที)
   for(datetime t = StartDate; t <= EndDate; t += 3600)
     {
      // --- วาร์ปไปยังวันที่และเวลา t ---
      // CHART_GO_TO_DATE : สั่งกราฟวิ่งไปหาเวลานั้น
      ChartNavigate(0, CHART_GO_TO_DATE, t);
      
      // --- สำคัญมาก: ต้องรอให้กราฟเรนเดอร์ ---
      // ถ้าคอมช้า หรือเน็ตช้า อาจต้องเพิ่ม SleepTime
      Sleep(SleepTime); 
      ChartRedraw(0);

      // --- ตั้งชื่อไฟล์ตามเวลา ---
      // ชื่อไฟล์จะออกมาเป็น: EURUSD_2015.01.01_09.00.png
      string filename = _Symbol + "_" + TimeToString(t, TIME_DATE|TIME_MINUTES);
      StringReplace(filename, ":", "."); // Windows ตั้งชื่อไฟล์มี : ไม่ได้
      
      // --- สั่งแคปรูป ---
      if(ChartScreenShot(0, filename + ".png", Width, Height))
        {
         Print("Saved: ", filename);
        }
      else
        {
         Print("Error saving: ", GetLastError());
        }
     }
     
   Print("Done! Completed.");
  }
//+------------------------------------------------------------------+