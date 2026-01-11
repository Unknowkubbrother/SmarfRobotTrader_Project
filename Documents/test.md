## 📚 Data Dictionary, Formulas & SQL Queries

ตารางสรุปความหมาย สูตรการคำนวณ และตัวอย่าง SQL Query สำหรับดึงข้อมูล

| หมวดหมู่ (Category) | ชื่อตัวแปร (Variable) | ความหมาย (Definition) | สูตร/ที่มา (Formula) | ตาราง (Table) | SQL Query Example (ใช้ `?` แทน `account_id`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. สถานะพอร์ต**<br>(Real-time) | **`Balance`** | เงินในบัญชี (ปิดออเดอร์แล้ว) | MT5 Direct | `Trading_accounts` | `SELECT balance FROM Trading_accounts WHERE account_id = ?` |
| | **`Equity`** | ทรัพย์สินสุทธิ (รวม Floating) | $Balance + P/L$ | `Trading_accounts` | `SELECT equity FROM Trading_accounts WHERE account_id = ?` |
| | **`Leverage`** | อัตราทด | Config | `Trading_accounts` | `SELECT leverage FROM Trading_accounts WHERE account_id = ?` |
| | **`Margin`** | เงินมัดจำ | $Formula$ | `Trading_accounts` | `SELECT margin FROM Trading_accounts WHERE account_id = ?` |
| | **`Free Margin`** | เงินเหลือเปิดไม้เพิ่ม | $Equity - Margin$ | `Trading_accounts` | `SELECT margin_free FROM Trading_accounts WHERE account_id = ?` |
| | **`Margin Level`** | % ความปลอดภัย | $(Eq/Mg)*100$ | `Trading_accounts` | `SELECT margin_level FROM Trading_accounts WHERE account_id = ?` |
| **2. วัดผลงาน**<br>(Performance) | **`Win Rate`** | อัตราการชนะ (%) | $(Won/Total)*100$ | `Orders_History` | `SELECT (COUNT(CASE WHEN profit > 0 THEN 1 END) * 100.0 / COUNT(*)) FROM Orders_History WHERE account_id = ?` |
| | **`Net Profit`** | กำไรสุทธิรวม | $\sum(Pr+Sw+Cm)$ | `Orders_History` | `SELECT SUM(profit + swap + commission) FROM Orders_History WHERE account_id = ?` |
| | **`Profit Factor`** | สัดส่วนกำไร/ขาดทุน | $GrossPr / \|GrossLoss\|$ | `Orders_History` | `SELECT ABS(SUM(CASE WHEN profit>0 THEN profit ELSE 0 END) / NULLIF(SUM(CASE WHEN profit<0 THEN profit ELSE 0 END),0)) FROM Orders_History WHERE account_id = ?` |
| | **`Max Drawdown`** | การขาดทุนสะสมสูงสุด (%) | $(Peak-Low)/Peak$ | `History` | *(ดู SQL แบบละเอียดด้านล่างตาราง)* |
| | **`Total Trades`** | จำนวนเทรดทั้งหมด | Count All | `Orders_History` | `SELECT COUNT(*) FROM Orders_History WHERE account_id = ?` |
| **3. ต้นทุน**<br>(Costs) | **`Swap`** | ดอกเบี้ยข้ามคืน | MT5 Direct | `Orders_History` | `SELECT SUM(swap) FROM Orders_History WHERE account_id = ?` |
| | **`Commission`** | ค่าธรรมเนียมเทรด | MT5 Direct | `Orders_History` | `SELECT SUM(commission) FROM Orders_History WHERE account_id = ?` |
| **4. การแจ้งเตือน**<br>(Alerts) | **`Margin Alert`** | จุดเตือนความปลอดภัย | User Input | `Notify_Configs` | `SELECT alert_margin_level_threshold FROM Notification_Configs WHERE user_id = ?` |
| | **`Profit Target`** | เป้ากำไรรายวัน | User Input | `Notify_Configs` | `SELECT alert_daily_profit_target FROM Notification_Configs WHERE user_id = ?` |
| | **`Loss Limit`** | ลิมิตขาดทุนรายวัน | User Input | `Notify_Configs` | `SELECT alert_daily_loss_limit FROM Notification_Configs WHERE user_id = ?` |

---

### 📝 Complex SQL Queries (ส่วนขยาย)

เนื่องจาก Query บางตัวมีความซับซ้อนเกินกว่าจะใส่ในตารางได้ จึงขอแยกออกมาแสดงผลดังนี้:

**SQL for Max Drawdown Calculation:**
```sql
WITH BalanceCurve AS (
    SELECT close_time, SUM(profit) OVER (ORDER BY close_time) as running_balance
    FROM Orders_History WHERE account_id = ?
),
Peaks AS (
    SELECT running_balance, MAX(running_balance) OVER (ORDER BY close_time) as peak
    FROM BalanceCurve
)
SELECT MAX((peak - running_balance) / peak * 100) as mdd_percent FROM Peaks;