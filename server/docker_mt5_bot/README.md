# MT5 Docker + Bot Runner (Single Command + Multi Instance)

โปรเจกต์นี้เอาไว้รัน MT5 และบอทใน container เดียวกัน โดยมี launcher สำหรับรันหลาย instance พร้อมกันได้

## สิ่งที่ต้องมี

- Docker Desktop
- macOS หรือ Linux

สำหรับ Apple Silicon (M1/M2/M3):
- ปิด `Use Rosetta for x86/amd64 emulation on Apple Silicon` ใน Docker Desktop

## โครงหลักที่ใช้จริง

- `start_bot.sh` ตัว orchestrator หลัก (start MT5, login, enable algo, install requirements, start bot)
- `run_instance.sh` ตัวเรียกใช้งานต่อ instance (แยก project/port/container)

## ตั้งค่า

คัดลอกไฟล์ env:

```bash
cp .env.example .env
```

กำหนดค่าอย่างน้อย:

```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
LIVE_SYMBOL=EURUSD
LIVE_TIMEFRAME=H1
BOT_CONFIG_ID=your_bot_config_id
BOT_WS_URL=ws://host.docker.internal:8000/bot/ws
VISION_LLM_API_URL=http://host.docker.internal:8000/vision_llm/
```

ถ้า `CUSTOM_USER/PASSWORD` เว้นว่าง ระบบจะใช้ `MT5_LOGIN/MT5_PASSWORD` ให้อัตโนมัติ
และถ้า `USE_SHARED_PYDEPS=1` ระบบจะติดตั้ง Python dependencies ลง volume กลางครั้งเดียว
แล้วทุก instance reuse ได้ (ไม่ต้องโหลด `torch` ซ้ำทุก user)

## รันแบบ command เดียว

```bash
./run_instance.sh
```

`run_instance.sh` จะ derive ค่าอัตโนมัติจาก env:

- `instance_name = MT5_LOGIN` (เช่น `103853956`)
- `profile = <LIVE_SYMBOL>_<LIVE_TIMEFRAME>` (lowercase เช่น `EURUSD + H1 => eurusd_h1`)
- เลือก bot จากโฟลเดอร์ `bots/<profile>`

## คำสั่งที่ server เรียกได้

เริ่มรัน:

```bash
./run_instance.sh start
```

หยุด:

```bash
./run_instance.sh stop
```

รีสตาร์ท:

```bash
./run_instance.sh restart
```

รองรับระบุ instance เอง:

```bash
./run_instance.sh stop 103853956
./run_instance.sh restart 103853956 eurusd_h1
```

## Server Integration (Run/Stop จาก API)

แนะนำให้ server เรียกแบบระบุ `instance_name = BOT_CONFIG_ID` เพื่อให้ 1 bot = 1 docker project:

```bash
./run_instance.sh start <BOT_CONFIG_ID> <profile>
./run_instance.sh stop <BOT_CONFIG_ID>
./run_instance.sh restart <BOT_CONFIG_ID> <profile>
```

ถ้าต้องการ `pull + start` ทุกครั้งตอนกด Run:

```env
AUTO_BUILD=0
PULL_LATEST_IMAGE=1
METATRADER_IMAGE=your-registry/mt5-bot-image:tag
```

`start_bot.sh` จะ `docker compose pull` ก่อน แล้วค่อย start/restart

ถ้าต้องการบังคับให้ MT5 เปิด `Allow algorithmic trading` ทุกครั้ง:

```env
MT5_FORCE_ALGO_TRADING=1
MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE=0
MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE=0
MT5_EXPERTS_DISABLE_ON_CHART_CHANGE=0
MT5_EXPERTS_DISABLE_VIA_PYTHON_API=0
```

โดย `start_bot.sh` จะพยายาม 2 ชั้น:
- ตั้งค่าใน `common.ini` ([Experts])
- fallback ด้วย UI automation (`Ctrl+E` และเปิด `Options -> Expert Advisors`) ถ้ายังไม่ขึ้น

เพื่อให้เริ่มบอทหลังล็อกอิน MT5 ชัวร์ขึ้น (แนะนำ):

```env
MT5_SKIP_PRECHECKS=0
```

ค่า `0` จะไม่ข้ามขั้น precheck และจะพยายามล็อกอิน MT5 ผ่าน API ก่อนเริ่มบอท

ถ้าเคยมี account/server ghost ใน Navigator (บรรทัดว่างเหนือชื่อ server):

```env
MT5_CLEAN_ACCOUNT_CACHE_ON_START=1
```

ระบบจะล้าง cache ผีของ account id `0` อัตโนมัติ และรีเซ็ตรายการ server/account cache ตอน start

## Docker Outside of Docker (Server อยู่ใน container แต่สร้าง bot container บน host)

ให้ mount docker socket ของ host เข้า container server:

```yaml
services:
  bot-server:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /path/to/project/server/docker_mt5_bot:/opt/mt5-runner
    environment:
      - RUNNER_DIR=/opt/mt5-runner
```

และใน image ของ server ต้องมี Docker CLI (`docker` + `docker compose`)

## รีเซ็ตทั้งระบบแล้วขึ้นใหม่ทีเดียว

ล้างของเก่าทั้งหมดของ MT5 (container/network/volume/image) แล้วขึ้นใหม่ด้วยคำสั่งเดียว:

```bash
./reset_fresh_stack.sh user_a eurusd_h1
```

ถ้าต้องการแค่ล้างและ build ใหม่ แต่ยังไม่ start:

```bash
SKIP_START=1 ./reset_fresh_stack.sh user_a eurusd_h1
```

รองรับ profile ตามโฟลเดอร์ที่มีไฟล์ `run_live.py` ใต้ `bots/`

ดูรายการ instance:

```bash
./run_instance.sh list
```

หยุด instance:

```bash
./run_instance.sh stop 103853956
```

## Reuse Python Libs ข้ามทุก Instance

ระบบใช้ Docker volume กลางชื่อ `mt5_pydeps_shared` สำหรับ dependencies ของบอท

- รอบแรกของเครื่อง: ยังต้องติดตั้ง
- รอบถัดไปของ user อื่น: ข้ามได้ทันทีถ้า requirements hash เดิม

ตรวจว่ามี volume แล้ว:

```bash
docker volume ls | grep mt5_pydeps_shared
```

ถ้าต้องการล้างแล้วติดตั้งใหม่:

```bash
docker volume rm mt5_pydeps_shared
```

## พอร์ตของแต่ละ instance

`run_instance.sh` จะสร้างพอร์ตแยกให้อัตโนมัติและบันทึกไว้ที่ `.instances/<instance>.env`

ตัวอย่างตรวจค่า:

```bash
cat .instances/103853956.env
```

จากนั้นใช้งาน:

- Web VNC: `http://localhost:<MT5_WEB_PORT>`

## สร้าง Snapshot เพื่อลดเวลาติดตั้งรอบแรก

หลังรอบแรกติดตั้ง MT5 เสร็จแล้ว:

```bash
./create_mt5_snapshot.sh
```

หรือกำหนด path เอง:

```bash
./create_mt5_snapshot.sh /Users/unknowkubbrother/Coding/mt5-snapshots/eurusd_h1/mt5-config-snapshot.tgz
```

รอบถัดไปถ้า mount snapshot path ตรง ระบบจะ restore แล้วข้ามขั้นติดตั้งยาว

## Bot logs

ตัวอย่าง follow log ของบอทใน instance:

```bash
docker compose -p mt5_103853956 exec -it metatrader5-macos tail -f /config/103853956_eurusd_h1.log
```

## หมายเหตุสำคัญ

- รอบแรกอาจใช้เวลา 10-30 นาที (Wine + MT5 + python libs)
- ถ้าตลาดปิด ค่า tick/delta บางชุดอาจนิ่งหรือเป็น 0 ได้
- `start_bot.sh` ยังจำเป็น เพราะเป็นตัวทำ workflow ภายในทั้งหมด
