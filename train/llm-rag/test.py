import os
from datetime import datetime

folder = "../data_images/images"

def parse_dt_from_filename(name: str):
    try:
        base = os.path.splitext(name)[0]
        dt_str = base.split("_", 1)[1]
        return datetime.strptime(dt_str, "%Y.%m.%d %H.%M")
    except Exception:
        return None

items = []
bad = 0

for name in os.listdir(folder):
    full = os.path.join(folder, name)
    if not os.path.isfile(full):
        continue

    dt = parse_dt_from_filename(name)
    if dt is None:
        bad += 1
        continue

    items.append((dt, name))

items.sort(key=lambda x: x[0])

print("files parsed:", len(items), "bad names:", bad)
print("\n--- oldest 20 ---")
for dt, name in items[:20]:
    print(dt.strftime("%Y-%m-%d %H:%M"), name)

# print("\n--- newest 20 ---")
# for dt, name in items[-20:]:
#     print(dt.strftime("%Y-%m-%d %H:%M"), name)

# for dt, name in items:
#     print(dt.strftime("%Y-%m-%d %H:%M"), name)
