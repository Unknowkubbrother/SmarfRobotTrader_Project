import hashlib


MAGIC_NUMBER_MIN = 100_000
MAGIC_NUMBER_MAX = 2_147_000_000


def normalize_magic_number(raw_value) -> int | None:
    try:
        value = int(raw_value)
    except Exception:
        return None
    if value < MAGIC_NUMBER_MIN or value > MAGIC_NUMBER_MAX:
        return None
    return value


def derive_magic_number(account_id: str, bot_instance_id: int, *, salt: int = 0) -> int:
    account = str(account_id or "").strip().lower()
    instance = int(bot_instance_id or 0)
    seed = f"{account}:{instance}:{int(salt)}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    span = MAGIC_NUMBER_MAX - MAGIC_NUMBER_MIN
    value = int(digest[:12], 16) % span
    return MAGIC_NUMBER_MIN + value
