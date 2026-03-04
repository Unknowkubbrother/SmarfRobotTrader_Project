import os
import time
import traceback

try:
    import rpyc

    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = float(
        os.getenv("LIVE_MT5_SYNC_TIMEOUT_SEC", "20")
    )
except Exception:
    pass


if __name__ == "__main__":
    max_boot_retries = max(1, int(os.getenv("LIVE_BOOT_RETRIES", "60")))
    boot_retry_seconds = max(1.0, float(os.getenv("LIVE_BOOT_RETRY_SECONDS", "3")))

    for attempt in range(1, max_boot_retries + 1):
        try:
            from live.bot import main

            main()
            break
        except Exception:
            traceback.print_exc()
            if attempt >= max_boot_retries:
                raise
            print(
                f" [BOOT] run_live import/start failed ({attempt}/{max_boot_retries}), "
                f"retrying in {boot_retry_seconds:.1f}s..."
            )
            time.sleep(boot_retry_seconds)
