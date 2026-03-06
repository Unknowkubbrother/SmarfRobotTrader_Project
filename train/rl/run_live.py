import os
import time
import traceback

try:
    import rpyc

    # MT5 RPC handshake can take >20s under Apple Silicon qemu emulation.
    # Keep a safer default while allowing explicit override.
    default_sync_timeout = max(
        60.0,
        float(int(os.getenv("MT5_RPC_TIMEOUT_MS", "180000")) / 1000.0),
    )
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = float(
        os.getenv("LIVE_MT5_SYNC_TIMEOUT_SEC", str(default_sync_timeout))
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
