import os
import httpx

TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "1x0000000000000000000000000000000AA")

async def verify_turnstile(token: str, ip: str = None) -> bool:
    if not token:
        return False
        
    
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
    }
    if ip:
        payload["remoteip"] = ip
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            result = response.json()
            return result.get("success", False)
    except Exception as e:
        print(f"Turnstile verification error: {e}")
        return False
