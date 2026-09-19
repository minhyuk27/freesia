"""텔레그램 알림 (freesia 전용 봇). 토큰·채팅 ID는 ~/freesia/.env 에만 둔다(git 제외).
- FREESIA_TG_TOKEN: 봇 토큰, FREESIA_TG_CHAT: 채팅 ID (없으면 봇에 온 최근 메시지에서 자동 탐색해 .env 에 저장)
- 실패는 삼키지 않고 False 와 사유를 반환한다(조용한 실패 방지)."""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def _load():
    env = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("FREESIA_TG_")})
    return env


def _call(token, method, **params):
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode() if params else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20) as r:
        return json.load(r)


def find_chat(token):
    res = _call(token, "getUpdates").get("result", [])
    for u in reversed(res):
        m = u.get("message") or u.get("edited_message") or {}
        if m.get("chat", {}).get("id"):
            return str(m["chat"]["id"])
    return None


def send(text):
    """반환: (성공 여부, 사유)"""
    env = _load()
    token = env.get("FREESIA_TG_TOKEN")
    if not token:
        return False, "토큰 없음(.env)"
    chat = env.get("FREESIA_TG_CHAT")
    try:
        if not chat:
            chat = find_chat(token)
            if not chat:
                return False, "채팅 ID 없음: 봇에게 먼저 메시지를 보내세요"
            with ENV.open("a") as f:
                f.write(f"FREESIA_TG_CHAT={chat}\n")
        r = _call(token, "sendMessage", chat_id=chat, text=text[:3900])
        return bool(r.get("ok")), "ok" if r.get("ok") else str(r)
    except Exception as e:  # noqa: BLE001
        return False, repr(e)[:200]


if __name__ == "__main__":
    import sys
    ok, why = send(" ".join(sys.argv[1:]) or "freesia 알림 테스트")
    print("보냄" if ok else "실패", why)
    sys.exit(0 if ok else 1)
