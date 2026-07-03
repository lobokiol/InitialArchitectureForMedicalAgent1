import re

_CN_MOBILE = re.compile(r"^1\d{10}$")


def normalize_phone(raw: str) -> str:
    s = raw.strip().replace(" ", "").replace("-", "")
    if s.startswith("+"):
        if len(s) < 8:
            raise ValueError("invalid phone")
        return s
    if s.startswith("86") and len(s) == 13:
        s = s[2:]
    if _CN_MOBILE.match(s):
        return f"+86{s}"
    raise ValueError(f"invalid phone: {raw!r}")
