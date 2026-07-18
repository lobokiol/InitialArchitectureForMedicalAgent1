"""Canonical list of unauthenticated routes.

Auth is enforced per-route via ``Depends(get_current_user)``. This module is the
shared reference for public endpoints (docs, future middleware, audits).
"""

PUBLIC_PATHS = {"/healthz", "/ready"}
PUBLIC_PREFIXES = (
    "/auth/wechat/login",
    "/auth/wechat/bind-phone",
    "/auth/register",
    "/auth/login",
    "/auth/refresh",
)
