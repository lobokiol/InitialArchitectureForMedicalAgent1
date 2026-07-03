from fastapi import HTTPException, status


def api_error(
    code: str,
    detail: str,
    status_code: int,
    **extra: object,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"detail": detail, "code": code, **extra},
    )


def auth_error(code: str, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return api_error(code, detail, status_code)
