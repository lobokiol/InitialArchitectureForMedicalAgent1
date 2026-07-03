from fastapi import HTTPException, status


def auth_error(code: str, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"detail": detail, "code": code})
