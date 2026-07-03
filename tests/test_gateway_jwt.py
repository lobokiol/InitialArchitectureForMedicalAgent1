import jwt as pyjwt

from app.gateway.jwt import decode_token, issue_token_pair


def test_issue_and_decode_access():
    pair = issue_token_pair("+8613800138000")
    claims = decode_token(pair.access_token, "access")
    assert claims.sub == "+8613800138000"
    assert claims.type == "access"
    assert pair.expires_in == 120 * 60


def test_refresh_has_jti():
    pair = issue_token_pair("+8613800138000")
    claims = decode_token(pair.refresh_token, "refresh")
    assert claims.jti is not None


def test_wrong_type_rejected():
    pair = issue_token_pair("+8613800138000")
    try:
        decode_token(pair.access_token, "refresh")
        assert False, "should raise"
    except pyjwt.InvalidTokenError:
        pass
