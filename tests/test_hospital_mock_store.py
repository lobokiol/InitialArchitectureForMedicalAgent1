from hospital_mcp.adapters.mock_store import (
    doctors_for_department,
    intro_for_department,
    route_for_department,
)


def test_doctors_for_department():
    docs = doctors_for_department("骨科")
    assert len(docs) == 3
    assert docs[0]["name"].startswith("骨科")


def test_intro_known_department():
    intro = intro_for_department("消化内科")
    assert intro["department"] == "消化内科"
    assert "summary" in intro
    assert "error" not in intro


def test_intro_unknown_department():
    intro = intro_for_department("未知科")
    assert intro["error"] == "department_not_found"


def test_route_known_department():
    route = route_for_department("骨科")
    assert route["department"] == "骨科"
    assert len(route["steps"]) >= 1


def test_route_unknown_department():
    route = route_for_department("未知科")
    assert route["error"] == "department_not_found"
