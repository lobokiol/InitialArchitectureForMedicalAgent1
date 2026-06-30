from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models import AppState, DepartmentIntro, DepartmentRoute
from app.graph.nodes.mcp_followup import mcp_followup_node
from app.mcp.followup import FollowupToolPick


def test_mcp_followup_intro_reply(monkeypatch):
    intro = DepartmentIntro(
        department="消化内科",
        summary="诊治胃肠疾病",
        scope=["胃炎"],
        visit_tips="空腹就诊",
        floor="门诊楼 3 层",
        phone="0571-00001111",
    )

    class FakeStructured:
        def invoke(self, _msgs):
            return FollowupToolPick(tool_name="get_department_intro", reason="介绍")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructured()

        def invoke(self, _msgs):
            return AIMessage(content="## 消化内科介绍\n诊治胃肠疾病")

    monkeypatch.setattr("app.core.llm.get_chat_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        "app.mcp.followup.asyncio_run_list_tools",
        lambda: [type("T", (), {"name": "get_department_intro", "description": ""})()],
    )
    monkeypatch.setattr("app.mcp.followup.fetch_department_intro_sync", lambda dept: intro)

    state = AppState(
        last_recommended_department="消化内科",
        messages=[HumanMessage(content="这个科看什么？")],
    )
    result = mcp_followup_node(state)
    assert result["messages"][0].content.startswith("##")
    assert result["tool_call_result"]["tool"] == "get_department_intro"


def test_mcp_followup_user_dept_overrides_last_recommended(monkeypatch):
    route = DepartmentRoute(
        department="骨科",
        from_location="导诊台",
        to="门诊楼 2 层 骨科",
        estimated_minutes=4,
        steps=["乘电梯至 2 层", "左转到达骨科"],
        landmarks=["门诊楼"],
    )

    class FakeStructured:
        def invoke(self, _msgs):
            return FollowupToolPick(tool_name="get_department_route", reason="路线")

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructured()

        def invoke(self, _msgs):
            return AIMessage(content="## 骨科路线")

    captured: list[str] = []

    def capture_route(dept, from_location="导诊台"):
        captured.append(dept)
        return route

    monkeypatch.setattr("app.core.llm.get_chat_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        "app.mcp.followup.asyncio_run_list_tools",
        lambda: [type("T", (), {"name": "get_department_route", "description": ""})()],
    )
    monkeypatch.setattr("app.mcp.followup.fetch_department_route_sync", capture_route)

    state = AppState(
        last_recommended_department="消化内科",
        messages=[HumanMessage(content="骨科怎么走？")],
    )
    result = mcp_followup_node(state)
    assert captured == ["骨科"]
    assert result["last_recommended_department"] == "骨科"
    assert result["tool_call_result"]["department"] == "骨科"
