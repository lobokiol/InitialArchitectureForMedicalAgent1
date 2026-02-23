# pip install rich requests
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "success": "bold green",
        "error": "bold red",
    }
)
console = Console(theme=custom_theme)


class ChatCLI:
    def __init__(self) -> None:
        self.base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
        # 默认等久一点（后台可能检索/推理较慢）；可通过 BACKEND_TIMEOUT 调整
        self.timeout = float(os.getenv("BACKEND_TIMEOUT", "120"))
        self.session = requests.Session()
        self.user_id: Optional[str] = None
        self.user_name: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []

    def run(self) -> None:
        console.print(Panel("医院导诊 Agentic 助手 CLI", style="success"))
        self._check_health()
        self._init_user()
        self._init_thread()
        self._loop()

    def _check_health(self) -> None:
        try:
            resp = self.session.get(f"{self.base_url}/healthz", timeout=self.timeout)
            resp.raise_for_status()
            console.print(Panel("后端已就绪", style="success"))
        except requests.RequestException as exc:
            console.print(Panel(f"无法连接后端：{exc}", style="warning"))

    def _init_user(self) -> None:
        self.user_id = Prompt.ask("请输入 user_id", default="demo-user", console=console).strip()
        name = Prompt.ask("请输入昵称（可留空）", default="", console=console).strip()
        self.user_name = name or None
        payload = {"user_id": self.user_id}
        if self.user_name:
            payload["name"] = self.user_name
        try:
            resp = self.session.post(
                f"{self.base_url}/users", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            self.user_name = data.get("name")
            console.print(
                Panel(
                    f"用户已就绪：{data.get('user_id')} {self.user_name or ''}".strip(),
                    style="success",
                )
            )
        except requests.RequestException as exc:
            console.print(Panel(f"创建/更新用户失败：{exc}", style="warning"))

    def _init_thread(self) -> None:
        if not self.user_id:
            return
        try:
            resp = self.session.get(
                f"{self.base_url}/threads/current",
                params={"user_id": self.user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self.thread_id = data.get("thread_id")
            title = data.get("title", "未命名会话")
            self._render_header(title)
        except requests.RequestException as exc:
            console.print(Panel(f"获取当前会话失败：{exc}", style="warning"))

    def _render_header(self, title: str) -> None:
        user_line = f"用户: {self.user_id}"
        if self.user_name:
            user_line += f"（{self.user_name}）"
        thread_line = f"当前会话: {title} ({self.thread_id})"
        console.print(Panel(f"{user_line}\n{thread_line}", style="info"))

    def _loop(self) -> None:
        console.print(
            Panel(
                "输入问题直接对话；以 / 开头执行命令（/help 查看）；/exit 退出。",
                style="info",
            )
        )
        while True:
            try:
                raw = Prompt.ask(">>>", console=console).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[warning]已退出[/warning]")
                break
            if not raw:
                continue
            if raw.startswith("/"):
                if raw in ("/exit", "/quit"):
                    console.print("[warning]已退出[/warning]")
                    break
                self._handle_command(raw)
                continue
            self._ask_chat(raw)

    def _handle_command(self, cmd: str) -> None:
        if cmd == "/help":
            self._render_help()
        elif cmd == "/threads":
            self._list_threads()
        elif cmd == "/new":
            self._create_thread()
        elif cmd == "/switch":
            self._switch_thread()
        elif cmd == "/delete":
            self._delete_thread()
        elif cmd == "/user":
            self._show_user()
        else:
            console.print(Panel(f"未知命令：{cmd}", style="warning"))

    def _render_help(self) -> None:
        text = "\n".join(
            [
                "/help      查看命令列表",
                "/threads   列出所有会话",
                "/new       创建新会话",
                "/switch    切换会话",
                "/delete    删除当前或指定会话",
                "/user      查看或更新用户信息",
                "/exit      退出",
            ]
        )
        console.print(Panel(text, title="帮助", style="info"))

    def _list_threads(self) -> List[Dict[str, Any]]:
        if not self.user_id:
            return []
        try:
            resp = self.session.get(
                f"{self.base_url}/threads",
                params={"user_id": self.user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            threads = resp.json()
            if not threads:
                console.print(Panel("暂无会话", style="warning"))
                return []
            lines = []
            for idx, t in enumerate(threads, 1):
                mark = " (当前)" if t.get("thread_id") == self.thread_id else ""
                lines.append(
                    f"{idx}. {t.get('title','未命名')} [{t.get('thread_id')}] - 最近: {t.get('last_active_at')}{mark}"
                )
            console.print(Panel("\n".join(lines), title="会话列表", style="info"))
            return threads
        except requests.RequestException as exc:
            console.print(Panel(f"获取会话列表失败：{exc}", style="warning"))
            return []

    def _create_thread(self) -> None:
        if not self.user_id:
            return
        title = Prompt.ask("新会话标题（可留空）", default="", console=console).strip() or None
        payload = {"user_id": self.user_id}
        if title:
            payload["title"] = title
        try:
            resp = self.session.post(
                f"{self.base_url}/threads", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            self.thread_id = data.get("thread_id")
            self._render_header(data.get("title", "未命名会话"))
        except requests.RequestException as exc:
            console.print(Panel(f"创建会话失败：{exc}", style="warning"))

    def _switch_thread(self) -> None:
        threads = self._list_threads()
        if not threads:
            return
        choice = Prompt.ask("输入序号或 thread_id", console=console).strip()
        target_id = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(threads):
                target_id = threads[idx].get("thread_id")
        else:
            target_id = choice
        if not target_id:
            console.print(Panel("未选择有效会话", style="warning"))
            return
        payload = {"user_id": self.user_id, "thread_id": target_id}
        try:
            resp = self.session.post(
                f"{self.base_url}/threads/switch",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self.thread_id = data.get("thread_id")
            self._render_header(data.get("title", "未命名会话"))
        except requests.RequestException as exc:
            console.print(Panel(f"切换会话失败：{exc}", style="warning"))

    def _delete_thread(self) -> None:
        if not self.thread_id or not self.user_id:
            console.print(Panel("当前无可删除会话", style="warning"))
            return
        target = Prompt.ask(
            f"输入要删除的 thread_id（回车删除当前 {self.thread_id}）", default="", console=console
        ).strip() or self.thread_id
        try:
            resp = self.session.delete(
                f"{self.base_url}/threads/{target}",
                params={"user_id": self.user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            console.print(Panel(f"删除结果：{data}", style="info"))
            if data.get("new_current_thread_id"):
                self.thread_id = data["new_current_thread_id"]
            elif data.get("deleted"):
                self.thread_id = None
                self._init_thread()
        except requests.RequestException as exc:
            console.print(Panel(f"删除会话失败：{exc}", style="warning"))

    def _show_user(self) -> None:
        if not self.user_id:
            return
        try:
            resp = self.session.get(
                f"{self.base_url}/users/{self.user_id}", timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            self.user_name = data.get("name")
            console.print(Panel(f"用户信息：{data}", style="info"))
        except requests.RequestException:
            console.print(Panel("获取用户信息失败", style="warning"))
        if Prompt.ask("更新昵称？(y/n)", choices=["y", "n"], default="n", console=console) == "y":
            new_name = Prompt.ask("输入新的昵称", default=self.user_name or "", console=console).strip()
            payload = {"user_id": self.user_id, "name": new_name}
            try:
                resp = self.session.post(
                    f"{self.base_url}/users", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                self.user_name = resp.json().get("name")
                console.print(Panel("昵称已更新", style="success"))
            except requests.RequestException as exc:
                console.print(Panel(f"更新失败：{exc}", style="warning"))

    def _ask_chat(self, message: str) -> None:
        if not self.user_id:
            console.print(Panel("请先初始化用户", style="warning"))
            return
        payload = {"user_id": self.user_id, "message": message}
        if self.thread_id:
            payload["thread_id"] = self.thread_id
        console.print(Panel(message, title="你", style="info"))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("助手思考中...", start=True)
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                progress.stop_task(task_id)
                console.print(Panel(f"请求失败：{exc}", style="warning"))
                return
            finally:
                progress.stop()
        self.thread_id = data.get("thread_id") or self.thread_id
        reply = data.get("reply", "")
        console.print(Panel(Markdown(reply), title="助手", style="success"))
        self._render_intent(data.get("intent_result"))
        self._render_docs(data.get("used_docs", {}))

    def _render_intent(self, intent: Optional[Dict[str, Any]]) -> None:
        if not intent:
            return
        lines = []
        for key, val in intent.items():
            lines.append(f"{key}: {val}")
        console.print(Panel("\n".join(lines), title="意图识别", style="info"))

    def _render_docs(self, used_docs: Dict[str, Any]) -> None:
        if not used_docs:
            return
        parts = []
        for key in ("medical", "process"):
            docs = used_docs.get(key) or []
            if not docs:
                continue
            top = docs[:3]
            doc_lines = [
                f"- {d.get('title','无标题')} ({d.get('source','未知来源')}) score={d.get('score')}"
                for d in top
            ]
            parts.append(f"[{key}] \n" + "\n".join(doc_lines))
        if parts:
            console.print(Panel("\n\n".join(parts), title="参考文档", style="info"))


def main() -> None:
    cli = ChatCLI()
    cli.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        console.print(Panel(f"程序异常：{exc}", style="error"))
        sys.exit(1)
