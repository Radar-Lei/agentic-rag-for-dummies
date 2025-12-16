from langchain_core.messages import HumanMessage

class ChatInterface:
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    def chat(self, message, history):

        if not self.rag_system.agent_graph:
            return "⚠️ 系统尚未初始化，请稍后重试。"
            
        try:
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            return result["messages"][-1].content
            
        except Exception as e:
            return f"❌ 发生错误：{str(e)}"

    def format_for_excel(self, text: str) -> str:
        """
        可选的“二次整理”：调用底层 LLM 将文本整理为严格 JSON，便于导出 Excel。
        约定：只返回 JSON（list[object] 或 object）。若无法结构化，返回 {"raw": "..."}。
        """
        if not (self.rag_system and getattr(self.rag_system, "llm", None)):
            return text

        raw = (text or "").strip()
        if not raw:
            return text

        prompt = (
            "你是数据整理助手。请把下面的内容整理为【严格 JSON】以便导入 Excel。\n"
            "要求：\n"
            "1) 只输出 JSON，不要输出任何解释/Markdown/代码块。\n"
            "2) 优先输出 JSON 数组（数组元素为对象），对象字段名尽量统一。\n"
            "3) 如果内容无法结构化为表格，请输出：{\"raw\": \"...\"}（保留原文，必要时做最小转义）。\n"
            "4) 输出必须能被 json.loads 直接解析。\n"
            "下面是原文：\n"
            f"{raw}"
        )

        try:
            resp = self.rag_system.llm.invoke([HumanMessage(content=prompt)])
            # langchain message/content 兼容
            formatted = getattr(resp, "content", None) or str(resp)
            return (formatted or "").strip() or text
        except Exception:
            return text
    
    def clear_session(self):
        self.rag_system.reset_thread()