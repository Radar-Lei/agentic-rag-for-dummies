import uuid
from langchain_deepseek import ChatDeepSeek
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
import json
import time
from pathlib import Path

class RAGSystem:
    
    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.llm = None
        self.thread_id = str(uuid.uuid4())
        
    def initialize(self):
        # region agent log
        try:
            Path("/Users/leida/Cline/agentic-rag-for-dummies/.cursor/debug.log").open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H1",
                        "location": "project/core/rag_system.py:initialize:entry",
                        "message": "RAGSystem.initialize entered",
                        "data": {"collection_name": self.collection_name},
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception:
            pass
        # endregion agent log

        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)
        
        # region agent log
        try:
            Path("/Users/leida/Cline/agentic-rag-for-dummies/.cursor/debug.log").open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H2",
                        "location": "project/core/rag_system.py:initialize:llm_config",
                        "message": "Preparing DeepSeek LLM config (no secrets)",
                        "data": {
                            "model": getattr(config, "DEEPSEEK_MODEL", None),
                            "base_url": getattr(config, "DEEPSEEK_BASE_URL", None),
                            "temperature": getattr(config, "LLM_TEMPERATURE", None),
                            "has_api_key": bool(getattr(config, "DEEPSEEK_API_KEY", "")),
                            "llm_class": "ChatDeepSeek",
                        },
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception:
            pass
        # endregion agent log

        llm = ChatDeepSeek(
            model=config.DEEPSEEK_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.llm = llm
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

        # region agent log
        try:
            Path("/Users/leida/Cline/agentic-rag-for-dummies/.cursor/debug.log").open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H3",
                        "location": "project/core/rag_system.py:initialize:exit",
                        "message": "RAGSystem.initialize completed",
                        "data": {"agent_graph_ready": self.agent_graph is not None},
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception:
            pass
        # endregion agent log
        
    def get_config(self):
        # recursion_limit: LangGraph 每次 invoke 的最大“步数/递归”上限（默认 25，容易触发）
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": getattr(config, "LANGGRAPH_RECURSION_LIMIT", 50),
        }
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
