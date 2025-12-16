import base64
from pathlib import Path
import json
import re
import uuid
from datetime import datetime

import gradio as gr
import pandas as pd
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem


APP_TITLE = "智慧问答助手"
COMPANY_NAME = "北京城建设计院"
LOGO_REL_PATH = Path("assets") / "logo_replace.png"


def _img_to_data_uri(img_path: Path) -> str | None:
    if not img_path.exists():
        return None
    try:
        raw = img_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        # 目前仅使用 png 资源；若后续换格式，可在此扩展 mime 推断
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)

    def _coerce_to_text(val) -> str:
        """
        gr.Chatbot 的 message/content 在不同模式下可能是：
        - str
        - list（多段内容/多模态）
        - dict（如 {"type": "...", "text": "..."}）
        - tuple/list（二元组消息）
        这里做成“尽量提取文本”的统一入口，保证导出不崩。
        """
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            # 常见：{"type": "text", "text": "..."} 或 {"content": "..."}
            for k in ("text", "content", "value", "message"):
                if k in val and isinstance(val.get(k), str):
                    return val.get(k) or ""
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        if isinstance(val, (list, tuple)):
            parts: list[str] = []
            for x in val:
                t = _coerce_to_text(x)
                if t:
                    parts.append(t)
            return "\n".join(parts)
        return str(val)

    def _extract_last_assistant_text(history) -> str:
        if not history:
            raise gr.Error("当前没有可导出的对话内容。")
        # history 可能是：
        # 1) [{"role": "user/assistant", "content": ...}, ...]
        # 2) [(user, assistant), ...] 或 [[user, assistant], ...]
        for item in reversed(history):
            if isinstance(item, dict) and item.get("role") == "assistant":
                text = _coerce_to_text(item.get("content")).strip()
                if text:
                    return text
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = _coerce_to_text(item[1]).strip()
                if text:
                    return text
        raise gr.Error("未找到可导出的助手回复。")

    def _try_parse_json(text: str):
        s = text.strip()
        if not s:
            return None
        if not (s.startswith("{") or s.startswith("[")):
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    def _split_markdown_tables(text: str) -> list[list[str]]:
        """
        极简 Markdown 表格检测：收集连续包含 '|' 的行块。
        返回：每个表格块的行列表。
        """
        lines = [ln.rstrip() for ln in (text or "").splitlines()]
        blocks: list[list[str]] = []
        cur: list[str] = []
        for ln in lines:
            if "|" in ln:
                cur.append(ln)
            else:
                if cur:
                    blocks.append(cur)
                    cur = []
        if cur:
            blocks.append(cur)
        # 过滤掉太短/不像表格的块（至少 header + 分隔 + 1 行）
        def looks_like_table(block: list[str]) -> bool:
            if len(block) < 3:
                return False
            # 分隔行通常包含 --- 或 :---:
            sep_like = any(re.search(r"\|?\s*:?-{3,}:?\s*\|", x) for x in block[1:3])
            return sep_like
        return [b for b in blocks if looks_like_table(b)]

    def _markdown_table_to_df(table_lines: list[str]) -> pd.DataFrame | None:
        """
        将一段 markdown 表格（行列表）解析成 DataFrame。
        支持首行 header + 第二行分隔线的常见格式。
        """
        if not table_lines or len(table_lines) < 2:
            return None

        def split_row(row: str) -> list[str]:
            r = row.strip().strip("|")
            return [c.strip() for c in r.split("|")]

        header = split_row(table_lines[0])
        # 找到分隔线位置
        sep_idx = None
        for i in range(1, min(len(table_lines), 4)):
            if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", table_lines[i]):
                sep_idx = i
                break
        if sep_idx is None:
            return None

        rows = []
        for ln in table_lines[sep_idx + 1 :]:
            if not ln.strip():
                continue
            cells = split_row(ln)
            # 容错：列数不齐就补空/截断
            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            if len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(cells)
        if not rows:
            return None
        return pd.DataFrame(rows, columns=header)

    def _text_to_excel(path: Path, text: str) -> Path:
        """
        将 text 尽量解析为结构化表格写入 xlsx，同时附带 raw 文本备份 sheet。
        """
        payload = _try_parse_json(text)
        tables: list[pd.DataFrame] = []

        if payload is not None:
            try:
                if isinstance(payload, list):
                    tables.append(pd.DataFrame(payload))
                elif isinstance(payload, dict):
                    # dict -> 两列 key/value 形式更通用
                    tables.append(pd.DataFrame([{"key": k, "value": v} for k, v in payload.items()]))
                else:
                    tables.append(pd.DataFrame([{"value": payload}]))
            except Exception:
                tables = []
        else:
            md_tables = _split_markdown_tables(text)
            for block in md_tables[:5]:  # 避免极端情况下 sheet 过多
                df = _markdown_table_to_df(block)
                if df is not None and not df.empty:
                    tables.append(df)

        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            wrote_any = False
            for i, df in enumerate(tables, start=1):
                sheet = f"table_{i}"
                df.to_excel(writer, index=False, sheet_name=sheet)
                wrote_any = True

            # raw 备份：即使解析成表格也保留原文，防止信息丢失
            raw_df = pd.DataFrame({"content": [text]})
            raw_df.to_excel(writer, index=False, sheet_name="raw")

            if not wrote_any:
                # 如果完全无法识别表格/JSON，也给一个最简 data sheet，便于用户打开就看到内容
                pd.DataFrame({"content": [text]}).to_excel(writer, index=False, sheet_name="data")

        return path

    def _maybe_llm_format_for_excel(text: str, use_llm: bool) -> str:
        if not use_llm:
            return text
        try:
            return chat_interface.format_for_excel(text)
        except Exception:
            return text
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 知识库中暂无文档"
        return "\n".join([f"{f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(f"✅ 已添加：{added} | 已跳过：{skipped}")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info("🗑️ 已删除所有文档")
        return format_file_list()
    
    def chat_handler(msg, hist):
        return chat_interface.chat(msg, hist)
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    repo_root = Path(__file__).resolve().parents[2]
    logo_path = repo_root / LOGO_REL_PATH
    logo_uri = _img_to_data_uri(logo_path)
    logo_html = (
        f'<img id="app-logo" src="{logo_uri}" alt="标志" />' if logo_uri else ""
    )
    header_html = f"""
    <div id="app-header">
      <div id="app-header-left">
        {logo_html}
        <div id="app-brand">
          <div id="app-title">{APP_TITLE}</div>
          <div id="app-company">{COMPANY_NAME}</div>
        </div>
      </div>
    </div>
    """

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(header_html)
        
        with gr.Tab("文档管理", elem_id="doc-management-tab"):
            gr.Markdown("## 添加新文档")
            gr.Markdown("支持上传 PDF 或 Markdown 文件；重复文件将自动跳过。")
            
            files_input = gr.UploadButton(
                label="选择 PDF/Markdown 文件并导入",
                variant="primary",
                size="lg",
                file_count="multiple",
                type="filepath",
                file_types=[".pdf", ".md", ".markdown"],
            )
            
            gr.Markdown("## 知识库当前文档")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines = 7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("刷新列表", size="md")
                clear_btn = gr.Button("清空全部", variant="stop", size="md")
            
            files_input.upload(
                upload_handler,
                [files_input],
                [files_input, file_list],
                show_progress="corner",
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("对话"):
            chatbot = gr.Chatbot(
                height=600, 
                placeholder="可以围绕已上传的知识库文档向我提问。",
                show_label=False,
            )
            chatbot.clear(clear_chat_handler)

            msg = gr.Textbox(
                placeholder="请输入你的问题（将基于知识库文档作答）",
                show_label=False,
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary", size="md")
                clear_chat_btn = gr.Button("清空对话", size="md")
                export_xlsx_btn = gr.Button("导出为Excel", size="md")
                smart_export_xlsx_btn = gr.Button("智能整理后导出", size="md")

            # 下载组件：优先使用 DownloadButton（若当前 gradio 版本支持），否则退化为 File
            DownloadButton = getattr(gr, "DownloadButton", None)
            if DownloadButton is not None:
                download_xlsx = DownloadButton("下载xlsx", visible=False)
            else:
                download_xlsx = gr.File(label="下载xlsx", interactive=False, visible=False)

            def _respond(user_message, history):
                text = (user_message or "").strip()
                if not text:
                    return "", history, gr.update(visible=False, value=None)
                bot = chat_handler(text, history)
                new_history = list(history or [])
                new_history.append({"role": "user", "content": text})
                new_history.append({"role": "assistant", "content": bot})
                # 新消息发送后，隐藏旧的下载，避免误下旧文件
                return "", new_history, gr.update(visible=False, value=None)

            def _clear_chat():
                clear_chat_handler()
                return [], gr.update(visible=False, value=None)

            def _export_last_answer_to_xlsx(history):
                text = _extract_last_assistant_text(history)
                repo_root = Path(__file__).resolve().parents[2]
                out_dir = repo_root / "exports"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"llm_output_{ts}_{uuid.uuid4().hex[:8]}.xlsx"
                _text_to_excel(out_path, text)
                gr.Info("✅ 已生成 xlsx，可点击下载。")
                return gr.update(value=str(out_path), visible=True)

            def _smart_export_last_answer_to_xlsx(history):
                text = _extract_last_assistant_text(history)
                gr.Info("⏳ 正在调用模型整理结构，请稍候 ...")
                formatted = _maybe_llm_format_for_excel(text, use_llm=True)
                repo_root = Path(__file__).resolve().parents[2]
                out_dir = repo_root / "exports"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"llm_output_structured_{ts}_{uuid.uuid4().hex[:8]}.xlsx"
                _text_to_excel(out_path, formatted)
                gr.Info("✅ 已生成（智能整理）xlsx，可点击下载。")
                return gr.update(value=str(out_path), visible=True)

            send_btn.click(_respond, inputs=[msg, chatbot], outputs=[msg, chatbot, download_xlsx])
            msg.submit(_respond, inputs=[msg, chatbot], outputs=[msg, chatbot, download_xlsx])
            clear_chat_btn.click(_clear_chat, inputs=None, outputs=[chatbot, download_xlsx])
            export_xlsx_btn.click(_export_last_answer_to_xlsx, inputs=[chatbot], outputs=[download_xlsx])
            smart_export_xlsx_btn.click(_smart_export_last_answer_to_xlsx, inputs=[chatbot], outputs=[download_xlsx])

        gr.HTML(f'<div id="app-footer">{COMPANY_NAME} · {APP_TITLE}</div>')
    
    return demo