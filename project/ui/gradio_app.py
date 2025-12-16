import base64
from pathlib import Path

import gradio as gr
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

            def _respond(user_message, history):
                text = (user_message or "").strip()
                if not text:
                    return "", history
                bot = chat_handler(text, history)
                new_history = list(history or [])
                new_history.append({"role": "user", "content": text})
                new_history.append({"role": "assistant", "content": bot})
                return "", new_history

            def _clear_chat():
                clear_chat_handler()
                return []

            send_btn.click(_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            msg.submit(_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            clear_chat_btn.click(_clear_chat, inputs=None, outputs=[chatbot])

        gr.HTML(f'<div id="app-footer">{COMPANY_NAME} · {APP_TITLE}</div>')
    
    return demo