# @Time    : 2026/2/25 9:54
# @Author  : Yun
# @FileName: main
# @Software: PyCharm
# @Desc    :
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from auto_test_assistant.graph.main_graph import generate_graph


if __name__ == '__main__':
    app = generate_graph()
    config = {
        "configurable": {
            "thread_id": "customer_123"
        }
    }
    load_dotenv()
    uploaded = os.getenv("UPLOADED_FILE")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("退出。")
            break
        for chunk in app.stream({
            "messages": [HumanMessage(content=q)],
            "human_message": HumanMessage(content=q),
            "uploaded_files": [uploaded],
            "uploaded_flag": True
        },
                config=config,
                stream_mode="custom"):
            print(chunk)
