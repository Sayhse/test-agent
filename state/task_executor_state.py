# @Time    : 2026/2/26 21:22
# @Author  : Yun
# @FileName: task_executor_state
# @Software: PyCharm
# @Desc    :
from operator import add
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage


class TaskExecutorState(TypedDict):
    messages: Annotated[list, add]  # 标准messages
    human_message: HumanMessage
    task: dict  # {"id":1, "content":"...", "status":"pending"}
    history_task_results: list
    uploaded_files_metadata: list[dict]
    execution_result: dict
    needs_input: bool
    question: str  # review_plan生成的提问
    user_answer: str  # human_input的回答
