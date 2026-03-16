# @Time    : 2026/2/25 17:31
# @Author  : Yun
# @FileName: file_generator_state
# @Software: PyCharm
# @Desc    :
from operator import add
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage

from auto_test_assistant.utils.skill_loader import SkillMetadata


class FileGeneratorAgentState(TypedDict):
    messages: Annotated[list, add]  # 标准messages
    human_message: HumanMessage
    tasks: list[dict]  # [{"id":1, "content":"...", "status":"pending"}]
    skills: list[SkillMetadata]  # 选择了哪些skill
    skill_doc: list[str]  # 传入的skill文档
    uploaded_files_metadata: list[dict]
    current_task_idx: int  # 当前执行的任务索引，从0开始
    single_task_result: Annotated[list, add]  # 单条任务执行结果
    generated_file: str  # 最终输出
