# @Time    : 2026/2/25 9:57
# @Author  : Yun
# @FileName: state
# @Software: PyCharm
# @Desc    :
from operator import add
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, HumanMessage

from auto_test_assistant.utils.skill_loader import SkillMetadata


class TestAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    human_message: HumanMessage
    selected_skills: list[SkillMetadata]
    reason: str
    type: str
    uploaded_files: Annotated[list[str], add]
    uploaded_flag: bool
    uploaded_files_metadata: Annotated[list[dict], add]
