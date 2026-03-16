# @Time    : 2026/3/2 15:50
# @Author  : Yun
# @FileName: ui_usecase_code_generator_state
# @Software: PyCharm
# @Desc    :
from operator import add
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage


class UiUseCaseCodeGeneratorState(TypedDict):
    messages: Annotated[list, add]
    human_message: HumanMessage
    uploaded_files_metadata: list[dict]
    ui_use_cases: list[dict]
