# @Time    : 2026/3/4 14:47
# @Author  : Yun
# @FileName: ui_usecase_step_executor_state
# @Software: PyCharm
# @Desc    :
from operator import add
from typing import TypedDict, Annotated


class UiUseCaseStepExecutorState(TypedDict):
    messages: Annotated[list, add]
    ui_use_cases: list[dict]
    ui_use_case: dict
    path: list
    current_use_case_idx: int
    current_use_case: dict
    current_step: int
    script_path: str
    generate_degree: int
    generate_state: bool
