# @Time    : 2026/2/25 11:02
# @Author  : Yun
# @FileName: intent_recognition
# @Software: PyCharm
# @Desc    :
from dataclasses import dataclass
from pathlib import Path, WindowsPath
from typing import List, Dict, Any, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

from ..utils.skill_loader import SkillMetadata, load_all_skills
from ..utils.tools import list_all_mcp_tools


@dataclass
class RoutingDecision:
    """主 agent 对一次用户请求的路由决策结果。"""
    type: str
    reason: str
    selected_skills: List[SkillMetadata]


class AutoTestAgentSystem:
    def __init__(
            self,
            llm: BaseChatModel,
            skills: List[SkillMetadata] | Path = "skills",
    ) -> None:
        self.llm = llm
        if type(skills) == WindowsPath:
            self.skills: List[SkillMetadata] = load_all_skills(skills)
        else:
            self.skills = skills
        self.tools = list_all_mcp_tools()

    def handle_request(self, messages: List) -> RoutingDecision:
        """
        意图识别agent使用 LLM 处理state中的messages列表，分析是否需要进行本系统支持的功能
        返回 str，为下一节点node type
        :param messages: 消息列表，按时间顺序排列
        """
        skills_brief = "\n".join(
            f"- {s.name}: {s.description}" for s in self.skills
        ) or "(当前没有可用技能)"

        # 确定最后一条消息的角色
        last_message = messages[-1] if messages else None
        role = ""
        if last_message:
            if type(last_message) == AIMessage or type(last_message) == SystemMessage or type(
                    last_message) == ToolMessage:
                role = "assistant"
            elif type(last_message) == HumanMessage:
                role = "user"

        # 查找最后一条 HumanMessage 及其索引
        last_human_index = -1
        last_human_content = ""
        for i, msg in enumerate(messages):
            if type(msg) == HumanMessage:
                last_human_index = i
                last_human_content = msg.content if hasattr(msg, 'content') else str(msg)

        # 提取最后一条 HumanMessage 之后的所有消息作为历史操作
        history_messages = []
        if last_human_index >= 0 and last_human_index + 1 < len(messages):
            history_messages = messages[last_human_index + 1:]

        # 构建历史操作描述
        history_description = ""
        if history_messages:
            history_lines = []
            for msg in history_messages:
                msg_type = type(msg).__name__
                msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                history_lines.append(f"- {msg_type}: {msg_content}")
            history_description = "系统已完成以下操作：\n" + "\n".join(history_lines) + "\n\n"
        else:
            history_description = "系统尚未执行任何操作。\n\n"

        available_nodes = ["parse_file", "file_generation", "ui_use_case_code_generation", "code_review", "code_execution",
                           "error_analysis"]
        judge_info_schema = {
            "type": "object",
            "description": "判断任务类型。",
            "properties": {
                "type": {"type": "string",
                         "description": f"任务类型字段，判断任务是否结束，结束返回‘END’，否则判断任务类型是否属于{available_nodes}中一项，如果没有则返回 'other' 字段，否则返回{available_nodes}中最符合的任务类型字段"},
                "reason": {"type": "string", "description": "用中文解释为什么是这个类型的任务。"},
                "selected_skills": {"type": "array",
                                    "description": "如果 'type' 字段不等于'file_generation'则为空数组；否则判断如果需要生成测试计划、测试用例、测试脚本等任务，则从技能列表中选择一个或多个最相关的技能 name 放入 selected_skills。",
                                    "items": {"type": "string"}}
            },
            "required": ["type", "reason", "selected_skills"]
        }

        system_prompt = (
            f"你是一个智能助手中的一个路由调度器，负责判断请求的任务类型是否属于{available_nodes}其中一项，如果没有则类型字段为 'other'。\n"
            "向你发送请求的可能是用户也有可能是智能助手中其他调度器。如果角色为用户则不可将任务类型字段填充为‘END’！！\n"
            f"当前发送请求的角色为：{role}\n\n"
            "你会看到一组可用技能列表（name + description），以及自然语言请求。\n"
            "你可以使用以下 MCP 工具来帮助你完成任务：\n"
            f"{self.tools}\n"
            "请只输出 JSON"
        )
        user_prompt = (
            "可用技能列表：\n"
            f"{skills_brief}\n\n"
            f"用户任务：{last_human_content}\n\n"
            f"当前系统已经完成以下操作：{history_description}"
            "请根据上述规则输出 JSON。"
        )
        agent = create_agent(model=self.llm, system_prompt=system_prompt, response_format=judge_info_schema,
                             tools=self.tools)

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": user_prompt}]
        })
        try:
            data = result["structured_response"]
            task_type = str(data.get("type", ""))
            reason = str(data.get("reason", ""))
            skills = []
            for s in data.get("selected_skills", []):
                for skill in self.skills:
                    if skill.name == s:
                        skills.append(skill)
            return RoutingDecision(type=task_type, reason=reason, selected_skills=skills)
        except Exception:
            # 解析失败时保守退回其他任务类型
            return RoutingDecision(
                type="other",
                reason="路由 JSON 解析失败，回退为简单任务处理。",
                selected_skills=[],
            )

    def extract_workflow_from_skill_md(self, skill_doc: str) -> List[Dict[str, Any]]:
        """
        使用 LLM 从 SKILL.md 文档中智能提取工作流程步骤，生成 TodoList。

        策略：
        1. 将 SKILL.md 文档交给 LLM 分析
        2. 让 LLM 识别工作流程、分析步骤、执行步骤等章节
        3. 提取关键步骤并生成结构化的 TodoList
        4. 返回 JSON 格式的任务列表
        """
        # 定义返回的 JSON Schema
        workflow_schema = {
            "type": "object",
            "description": "从 SKILL.md 提取的工作流程 TodoList",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "工作流程任务列表，按照执行顺序排列",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "任务唯一标识符，从1开始递增"
                            },
                            "content": {
                                "type": "string",
                                "description": "任务描述，应具体、可执行，遵循SMART原则"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending"],
                                "description": "任务初始状态，固定为pending"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["must-have", "should-have", "could-have", "won't-have"],
                                "description": "任务优先级，使用MoSCoW法"
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "依赖的任务ID列表，如果没有依赖则为空数组"
                            }
                        },
                        "required": ["id", "content", "status", "priority"]
                    }
                }
            },
            "required": ["todos"]
        }

        system_prompt = (
            "你是一个工作流程分析专家，擅长从技能文档中提取工作流程步骤。\n"
            "你的任务是从给定的 SKILL.md 文档中识别并提取工作流程步骤，生成结构化的 TodoList。\n\n"
            "分析要求：\n"
            "1. 仔细阅读整个 SKILL.md 文档\n"
            "2. 识别文档中描述的工作流程、分析步骤、执行步骤等章节\n"
            "3. 提取关键的执行步骤，每个步骤应该是：\n"
            "   - 具体、可执行（遵循SMART原则）\n"
            "   - 有明确的执行顺序\n"
            "   - 可以独立完成或明确依赖关系\n"
            "4. 如果文档中有多个层级的步骤（主步骤和子步骤），优先提取主步骤\n"
            "5. 如果文档中没有明确的工作流程章节，根据文档内容推断合理的执行步骤\n"
            "6. 为每个步骤设置合适的优先级（must-have/should-have/could-have）\n"
            "7. 识别步骤之间的依赖关系\n\n"
            "你可以使用以下 MCP 工具来帮助你完成任务：\n"
            f"{self.tools}\n"
            "输出要求：\n"
            "请只输出 JSON 格式，包含 todos 数组，每个任务包含 id、content、status、priority、depends_on 字段。"
        )

        user_prompt = (
            "请分析以下 SKILL.md 文档，提取工作流程步骤并生成 TodoList：\n\n"
            "---------------- SKILL.md 开始 ----------------\n"
            f"{skill_doc}\n"
            "---------------- SKILL.md 结束 ----------------\n\n"
            "请识别文档中描述的工作流程步骤，生成结构化的 TodoList。"
        )

        try:
            # 使用 LLM 提取工作流程
            extractor_agent = create_agent(
                model=self.llm,
                system_prompt=system_prompt,
                response_format=workflow_schema,
                tools=self.tools
            )

            result = extractor_agent.invoke({
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            })

            # 解析返回结果
            if isinstance(result, dict):
                # 优先尝试从 structured_response 提取
                structured_response = result.get("structured_response", {})
                if structured_response and isinstance(structured_response, dict):
                    todos_data = structured_response.get("todos", [])
                    if todos_data and isinstance(todos_data, list):
                        return todos_data

                # 尝试从 output 字段提取
                output = result.get("output", "")
                if output and isinstance(output, str):
                    todos_data = self._parse_json_from_text(output)
                    if todos_data:
                        return todos_data

                # 尝试从 messages 中提取
                messages = result.get("messages", [])
                for msg in messages:
                    content = ""
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                    elif hasattr(msg, "content"):
                        content = str(msg.content)

                    if content:
                        todos_data = self._parse_json_from_text(content)
                        if todos_data:
                            return todos_data
            elif isinstance(result, str):
                # 如果直接返回字符串，尝试解析
                todos_data = self._parse_json_from_text(result)
                if todos_data:
                    return todos_data

            # 如果所有解析都失败，使用 fallback
            return self._extract_workflow_fallback()

        except Exception as e:
            # 如果出错，使用 fallback 方法
            return self._extract_workflow_fallback()

    def _parse_json_from_text(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        从文本中解析 JSON 格式的 TodoList。
        支持多种格式：完整的 JSON 对象、JSON 代码块、内嵌 JSON 等。
        """
        import json
        import re

        if not text or "todos" not in text:
            return None

        # 尝试1：查找 JSON 代码块
        json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                todos = data.get("todos", [])
                if todos and isinstance(todos, list):
                    return todos
            except:
                continue

        # 尝试2：查找最外层的大括号对
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            if "todos" in match:
                try:
                    data = json.loads(match)
                    todos = data.get("todos", [])
                    if todos and isinstance(todos, list):
                        return todos
                except:
                    continue

        # 尝试3：直接查找第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                json_str = text[start:end + 1]
                data = json.loads(json_str)
                todos = data.get("todos", [])
                if todos and isinstance(todos, list):
                    return todos
            except:
                pass

        return None

    def _extract_workflow_fallback(self) -> List[Dict[str, Any]]:
        """
        Fallback 方法：当 LLM 提取失败时使用简单的默认步骤。
        """
        return [
            {
                "id": "1",
                "content": "分析用户需求和输入材料，理解任务目标",
                "status": "pending",
                "priority": "must-have",
            },
            {
                "id": "2",
                "content": "按照 SKILL.md 中的工作流程执行任务",
                "status": "pending",
                "priority": "must-have",
                "depends_on": ["1"],
            },
            {
                "id": "3",
                "content": "生成最终输出并验证质量",
                "status": "pending",
                "priority": "must-have",
                "depends_on": ["2"],
            },
        ]
