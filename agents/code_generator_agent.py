# @Time    : 2026/3/2 17:19
# @Author  : Yun
# @FileName: code_generator_agent
# @Software: PyCharm
# @Desc    :
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from auto_test_assistant.utils.tools import list_all_mcp_tools


class CodeGeneratorAgentSystem:
    def __init__(
            self,
            llm: BaseChatModel,
    ) -> None:
        self.agent = None
        self.llm = llm
        self.tools = list_all_mcp_tools()

    def create_agent(self, all_steps: list, file_address: str):
        """
        创建测试步骤执行agent
        :param all_steps: 这个测试用例下面所有的测试步骤（供参考）
        :param file_address: 生成的脚本保存路径
        """
        # 定义JSON Schema
        execute_step_schema = {
            "type": "object",
            "description": "测试步骤执行结果",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "执行结果总结"
                },
                "status": {
                    "type": "string",
                    "description": "执行状态",
                    "enum": ["success", "failure", "partial_success"]
                },
                "generated_code_lines": {
                    "type": "array",
                    "description": "生成的代码行",
                    "items": {"type": "string"}
                },
                "rollback_step_id": {
                    "type": "string",
                    "description": "如果回退到上一个检查点，则返回回退到的步骤id"
                }
            },
            "required": ["summary", "status"]
        }

        # 构建系统提示
        system_prompt = (
            "你是一个测试步骤执行助手，负责根据测试步骤描述选择合适的工具执行操作。\n"
            "重要指令：不要自己生成代码，而是调用合适的工具来执行操作。\n"
            "你可以使用以下工具来帮助你完成任务：\n"
            f"{self.tools}\n"
            "根据提供的当前测试用例ID和当前测试步骤id执行测试步骤\n"
            "所有测试步骤（供参考）：\n"
            f"{all_steps}\n"
            "请根据测试步骤描述选择合适的工具执行操作。\n"
            "如果一个测试步骤需要调用多个工具，请按逻辑顺序调用。\n"
            "将操作记录到脚本文件中，使用 operation_log_path 参数指定文件路径。\n"
            f"脚本文件路径：{file_address}\n"
            "请在调用工具时提供 operation_log_path 参数，值为上述文件路径。\n"
            "\n"
            "输出要求：\n"
            "你必须输出严格符合JSON Schema的JSON格式结果。JSON必须包含以下字段：\n"
            "1. summary: 执行结果总结\n"
            "2. status: 执行状态（success/failure/partial_success）\n"
            "3. generated_code_lines: 生成的代码行列表（可选）\n"
            "4. rollback_step_id: 回退到的上一个步骤id\n"
        )

        # 创建agent，指定response_format为JSON Schema
        agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
            response_format=execute_step_schema,
            checkpointer=InMemorySaver(),
        )
        self.agent = agent
        return agent

    def execute_use_case_step(self, use_case_id: str, step_id: int):
        """
        按照测试步骤选择合适工具进行执行
        :param use_case_id: 所属用例的id
        :param step_id: 这一步测试步骤的id
        """
        if self.agent is None:
            raise Exception("请先调用create_agent函数！")
        # 执行任务
        result = self.agent.invoke({
            "messages": [
                {"role": "user", "content": f"当前测试用例ID: {use_case_id}\n"
                                            f"当前测试步骤id: {step_id}\n"},
            ]
        }, {"configurable": {"thread_id": "1"}})

        # 解析执行结果
        try:
            data = result["structured_response"]
            summary = data.get("summary", "")
            status = data.get("status", "")
            generated_code_lines = data.get("generated_code_lines", [])
            return {
                "summary": summary,
                "status": status,
                "generated_code_lines": generated_code_lines
            }
        except Exception as e:
            return {
                "summary": f"执行失败，失败原因：{e}",
                "status": "failure",
                "generated_code_lines": []
            }
