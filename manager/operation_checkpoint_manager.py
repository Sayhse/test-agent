# @Time    : 2026/3/5 14:40
# @Author  : Yun
# @FileName: operation_checkpoint_manager
# @Software: PyCharm
# @Desc    : 操作检查点管理器，支持版本控制、双向链表历史记录和模型验证回滚
import os
import shutil
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from auto_test_assistant.manager.checkpoint_linked_list import CheckpointLinkedList, CheckpointNode


class OperationCheckpointManager:
    """操作检查点管理器，用于版本控制和回滚"""

    def __init__(self, checkpoint_dir: str = "./checkpoints", copy_dir: str = "./copies"):
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.copy_dir = Path(copy_dir)
        os.makedirs(self.copy_dir, exist_ok=True)
        self.test_cases: Dict[str, CheckpointLinkedList] = {}  # 测试用例ID -> 链表
        self.current_test_case_id: Optional[str] = None

    def set_current_test_case(self, test_case_id: str, operation_log_path: Optional[str] = None) -> None:
        """设置当前测试用例ID，如果不存在则创建新的链表"""
        self.current_test_case_id = test_case_id
        if operation_log_path is not None:
            log_path = Path(operation_log_path)
            if test_case_id not in self.test_cases:
                self.test_cases[test_case_id] = CheckpointLinkedList(test_case_id)
                if not log_path.exists():
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write("import time\n")
                        f.write("import pyautogui\n\n")
                        f.write("time.sleep(1)\n")

    def get_current_linked_list(self) -> Optional[CheckpointLinkedList]:
        """获取当前测试用例的链表"""
        if self.current_test_case_id:
            return self.test_cases.get(self.current_test_case_id)
        return None

    def create_checkpoint(self, operation_log_path: str, steps: List[str], screenshot_path: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        为当前测试用例创建检查点
        
        Args:
            operation_log_path: 操作日志文件路径
            steps: 从上个检查点到当前检查点的步骤列表（不包括当前步骤）
            screenshot_path: 截图文件路径
            metadata: 额外元数据
            
        Returns:
            检查点ID
        """
        if not self.current_test_case_id:
            raise ValueError("未设置当前测试用例ID，请先调用set_current_test_case")

        linked_list = self.test_cases[self.current_test_case_id]
        checkpoint_id = f"checkpoint_{int(time.time() * 1000)}"
        node = CheckpointNode(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            screenshot_path=screenshot_path,
            steps=steps,
            metadata=metadata or {}
        )
        linked_list.append(node)

        # 保存检查点元数据到文件
        self._save_checkpoint_file(node)
        # 保存链表状态
        self._save_linked_list_state(linked_list)
        # 保存当前状态下操作日志到文件
        self._save_operation_log(operation_log_path, checkpoint_id)

        return checkpoint_id

    def get_current_checkpoint(self) -> Optional[CheckpointNode]:
        """获取当前测试用例下当前的检查点"""
        linked_list = self.get_current_linked_list()
        if not linked_list:
            return None
        return linked_list.current

    def get_previous_checkpoint(self) -> Optional[CheckpointNode]:
        """获取当前测试用例的上一个检查点（相对于当前检查点）"""
        linked_list = self.get_current_linked_list()
        if not linked_list or not linked_list.current:
            return None
        return linked_list.get_previous_node(linked_list.current)

    def get_steps_since_last_checkpoint(self) -> List[str]:
        """获取自上一个检查点以来的步骤列表（用于验证）"""
        linked_list = self.get_current_linked_list()
        if not linked_list or not linked_list.current:
            return []
        prev_node = linked_list.get_previous_node(linked_list.current)
        if not prev_node:
            return []
        return linked_list.get_steps_since(prev_node.checkpoint_id)

    def validate_and_create_checkpoint(self, operation_log_path: str, steps: List[str], current_screenshot_path: str,
                                       model_validator: callable, step_id: int) -> bool:
        """
        验证步骤并创建检查点（集成模型验证）
        
        Args:
            operation_log_path: 操作日志文件路径
            steps: 待验证的步骤列表（从上个检查点到当前步骤的前一步）
            current_screenshot_path: 当前截图路径
            model_validator: 模型验证函数，接收(步骤列表, 当前截图路径, 上一个检查点截图路径)，
                            返回验证结果(bool)和错误信息(可选)
            step_id: 当前步骤的id
            
        Returns:
            bool: 验证是否成功，成功则创建检查点
        """
        linked_list = self.get_current_linked_list()
        if not linked_list:
            raise ValueError("未设置当前测试用例")

        prev_node = linked_list.current
        prev_screenshot_path = prev_node.screenshot_path if prev_node else None

        # 调用模型验证
        try:
            validation_result = model_validator(steps, current_screenshot_path, prev_screenshot_path)
            if isinstance(validation_result, tuple):
                success, message = validation_result
            else:
                success = bool(validation_result)
                message = "验证成功" if success else "验证失败"
        except Exception as e:
            success = False
            message = f"模型验证异常: {e}"

        if success:
            # 验证成功，创建检查点
            self.create_checkpoint(operation_log_path, steps, current_screenshot_path, {
                "validation_result": "success",
                "validation_message": message,
                "step_id": step_id
            })
            return True
        else:
            # 验证失败，记录日志但不创建检查点
            print(f"步骤验证失败: {message}")
            return False

    def rollback_to_checkpoint(self, checkpoint_id: str, operation_log_path: str) -> Dict[str, Any]:
        """
        回滚到指定检查点
        
        Args:
            checkpoint_id: 目标检查点ID
            operation_log_path: 操作日志文件路径
            
        Returns:
            回滚信息，包括步骤列表和截图路径
        """
        linked_list = self.get_current_linked_list()
        if not linked_list:
            raise ValueError("未设置当前测试用例")

        target_node = linked_list.rollback_to(checkpoint_id)
        if not target_node:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")

        # 获取需要重新执行的步骤（从目标检查点之后到当前检查点之间）
        steps_to_redo = linked_list.get_steps_since(checkpoint_id)

        first_step_to_redo = linked_list.get_step_id_by_checkpoint(checkpoint_id)

        # 保存回滚记录
        rollback_info = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "target_screenshot_path": target_node.screenshot_path,
            "steps_to_redo": steps_to_redo,
            "first_step_to_redo": first_step_to_redo,
            "message": f"已回滚到检查点 {checkpoint_id}"
        }

        self._save_rollback_record(rollback_info)
        self._redo_operation_log(operation_log_path, checkpoint_id)
        return rollback_info

    def _redo_operation_log(self, operation_log_path: str, checkpoint_id: str):
        operation_log_path_copy = self.copy_dir / self.current_test_case_id / f"{checkpoint_id}.py"
        shutil.copy2(operation_log_path_copy, operation_log_path)

    def _save_checkpoint_file(self, node: CheckpointNode) -> None:
        """保存检查点元数据到JSON文件"""
        checkpoint_file = self.checkpoint_dir / f"{node.checkpoint_id}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(node.to_dict(), f, ensure_ascii=False, indent=2)

    def _save_linked_list_state(self, linked_list: CheckpointLinkedList) -> None:
        """保存链表状态到文件"""
        state_file = self.checkpoint_dir / f"{linked_list.test_case_id}_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(linked_list.to_dict(), f, ensure_ascii=False, indent=2)

    def _save_rollback_record(self, rollback_info: Dict[str, Any]) -> None:
        """保存回滚记录"""
        record_file = self.checkpoint_dir / f"rollback_{int(time.time() * 1000)}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(rollback_info, f, ensure_ascii=False, indent=2)

    def _save_operation_log(self, operation_log_path: str, checkpoint_id: str) -> None:
        """保存操作日志副本"""
        os.makedirs(self.copy_dir / self.current_test_case_id, exist_ok=True)
        operation_log_path_copy = self.copy_dir / self.current_test_case_id / f"{checkpoint_id}.py"
        shutil.copy2(operation_log_path, operation_log_path_copy)

    def load_test_case_state(self, test_case_id: str) -> bool:
        """从文件加载测试用例的链表状态"""
        state_file = self.checkpoint_dir / f"{test_case_id}_state.json"
        if not state_file.exists():
            return False

        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        linked_list = CheckpointLinkedList.from_dict(data)
        self.test_cases[test_case_id] = linked_list
        return True

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """获取检查点详细信息"""
        for linked_list in self.test_cases.values():
            node = linked_list.get_node(checkpoint_id)
            if node:
                return node.to_dict()
        return None
