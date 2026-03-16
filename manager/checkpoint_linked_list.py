# @Time    : 2026/3/6 8:39
# @Author  : Yun
# @FileName: checkpoint_linked_list
# @Software: PyCharm
# @Desc    :
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class CheckpointNode:
    """检查点节点，双向链表结构"""
    checkpoint_id: str
    timestamp: float
    screenshot_path: str  # 截图文件路径
    steps: List[str]     # 从上个检查点到当前检查点的步骤列表（不包括当前步骤）
    prev_id: Optional[str] = None  # 上一个检查点ID
    next_id: Optional[str] = None  # 下一个检查点ID
    metadata: Dict[str, Any] = None  # 额外元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "screenshot_path": self.screenshot_path,
            "steps": self.steps,
            "prev_id": self.prev_id,
            "next_id": self.next_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointNode":
        """从字典还原节点"""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            screenshot_path=data["screenshot_path"],
            steps=data["steps"],
            prev_id=data.get("prev_id"),
            next_id=data.get("next_id"),
            metadata=data.get("metadata", {})
        )


class CheckpointLinkedList:
    """检查点双向链表，每个测试用例一个链表"""
    def __init__(self, test_case_id: str):
        self.test_case_id = test_case_id
        self.head: Optional[CheckpointNode] = None
        self.tail: Optional[CheckpointNode] = None
        self.current: Optional[CheckpointNode] = None  # 当前检查点指针
        self.nodes: Dict[str, CheckpointNode] = {}     # ID到节点的映射，便于快速查找

    def append(self, node: CheckpointNode) -> None:
        """在链表末尾添加新检查点"""
        self.nodes[node.checkpoint_id] = node
        if self.tail is None:
            # 第一个节点
            self.head = node
            self.tail = node
        else:
            node.prev_id = self.tail.checkpoint_id
            self.tail.next_id = node.checkpoint_id
            self.tail = node
        self.current = node

    def get_node(self, checkpoint_id: str) -> Optional[CheckpointNode]:
        """根据ID获取节点"""
        return self.nodes.get(checkpoint_id)

    def get_previous_node(self, node: CheckpointNode) -> Optional[CheckpointNode]:
        """获取给定节点的上一个节点"""
        if node.prev_id:
            return self.nodes.get(node.prev_id)
        return None

    def get_next_node(self, node: CheckpointNode) -> Optional[CheckpointNode]:
        """获取给定节点的下一个节点"""
        if node.next_id:
            return self.nodes.get(node.next_id)
        return None

    def rollback_to(self, checkpoint_id: str) -> Optional[CheckpointNode]:
        """回滚到指定检查点，返回该节点并更新当前指针"""
        node = self.get_node(checkpoint_id)
        if node:
            self.current = node
        return node

    def get_steps_since(self, checkpoint_id: str) -> List[str]:
        """获取从指定检查点之后到当前检查点之间的所有步骤（不包括指定检查点的步骤）"""
        steps = []
        node = self.get_node(checkpoint_id)
        if not node:
            return steps
        current = node.next_id
        while current:
            next_node = self.nodes.get(current)
            if not next_node:
                break
            steps.extend(next_node.steps)
            current = next_node.next_id
        return steps

    def get_step_id_by_checkpoint(self, checkpoint_id: str) -> Optional[int]:
        node = self.get_node(checkpoint_id)
        if not node:
            return None
        return node.metadata.get("step_id", None)

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        nodes_dict = {cid: node.to_dict() for cid, node in self.nodes.items()}
        return {
            "test_case_id": self.test_case_id,
            "head_id": self.head.checkpoint_id if self.head else None,
            "tail_id": self.tail.checkpoint_id if self.tail else None,
            "current_id": self.current.checkpoint_id if self.current else None,
            "nodes": nodes_dict
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointLinkedList":
        """从字典还原链表"""
        linked_list = cls(data["test_case_id"])
        nodes_dict = data.get("nodes", {})
        for cid, node_data in nodes_dict.items():
            node = CheckpointNode.from_dict(node_data)
            linked_list.nodes[cid] = node
        # 重建头尾指针
        head_id = data.get("head_id")
        if head_id and head_id in linked_list.nodes:
            linked_list.head = linked_list.nodes[head_id]
        tail_id = data.get("tail_id")
        if tail_id and tail_id in linked_list.nodes:
            linked_list.tail = linked_list.nodes[tail_id]
        current_id = data.get("current_id")
        if current_id and current_id in linked_list.nodes:
            linked_list.current = linked_list.nodes[current_id]
        return linked_list

