from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SkillMetadata:
    """轻量级的 SKILL 元信息，只保留 name 和 description。"""

    name: str
    description: str
    path: Path  # 指向 skill 目录


def _parse_front_matter(skill_md_path: Path) -> Dict[str, str]:
    """
    只解析 SKILL.md 顶部的 YAML front matter，返回字典。

    约定格式：
    ---
    name: xxx
    description: yyy
    ---
    """
    text = skill_md_path.read_text(encoding="utf-8")
    # 只找最前面的 --- ... --- 段落
    if not text.lstrip().startswith("---"):
        return {}

    # 找到第一个和第二个 '---' 的位置
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}

    fm_lines: List[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        fm_lines.append(line)

    if not fm_lines:
        return {}

    try:
        data = yaml.safe_load("\n".join(fm_lines)) or {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        # 元信息解析失败时，不让整个系统崩溃
        return {}


def load_all_skills(skills_root: Path | str = "skills") -> List[SkillMetadata]:
    """
    扫描 skills 目录下的所有子目录，读取每个 SKILL.md 的元信息。

    只保留 name 和 description，并记录 skill 目录路径。
    """
    if type(skills_root) == str:
        root = Path(skills_root)
    else:
        root = skills_root
    if not root.exists():
        return []

    results: List[SkillMetadata] = []
    for skill_dir in root.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue

        fm = _parse_front_matter(skill_md)
        name = fm.get("name")
        description = fm.get("description")
        if not name or not description:
            # 如果缺少关键字段，则跳过该 skill
            continue

        results.append(
            SkillMetadata(
                name=name,
                description=description,
                path=skill_dir,
            )
        )

    return results
