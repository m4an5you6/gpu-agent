"""Skill metadata tests for cluster-control optional skill."""

from __future__ import annotations

import re
from pathlib import Path

SKILL_PATH = Path(__file__).parents[2] / "optional-skills/devops/cluster-control/SKILL.md"


def test_cluster_control_skill_description_length():
    text = SKILL_PATH.read_text(encoding="utf-8")
    m = re.search(r"^description: (.*)$", text, re.MULTILINE)
    assert m, "description frontmatter missing"
    assert len(m.group(1)) <= 60


def test_cluster_control_skill_references_cluster_tools():
    text = SKILL_PATH.read_text(encoding="utf-8")
    for tool in (
        "cluster_status",
        "cluster_submit_job",
        "cluster_job_status",
        "cluster_logs",
    ):
        assert f"`{tool}`" in text
