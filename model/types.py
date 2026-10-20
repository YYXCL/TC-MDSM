'Experiment configuration and data structures for prediction and scheduling.'
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import deque


GROUP_DIMS = {"A": 6, "C": 5, "E": 4, "R": 4}

METHODS = ("TC-MDSM", "GMATANN", "HM-MATAS", "Rule-Latency", "Full-State", "StaticGraph")


@dataclass
class ModelConfig:
    'Model and scheduling hyperparameters.'
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    history: int = 16
    top_b: int = 16
    gate_threshold: float = 0.5
    candidate_budgets: tuple[int, ...] = (8,)
    planning_batch_size: int = 4
    dropout: float = 0.05
    ablation: str = "none"
    prediction_mode: str = "residual"
    deadline_policy: str = "hard"
    solver_start: str = "serial"
    execution_policy: str = "ready"

    @property
    def state_dimension_budget(self):
        'Number of retained dimensions per state group.'
        return self.top_b

    def __post_init__(self):

        if len(self.candidate_budgets)!=1 or self.candidate_budgets[0]<=0 or self.planning_batch_size<=0:
            raise ValueError("candidate budget and planning batch size must be positive")
        if self.hidden % self.heads or self.top_b > self.hidden:
            raise ValueError("hidden must divide heads and top_b must not exceed hidden")
        if self.deadline_policy not in ("hard","evaluate"):
            raise ValueError("deadline_policy must be hard or evaluate")
        if self.solver_start not in ("serial","reject"):
            raise ValueError("solver_start must be serial or reject")
        if self.execution_policy not in ("ready","planned"):
            raise ValueError("execution_policy must be ready or planned")


@dataclass
class Snapshot:
    'Environment state available at decision time, excluding future information.'
    time: float
    observed_at: list[float]
    states: dict[str, list]
    missing: dict[str, list]
    skills: list[list[str]]
    tools: list[list[str]]
    knowledge: list[list[str]]
    descriptions: list[str]
    agent_compute: list[int]
    capacities: list[list[int]]
    links: list[list[int]]
    bandwidth: list[int]
    available: list[float]
    revision: str = ""
    agent_profiles: list[dict] = field(default_factory=list)


    def feasible(self, task: Task) -> list[int]:
        'Return agents satisfying skill, tool, and knowledge requirements.'
        return [i for i, skills in enumerate(self.skills)
                if task.skill in skills and set(task.tools) <= set(self.tools[i])
                and set(task.knowledge) <= set(self.knowledge[i])
                and all(d<=c for d,c in zip(task.demand,self.capacities[self.agent_compute[i]]))]

    def path(self, source: int, target: int) -> list[int] | None:
        'Return physical-link indices, preferring direct links and otherwise agent-only forwarding; return None when unreachable.'
        if source == target:
            return []
        n = len(self.skills)

        cache_key=(id(self.links),len(self.links),self.revision)
        if getattr(self,"_path_cache_key",None)!=cache_key:
            self._path_lookup={tuple(sorted(edge)): i for i, edge in enumerate(self.links)}
            self._path_adjacency={}
            for index,(left,right) in enumerate(self.links):
                self._path_adjacency.setdefault(left,[]).append((right,index))
                self._path_adjacency.setdefault(right,[]).append((left,index))
            self._paths={}
            self._path_cache_key=cache_key
        lookup=self._path_lookup
        direct=lookup.get(tuple(sorted((source,target))))
        if direct is not None:return [direct]
        pair=(source,target)
        if pair in self._paths:return self._paths[pair]
        queue=deque([source]);previous={source:None}
        while queue and target not in previous:
            vertex=queue.popleft()
            for neighbor,edge in self._path_adjacency.get(vertex,[]):
                if neighbor in previous or (neighbor>=n and neighbor!=target):continue
                previous[neighbor]=(vertex,edge);queue.append(neighbor)
        result=None
        if target in previous:
            result=[];vertex=target
            while vertex!=source:
                vertex,edge=previous[vertex];result.append(edge)
            result.reverse()
        self._paths[pair]=result
        self._paths[target,source]=None if result is None else list(reversed(result))
        return result


@dataclass
class Task:
    'One subtask in a workflow.'
    id: int
    text: str
    skill: str
    tools: list[str]
    knowledge: list[str]
    work: float
    demand: list[int]


@dataclass
class Workflow:
    'A directed acyclic workflow of subtasks and dependencies.'
    id: str
    tasks: list[Task]
    edges: list[list[float]]
    release: float
    deadline: float
    min_reliability: float = 0.9
    scenario: str = ""
    difficulty: str = ""
    template: str = ""


@dataclass
class Observation:
    'An observed execution or transfer used for training and evaluation.'
    kind: str
    task: int
    agent: int
    duration: float
    success: bool
    started: float
    finished: float
    attempt: int
    propensity: float
    target_task: int = -1
    target_agent: int = -1
    failure_reason: str = ""


@dataclass
class Example:
    'A snapshot, workflow, observations, and sample parameters.'
    snapshot: Snapshot
    workflow: Workflow
    observations: list[Observation]
    provenance: dict[str, Any]

    def inputs(self):
        'Return model-visible snapshot and workflow inputs without outcome observations.'
        return self.snapshot, self.workflow


@dataclass
class Prediction:
    'Candidate agents and predicted node/edge durations and success probabilities.'
    workflow_id: str
    candidates: dict[int, list[int]]
    nodes: dict[tuple[int, int], tuple[float, float]]
    edges: dict[tuple[int, int, int, int], tuple[float, float]]


@dataclass
class Plan:
    'A schedule returned by CP-SAT.'
    assignments: list[dict]
    transfers: list[dict]
    admitted: dict[str, bool]
    status: str
    elapsed: float
    reasons: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
