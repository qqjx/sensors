from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from agent.tools import build_langchain_tools
from processing.advanced_tools import lstm_time_series_correction, mmd_distribution_normalization
from processing.basic_tools import iqr_anomaly_correction, knn_imputation

try:
    from langchain.agents import AgentType, initialize_agent

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False


SYSTEM_PROMPT = """
You are an industrial sensor data-governance planner.
You must choose and chain tools based on incoming data state:
1) Noise-Aware Normalization: prioritize outlier detection/correction.
2) Distribution-Aware Normalization: align source/target distributions.
3) Multi-Stage Processing: imputation -> noise correction -> distribution alignment.

Decision rules:
- If missing_ratio is high, prioritize knn_imputation before downstream steps.
- If has_outliers is true or noise_level is high, run iqr_anomaly_correction early.
- If data comes from heterogeneous sensors with distribution drift, run mmd_distribution_normalization.
- For strong temporal dependency signals, run lstm_time_series_correction as advanced stage.

Always explain the chosen strategy and order of tools in short, concrete steps.
""".strip()


def build_planning_prompt(data_state: Dict[str, object], semantic_summary: str) -> str:
    return (
        f"Semantic context: {semantic_summary}\n"
        f"Data state: {data_state}\n"
        "Select one of the three strategies (or combine them) and provide an execution plan."
    )


def initialize_governance_agent(llm: Optional[object] = None, verbose: bool = False) -> Optional[object]:
    """
    Initialize a LangChain ZERO_SHOT_REACT_DESCRIPTION agent.
    If LangChain/LLM is unavailable, returns None and pipeline uses heuristic fallback.
    """
    if not HAS_LANGCHAIN or llm is None:
        return None

    tools = build_langchain_tools()
    if not tools:
        return None

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        agent_kwargs={"prefix": SYSTEM_PROMPT},
    )
    return agent


def heuristic_strategy_plan(data_state: Dict[str, object], has_target: bool) -> List[str]:
    steps: List[str] = []
    missing_ratio = float(data_state.get("missing_ratio", 0.0))
    noise_level = str(data_state.get("noise_level", "low"))
    has_outliers = bool(data_state.get("has_outliers", False))

    if missing_ratio > 0:
        steps.append("knn_imputation")
    if has_outliers or noise_level in {"medium", "high"}:
        steps.append("iqr_anomaly_correction")
    if has_target:
        steps.append("mmd_distribution_normalization")
    if noise_level == "high":
        steps.append("lstm_time_series_correction")
    return steps


def execute_heuristic_flow(
    df: pd.DataFrame, data_state: Dict[str, object], target_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    out = df.copy()
    steps = heuristic_strategy_plan(data_state, has_target=target_df is not None)
    for step in steps:
        if step == "knn_imputation":
            out = knn_imputation(out)
        elif step == "iqr_anomaly_correction":
            out = iqr_anomaly_correction(out)
        elif step == "mmd_distribution_normalization" and target_df is not None:
            out = mmd_distribution_normalization(out, target_df)
        elif step == "lstm_time_series_correction":
            out = lstm_time_series_correction(out)
    return out


def run_agent_plan(agent: object, prompt: str) -> str:
    if hasattr(agent, "run"):
        return str(agent.run(prompt))
    if hasattr(agent, "invoke"):
        result = agent.invoke({"input": prompt})
        return str(result)
    return "Agent object does not support run/invoke."
