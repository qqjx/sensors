from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from agent.engine import build_planning_prompt, execute_heuristic_flow, initialize_governance_agent, run_agent_plan
from agent.perception import perceive_data_state
from agent.tools import get_working_df, set_working_frames
from catalog.snorkel_labeler import generate_probabilistic_labels
from processing.basic_tools import basic_fallback_processing
from schemas import SensorMetadata, SensorType


LOGGER = logging.getLogger(__name__)


def _canonical_name(sensor_type: SensorType) -> str:
    if sensor_type == SensorType.TEMPERATURE:
        return "temperature"
    if sensor_type == SensorType.PRESSURE:
        return "pressure"
    if sensor_type == SensorType.VIBRATION:
        return "vibration"
    if sensor_type == SensorType.CURRENT:
        return "current"
    if sensor_type == SensorType.VOLTAGE:
        return "voltage"
    return "unknown"


class IndustrialDataGovernancePipeline:
    def __init__(self, llm: Optional[object] = None, verbose: bool = False):
        self.agent = initialize_governance_agent(llm=llm, verbose=verbose)

    def _semantic_align(
        self, raw_df: pd.DataFrame, metadata_list: List[SensorMetadata]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        labels = generate_probabilistic_labels(metadata_list)

        semantic_map: Dict[str, str] = {}
        used_names: Dict[str, int] = {}
        for row in labels:
            original_name = str(row["field_name"])
            pred = row["predicted_label"]
            if not isinstance(pred, SensorType):
                pred = SensorType.ABSTAIN
            base = _canonical_name(pred)
            if base == "unknown":
                semantic_map[original_name] = original_name
                continue
            used_names[base] = used_names.get(base, 0) + 1
            mapped = base if used_names[base] == 1 else f"{base}_{used_names[base]}"
            semantic_map[original_name] = mapped

        aligned_df = raw_df.rename(columns=semantic_map)
        return aligned_df, semantic_map

    def run_pipeline(
        self, raw_df: pd.DataFrame, metadata_list: List[SensorMetadata], target_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        try:
            aligned_df, semantic_map = self._semantic_align(raw_df, metadata_list)
            data_state = perceive_data_state(aligned_df)
            semantic_summary = f"semantic_map={semantic_map}"
            prompt = build_planning_prompt(data_state=data_state, semantic_summary=semantic_summary)

            set_working_frames(aligned_df, target_df)
            if self.agent is not None:
                _ = run_agent_plan(self.agent, prompt)
                result_df = get_working_df()
                if result_df.empty:
                    raise RuntimeError("Agent returned an empty working DataFrame.")
                return result_df

            LOGGER.info("LangChain agent unavailable. Switch to heuristic execution.")
            return execute_heuristic_flow(aligned_df, data_state, target_df=target_df)
        except Exception as exc:
            LOGGER.exception("Pipeline failed (%s). Switch to fallback flow.", exc)
            return basic_fallback_processing(raw_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    import numpy as np

    demo_df = pd.DataFrame(
        {
            "temp_main": [36.1, 36.2, np.nan, 120.0, 36.4, 36.3],
            "press_line_a": [2.2, 2.3, 2.4, np.nan, 2.5, 9.9],
            "vib_motor_1": [0.03, 0.05, 0.04, 1.2, 0.06, 0.05],
        }
    )

    demo_meta = [
        SensorMetadata(field_name="temp_main", mean_value=36.0, unit="c", sampling_freq=10.0),
        SensorMetadata(field_name="press_line_a", mean_value=2.4, unit="mpa", sampling_freq=5.0),
        SensorMetadata(field_name="vib_motor_1", mean_value=0.05, unit="g", sampling_freq=5000.0),
    ]

    pipeline = IndustrialDataGovernancePipeline(llm=None, verbose=False)
    out = pipeline.run_pipeline(raw_df=demo_df, metadata_list=demo_meta)
    print(out.head())

