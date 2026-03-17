from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.main_pipeline import IndustrialDataGovernancePipeline
from schemas import SensorMetadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    raw_df = pd.DataFrame(
        {
            "temp_main": [36.5, 36.6, 36.4, 200.0, np.nan, 36.7, 36.8],
            "press_line_a": [2.1, 2.2, np.nan, 2.4, 2.3, 10.0, 2.5],
            "vib_motor_1": [0.04, 0.05, 0.04, 0.06, 1.5, 0.05, 0.04],
        }
    )
    target_df = pd.DataFrame(
        {
            "temp_main": [35.9, 36.0, 36.1, 36.2, 36.0, 36.1, 36.2],
            "press_line_a": [2.0, 2.1, 2.2, 2.2, 2.1, 2.0, 2.3],
            "vib_motor_1": [0.04, 0.05, 0.05, 0.06, 0.04, 0.05, 0.04],
        }
    )
    metadata_list = [
        SensorMetadata(field_name="temp_main", mean_value=36.5, unit="c", sampling_freq=10.0),
        SensorMetadata(field_name="press_line_a", mean_value=2.4, unit="mpa", sampling_freq=5.0),
        SensorMetadata(field_name="vib_motor_1", mean_value=0.06, unit="g", sampling_freq=4000.0),
    ]

    pipeline = IndustrialDataGovernancePipeline(llm=None, verbose=True)
    result = pipeline.run_pipeline(raw_df=raw_df, metadata_list=metadata_list, target_df=target_df)
    print(result)


if __name__ == "__main__":
    main()

