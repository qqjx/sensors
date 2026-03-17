from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from schemas import SensorMetadata, SensorType

try:
    from snorkel.labeling import LFApplier, labeling_function
    from snorkel.labeling.model import LabelModel

    HAS_SNORKEL = True
except Exception:
    HAS_SNORKEL = False

    def labeling_function(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        return _decorator


def _contains_any(text: str, keywords: List[str]) -> bool:
    base = (text or "").lower()
    return any(k in base for k in keywords)


@labeling_function()
def lf_keyword_temp(x: SensorMetadata) -> int:
    if _contains_any(x.field_name, ["temp", "temperature", "wendu", "deg"]):
        return int(SensorType.TEMPERATURE)
    return int(SensorType.ABSTAIN)


@labeling_function()
def lf_keyword_pressure(x: SensorMetadata) -> int:
    if _contains_any(x.field_name, ["press", "pressure", "pa"]):
        return int(SensorType.PRESSURE)
    return int(SensorType.ABSTAIN)


@labeling_function()
def lf_value_unit_temp(x: SensorMetadata) -> int:
    unit = (x.unit or "").lower()
    if -20.0 <= x.mean_value <= 150.0 and unit in {"c", "celsius"}:
        return int(SensorType.TEMPERATURE)
    return int(SensorType.ABSTAIN)


@labeling_function()
def lf_value_unit_pressure(x: SensorMetadata) -> int:
    unit = (x.unit or "").lower()
    if 0.0 <= x.mean_value <= 10.0 and unit in {"mpa", "bar", "kpa"}:
        return int(SensorType.PRESSURE)
    return int(SensorType.ABSTAIN)


@labeling_function()
def lf_freq_vibration(x: SensorMetadata) -> int:
    is_vib_name = _contains_any(x.field_name, ["vib", "vibration", "accel"])
    if x.sampling_freq > 1000.0 and is_vib_name:
        return int(SensorType.VIBRATION)
    return int(SensorType.ABSTAIN)


ALL_LFS = [
    lf_keyword_temp,
    lf_keyword_pressure,
    lf_value_unit_temp,
    lf_value_unit_pressure,
    lf_freq_vibration,
]


def _label_name_from_index(index: int) -> SensorType:
    try:
        return SensorType(index)
    except ValueError:
        return SensorType.ABSTAIN


def _fallback_vote_labeling(data_records: List[SensorMetadata]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    lf_funcs = [lf_keyword_temp, lf_keyword_pressure, lf_value_unit_temp, lf_value_unit_pressure, lf_freq_vibration]

    for item in data_records:
        votes = [f(item) for f in lf_funcs]
        valid_votes = [v for v in votes if v != int(SensorType.ABSTAIN)]
        if not valid_votes:
            pred = SensorType.ABSTAIN
            confidence = 0.0
            probs = {st.name: 0.0 for st in SensorType if st != SensorType.ABSTAIN}
        else:
            count = Counter(valid_votes)
            pred_idx, vote_count = count.most_common(1)[0]
            pred = _label_name_from_index(pred_idx)
            confidence = vote_count / len(votes)
            probs = {st.name: 0.0 for st in SensorType if st != SensorType.ABSTAIN}
            probs[pred.name] = confidence
        results.append(
            {
                "field_name": item.field_name,
                "predicted_label": pred,
                "confidence": float(confidence),
                "probs": probs,
            }
        )
    return results


def generate_probabilistic_labels(
    data_records: List[SensorMetadata], cardinality: int = 5, epochs: int = 500, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Apply labeling functions and infer probabilistic semantic labels.

    Returns one record per metadata item with:
    - field_name
    - predicted_label (SensorType)
    - confidence (float)
    - probs (dict from class name to probability)
    """
    if not data_records:
        return []

    if not HAS_SNORKEL:
        return _fallback_vote_labeling(data_records)

    applier = LFApplier(lfs=ALL_LFS)
    l_matrix = applier.apply(data_points=data_records)

    label_model = LabelModel(cardinality=cardinality, verbose=False)
    label_model.fit(L_train=l_matrix, n_epochs=epochs, log_freq=100, seed=seed)

    probas = label_model.predict_proba(L=l_matrix)
    preds = probas.argmax(axis=1)
    confs = probas.max(axis=1)

    results: List[Dict[str, Any]] = []
    class_names = [SensorType(i).name for i in range(cardinality)]
    for idx, item in enumerate(data_records):
        confidence = float(confs[idx])
        predicted = _label_name_from_index(int(preds[idx]))
        if confidence < 0.5:
            predicted = SensorType.ABSTAIN
        row_probs = {class_names[i]: float(probas[idx, i]) for i in range(probas.shape[1])}
        results.append(
            {
                "field_name": item.field_name,
                "predicted_label": predicted,
                "confidence": confidence,
                "probs": row_probs,
            }
        )
    return results


if __name__ == "__main__":
    demo_data = [
        SensorMetadata(field_name="temp_main", mean_value=36.5, unit="c", sampling_freq=10.0),
        SensorMetadata(field_name="press_line_a", mean_value=2.6, unit="mpa", sampling_freq=5.0),
        SensorMetadata(field_name="vib_motor_1", mean_value=0.05, unit="g", sampling_freq=5000.0),
    ]

    labels = generate_probabilistic_labels(demo_data)
    for row in labels:
        print(
            f"field={row['field_name']}, label={row['predicted_label'].name}, "
            f"confidence={row['confidence']:.3f}"
        )
