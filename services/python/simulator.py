#!/usr/bin/env python3
import pandas as pd

from ai_artifacts import load_latest_artifact

# Load the brain
model = load_latest_artifact("latency_model")
le_service = load_latest_artifact("service_encoder")
le_load = load_latest_artifact("load_encoder")


def simulate_environment(load_level):
    print(f"\n--- Simulating {load_level} Load Environment ---")

    # Define a hypothetical scenario
    scenarios = [
        {"parent": "lets-go", "child": "okey-dokey-0", "p_ms": 0.5},
        {"parent": "lets-go", "child": "okey-dokey-1", "p_ms": 0.5},
        {"parent": "lets-go", "child": "okey-dokey-2", "p_ms": 0.5},
    ]

    for s in scenarios:
        p_id = le_service.transform([s["parent"]])[0]
        c_id = le_service.transform([s["child"]])[0]
        l_id = le_load.transform([load_level])[0]

        input_df = pd.DataFrame(
            [[p_id, c_id, l_id, s["p_ms"]]],
            columns=["parent_id", "child_id", "load_id", "parent_ms"],
        )

        pred = model.predict(input_df)[0]
        print(f"Prediction: {s['child']} -> {pred:.2f}ms")


# Run the simulation
simulate_environment("Low")
simulate_environment("High")
