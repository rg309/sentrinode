#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai_artifacts import save_artifact


def main() -> None:
    df = pd.read_csv("training_data.csv")

    le_service = LabelEncoder()
    le_load = LabelEncoder()

    all_services = pd.concat([df["parent_service"], df["child_service"]])
    le_service.fit(all_services)

    df["parent_id"] = le_service.transform(df["parent_service"])
    df["child_id"] = le_service.transform(df["child_service"])
    df["load_id"] = le_load.fit_transform(df["system_load"])

    X = df[["parent_id", "child_id", "load_id", "parent_ms"]]
    y = df["target_ms"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = save_artifact(model, "latency_model")
    service_enc_path = save_artifact(le_service, "service_encoder")
    load_enc_path = save_artifact(le_load, "load_encoder")

    print("AI Training Complete!")
    print(f"Model Accuracy Score: {model.score(X_test, y_test):.2f}")
    print("Artifacts saved:")
    print(f" - Model: {model_path}")
    print(f" - Service encoder: {service_enc_path}")
    print(f" - Load encoder: {load_enc_path}")


if __name__ == "__main__":
    main()
