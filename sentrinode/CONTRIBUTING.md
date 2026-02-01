# Contributing

## Deployment guardrail

UI changes must be deployed only to the `sentrinode` Fly app. The `sentrinode-api` Fly app must never be overwritten by the UI deploy; it should always be built from `Dockerfile.api` and run the FastAPI service.
