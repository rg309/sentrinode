FROM otel/opentelemetry-collector-contrib:0.94.0

COPY otel-railway.yaml /etc/otelcol-contrib/config.yaml

ENTRYPOINT ["/otelcol-contrib", "--config", "/etc/otelcol-contrib/config.yaml"]
