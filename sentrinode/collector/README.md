OpenTelemetry Collector (OTLP/HTTP)
==============================================

This service exposes an OTLP HTTP endpoint on port 4318 and writes all received traces, metrics, and logs to the collector's logging exporter so they show up in platform logs.

Setup on Fly
------------
1) Create a new Fly app from this repo and set the `collector/` folder as the root (it contains the Dockerfile). Deploy it.
2) Expose port 4318 publicly so the Fly domain forwards to the collector.
3) After deployment, use the public domain to send OTLP/HTTP requests and watch the Fly logs for the logging exporter output.

Test commands
-------------
Replace `<collector-domain>` with the Fly service domain.

- Traces (`/v1/traces`, expected response `{"partialSuccess":{}}`):

```bash
curl -X POST "https://<collector-domain>/v1/traces" \
  -H "Content-Type: application/json" \
  -d '{
        "resourceSpans": [{
          "resource": { "attributes": [{ "key": "service.name", "value": { "stringValue": "sentrinode-demo" } }] },
          "scopeSpans": [{
            "scope": { "name": "sample" },
            "spans": [{
              "traceId": "00000000000000000000000000000001",
              "spanId": "0000000000000001",
              "name": "demo-span",
              "kind": "SPAN_KIND_INTERNAL",
              "startTimeUnixNano": "1697040000000000000",
              "endTimeUnixNano": "1697040001000000000",
              "status": {}
            }]
          }]
        }]
      }'
```

- Metrics (`/v1/metrics`):

```bash
curl -X POST "https://<collector-domain>/v1/metrics" \
  -H "Content-Type: application/json" \
  -d '{
        "resourceMetrics": [{
          "resource": { "attributes": [{ "key": "service.name", "value": { "stringValue": "sentrinode-demo" } }] },
          "scopeMetrics": [{
            "scope": { "name": "sample" },
            "metrics": [{
              "name": "demo_gauge",
              "description": "sample gauge metric",
              "gauge": {
                "dataPoints": [{
                  "asDouble": 1.23,
                  "timeUnixNano": "1697040002000000000",
                  "attributes": [{ "key": "env", "value": { "stringValue": "test" } }]
                }]
              }
            }]
          }]
        }]
      }'
```
