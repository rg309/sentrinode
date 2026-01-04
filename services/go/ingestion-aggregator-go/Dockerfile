FROM golang:1.24 as build
WORKDIR /src
COPY . .
RUN go build -o ingestion-aggregator

FROM gcr.io/distroless/base-debian12
WORKDIR /app
COPY --from=build /src/ingestion-aggregator ./ingestion-aggregator
EXPOSE 4317
ENTRYPOINT ["/app/ingestion-aggregator"]
