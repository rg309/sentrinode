package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
)

const (
	defaultServiceName = "telemetrygen"
	defaultSpanName    = "telemetrygen.request"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "traces":
		if err := runTraces(os.Args[2:]); err != nil {
			log.Fatalf("telemetrygen traces: %v", err)
		}
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprint(os.Stderr, `telemetrygen - lightweight synthetic OTLP generator

Usage:
  telemetrygen traces [flags]

Flags:
  --traces <n>           Number of traces/spans to emit (default 25)
  --otlp-endpoint <h:p>  OTLP gRPC endpoint (default "localhost:4317")
  --service-name <name>  Sets resource service.name (default "telemetrygen")
  --span-duration <dur>  Base span duration (default 75ms)
  --concurrency <n>      Number of workers emitting spans (default 4)
`)
}

type tracesConfig struct {
	total       int
	endpoint    string
	service     string
	concurrency int
	spanDur     time.Duration
}

func runTraces(args []string) error {
	fs := flag.NewFlagSet("telemetrygen traces", flag.ContinueOnError)
	cfg := tracesConfig{}
	fs.IntVar(&cfg.total, "traces", 25, "Total traces to emit")
	fs.StringVar(&cfg.endpoint, "otlp-endpoint", "localhost:4317", "OTLP gRPC endpoint (host:port)")
	fs.StringVar(&cfg.service, "service-name", defaultServiceName, "Value for resource service.name")
	fs.IntVar(&cfg.concurrency, "concurrency", 4, "Number of concurrent workers emitting spans")
	fs.DurationVar(&cfg.spanDur, "span-duration", 75*time.Millisecond, "Base synthetic span duration")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if cfg.total <= 0 {
		return errors.New("--traces must be greater than zero")
	}
	if cfg.concurrency <= 0 {
		return errors.New("--concurrency must be greater than zero")
	}

	return generateTraces(cfg)
}

func generateTraces(cfg tracesConfig) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client := otlptracegrpc.NewClient(
		otlptracegrpc.WithEndpoint(cfg.endpoint),
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithDialOption(grpc.WithBlock()),
	)
	exporter, err := otlptrace.New(ctx, client)
	if err != nil {
		return fmt.Errorf("create otlp exporter: %w", err)
	}

	res, err := resource.New(
		context.Background(),
		resource.WithSchemaURL(semconv.SchemaURL),
		resource.WithAttributes(
			semconv.ServiceName(cfg.service),
			attribute.String("telemetrygen.emitter", "local"),
		),
	)
	if err != nil {
		return fmt.Errorf("create resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)
	defer func() {
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		if err := tp.Shutdown(shutdownCtx); err != nil {
			log.Printf("failed to shutdown telemetrygen tracer provider: %v", err)
		}
	}()

	otel.SetTracerProvider(tp)
	tracer := tp.Tracer("telemetrygen")

	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < cfg.concurrency; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(worker)))
			for seq := range jobs {
				emitSyntheticSpan(tracer, seq, cfg.spanDur, rng)
			}
		}(w)
	}

	for i := 0; i < cfg.total; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	flushCtx, flushCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer flushCancel()
	if err := tp.ForceFlush(flushCtx); err != nil {
		return fmt.Errorf("flush spans: %w", err)
	}

	log.Printf("Emitted %d traces to %s (service.name=%s)", cfg.total, cfg.endpoint, cfg.service)
	return nil
}

func emitSyntheticSpan(tracer trace.Tracer, seq int, baseDuration time.Duration, rng *rand.Rand) {
	rootCtx, root := tracer.Start(
		context.Background(),
		fmt.Sprintf("%s.root.%d", defaultSpanName, seq),
		trace.WithAttributes(
			attribute.Int("telemetrygen.sequence", seq),
			attribute.String("telemetrygen.request_id", fmt.Sprintf("tg-%d", seq)),
		),
	)

	childWork := baseDuration/2 + time.Duration(rng.Intn(50))*time.Millisecond
	_, childA := tracer.Start(
		rootCtx,
		fmt.Sprintf("%s.childA.%d", defaultSpanName, seq),
		trace.WithAttributes(attribute.String("telemetrygen.child_role", "frontend")),
	)
	time.Sleep(childWork)
	childA.SetAttributes(
		attribute.String("telemetrygen.customer_tier", randomTier(rng)),
		attribute.Int64("telemetrygen.duration_ms", childWork.Milliseconds()),
	)
	childA.End()

	childWorkB := baseDuration/2 + time.Duration(rng.Intn(50))*time.Millisecond
	_, childB := tracer.Start(
		rootCtx,
		fmt.Sprintf("%s.childB.%d", defaultSpanName, seq),
		trace.WithAttributes(attribute.String("telemetrygen.child_role", "backend")),
	)
	time.Sleep(childWorkB)
	childB.SetAttributes(
		attribute.String("telemetrygen.customer_tier", randomTier(rng)),
		attribute.Int64("telemetrygen.duration_ms", childWorkB.Milliseconds()),
	)
	childB.End()

	rootDuration := childWork + childWorkB + time.Duration(rng.Intn(25))*time.Millisecond
	root.SetAttributes(
		attribute.String("telemetrygen.customer_tier", randomTier(rng)),
		attribute.Int64("telemetrygen.duration_ms", rootDuration.Milliseconds()),
	)
	root.AddEvent("telemetrygen.work_complete", trace.WithAttributes(
		attribute.Int("telemetrygen.random_jitter_ms", rng.Intn(20)),
	))
	root.End()
}

var tiers = []string{"bronze", "silver", "gold", "platinum"}

func randomTier(rng *rand.Rand) string {
	return tiers[rng.Intn(len(tiers))]
}
