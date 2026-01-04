package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"time"

	"github.com/IBM/sarama"
	"github.com/neo4j/neo4j-go-driver/v4/neo4j"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/grpc"
)

const (
	spanAnalyticsTopic = "span-analytics"
	missingLinkCypher  = `
        MATCH (child:Span), (parent:Span)
        WHERE child.parent_id <> "" AND child.parent_id = parent.id
          AND child.tenant_id = parent.tenant_id
          AND NOT (parent)-[:CALLS]->(child)
        MERGE (parent)-[:CALLS]->(child)
        RETURN count(*) AS linksCreated
    `
	spanUpsertCypher = `
        MERGE (s:Span {id: $spanID})
        ON CREATE SET s.created_at = timestamp()
        SET s.parent_id = $parentID,
            s.name = $name,
            s.duration_ns = $durationNs,
            s.latency_ms = $latencyMs,
            s.trace_id = $traceID,
            s.timestamp = timestamp(),
            s.tenant_id = $tenantID
        WITH s
        RETURN s
    `
)

var (
	driver                 neo4j.Driver
	kafkaConsumer          sarama.Consumer
	kafkaPartitionConsumer sarama.PartitionConsumer
	kafkaProducer          sarama.SyncProducer
	defaultTenantID        = getEnv("DEFAULT_TENANT_ID", "public")
	tenantKeyCandidates    = []string{"tenant_id", "org_id"}
)

type traceServer struct {
	coltracepb.UnimplementedTraceServiceServer
}

type spanJob struct {
	TraceID            string
	SpanID             string
	ParentID           string
	Name               string
	TenantID           string
	LatencyMs          float64
	DurationNano       int64
	StartTimeUnixNano  uint64
	EndTimeUnixNano    uint64
	Kind               string
	StatusCode         string
	StatusMessage      string
	Attributes         map[string]interface{}
	ResourceAttributes map[string]interface{}
	ScopeAttributes    map[string]interface{}
}

func buildSpanJob(span *tracepb.Span, resourceAttrs, scopeAttrs map[string]interface{}) *spanJob {
	if resourceAttrs == nil {
		resourceAttrs = map[string]interface{}{}
	}
	if scopeAttrs == nil {
		scopeAttrs = map[string]interface{}{}
	}

	traceID := bytesToHex(span.TraceId)
	spanID := bytesToHex(span.SpanId)
	parentID := bytesToHex(span.ParentSpanId)
	hasParent := len(span.ParentSpanId) > 0 && !isZero(span.ParentSpanId)
	log.Printf("RAW SPAN DATA - Name: %s, SpanID bytes: %v, ParentSpanID bytes: %v, ParentID hex: %s, hasParent: %v",
		span.GetName(), span.SpanId, span.ParentSpanId, parentID, hasParent)
	durationNs := computeDurationNs(span.StartTimeUnixNano, span.EndTimeUnixNano)
	latencyMs := computeLatencyMs(span.StartTimeUnixNano, span.EndTimeUnixNano)
	statusCode := ""
	statusMessage := ""
	if span.Status != nil {
		statusCode = span.Status.Code.String()
		statusMessage = span.Status.Message
	}

	attrMap := convertKeyValuesToMap(span.Attributes)

	return &spanJob{
		TraceID:            traceID,
		SpanID:             spanID,
		ParentID:           parentID,
		Name:               span.GetName(),
		LatencyMs:          latencyMs,
		DurationNano:       durationNs,
		StartTimeUnixNano:  span.StartTimeUnixNano,
		EndTimeUnixNano:    span.EndTimeUnixNano,
		Kind:               span.Kind.String(),
		StatusCode:         statusCode,
		StatusMessage:      statusMessage,
		Attributes:         attrMap,
		ResourceAttributes: resourceAttrs,
		ScopeAttributes:    scopeAttrs,
		TenantID:           determineTenantID(attrMap, resourceAttrs, scopeAttrs),
	}
}

func bytesToHex(value []byte) string {
	if len(value) == 0 {
		return ""
	}
	return hex.EncodeToString(value)
}

func isZero(id []byte) bool {
	for _, b := range id {
		if b != 0 {
			return false
		}
	}
	return true
}

func computeDurationNs(start, end uint64) int64 {
	if end <= start {
		return 0
	}
	diff := end - start
	if diff > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(diff)
}

func computeLatencyMs(start, end uint64) float64 {
	if end <= start {
		return 0
	}
	return float64(end-start) / 1_000_000.0
}

func convertKeyValuesToMap(attrs []*commonpb.KeyValue) map[string]interface{} {
	result := make(map[string]interface{}, len(attrs))
	for _, attr := range attrs {
		if attr == nil || attr.Value == nil {
			continue
		}
		result[attr.Key] = convertAnyValue(attr.Value)
	}
	return result
}

func convertAnyValue(value *commonpb.AnyValue) interface{} {
	if value == nil {
		return nil
	}

	switch v := value.Value.(type) {
	case *commonpb.AnyValue_StringValue:
		return v.StringValue
	case *commonpb.AnyValue_IntValue:
		return v.IntValue
	case *commonpb.AnyValue_DoubleValue:
		return v.DoubleValue
	case *commonpb.AnyValue_BoolValue:
		return v.BoolValue
	case *commonpb.AnyValue_BytesValue:
		return bytesToHex(v.BytesValue)
	case *commonpb.AnyValue_ArrayValue:
		return convertArrayValue(v.ArrayValue)
	case *commonpb.AnyValue_KvlistValue:
		return convertKeyValueList(v.KvlistValue)
	default:
		return nil
	}
}

func convertArrayValue(arrayValue *commonpb.ArrayValue) []interface{} {
	if arrayValue == nil {
		return nil
	}
	values := make([]interface{}, 0, len(arrayValue.Values))
	for _, item := range arrayValue.Values {
		values = append(values, convertAnyValue(item))
	}
	return values
}

func convertKeyValueList(kvList *commonpb.KeyValueList) map[string]interface{} {
	if kvList == nil {
		return map[string]interface{}{}
	}
	return convertKeyValuesToMap(kvList.Values)
}

func getEnv(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

func determineTenantID(attrMaps ...map[string]interface{}) string {
	for _, attrs := range attrMaps {
		for _, candidate := range tenantKeyCandidates {
			if val, ok := extractStringAttr(attrs, candidate); ok {
				return val
			}
		}
	}
	return defaultTenantID
}

func extractStringAttr(attrs map[string]interface{}, key string) (string, bool) {
	if attrs == nil {
		return "", false
	}
	val, ok := attrs[key]
	if !ok {
		return "", false
	}
	switch typed := val.(type) {
	case string:
		if typed == "" {
			return "", false
		}
		return typed, true
	case fmt.Stringer:
		return typed.String(), true
	default:
		str := fmt.Sprintf("%v", val)
		if str == "" || str == "<nil>" {
			return "", false
		}
		return str, true
	}
}

func convertInstrumentationScope(scope *commonpb.InstrumentationScope) map[string]interface{} {
	if scope == nil {
		return map[string]interface{}{}
	}

	scopeMap := map[string]interface{}{
		"name":                     scope.Name,
		"version":                  scope.Version,
		"dropped_attributes_count": scope.DroppedAttributesCount,
	}

	for key, value := range convertKeyValuesToMap(scope.Attributes) {
		scopeMap[key] = value
	}

	return scopeMap
}

func (s *traceServer) Export(ctx context.Context, req *coltracepb.ExportTraceServiceRequest) (*coltracepb.ExportTraceServiceResponse, error) {
	if req == nil {
		return &coltracepb.ExportTraceServiceResponse{}, nil
	}

	totalSpans := 0
	for _, rs := range req.ResourceSpans {
		if rs == nil {
			continue
		}

		resourceAttributes := map[string]interface{}{}
		if rs.Resource != nil {
			resourceAttributes = convertKeyValuesToMap(rs.Resource.Attributes)
		}

		for _, ss := range rs.ScopeSpans {
			if ss == nil {
				continue
			}

			scopeAttributes := convertInstrumentationScope(ss.Scope)

			for _, span := range ss.Spans {
				if span == nil {
					continue
				}

				job := buildSpanJob(span, resourceAttributes, scopeAttributes)
				log.Printf("Span: %s, Parent: %s", job.Name, job.ParentID)

				if err := persistSpan(job); err != nil {
					log.Printf("Failed to persist span %s: %v", job.SpanID, err)
					continue
				}

				if err := publishSpanAnalytics(job); err != nil {
					log.Printf("Failed to publish span %s to Kafka: %v", job.SpanID, err)
				}

				totalSpans++
			}
		}
	}

	fmt.Printf("Total Spans in this batch: %d\n", totalSpans)
	return &coltracepb.ExportTraceServiceResponse{}, nil
}

func initNeo4j() {
	uri := "bolt://localhost:7687"
	user := "neo4j"
	password := "Delahrg12"

	var err error
	driver, err = neo4j.NewDriver(uri, neo4j.BasicAuth(user, password, ""))
	if err != nil {
		log.Fatalf("Failed to create Neo4j driver: %v", err)
	}
	ensureSpanConstraint()
	log.Println("Neo4j Driver initialized.")
}

func ensureSpanConstraint() {
	session := driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		return tx.Run(`CREATE CONSTRAINT IF NOT EXISTS FOR (s:Span) REQUIRE s.id IS UNIQUE`, nil)
	})
	if err != nil {
		log.Fatalf("Failed to ensure Span ID constraint: %v", err)
	}
}

func initKafka() {
	brokers := []string{"localhost:9093"}

	consumerConfig := sarama.NewConfig()
	consumerConfig.Consumer.Offsets.Initial = sarama.OffsetOldest

	var err error
	kafkaConsumer, err = sarama.NewConsumer(brokers, consumerConfig)
	if err != nil {
		log.Fatalf("Failed to start Kafka consumer: %v", err)
	}

	kafkaPartitionConsumer, err = kafkaConsumer.ConsumePartition("raw_spans_topic", 0, sarama.OffsetOldest)
	if err != nil {
		log.Fatalf("Failed to consume partition: %v", err)
	}

	producerConfig := sarama.NewConfig()
	producerConfig.Producer.RequiredAcks = sarama.WaitForAll
	producerConfig.Producer.Retry.Max = 5
	producerConfig.Producer.Return.Successes = true

	kafkaProducer, err = sarama.NewSyncProducer(brokers, producerConfig)
	if err != nil {
		log.Fatalf("Failed to start Kafka producer: %v", err)
	}
	log.Printf("Kafka producer ready for topic %q", spanAnalyticsTopic)
}

func startKafkaConsumer() {
	if kafkaConsumer == nil || kafkaPartitionConsumer == nil {
		log.Fatalf("Kafka consumer is not initialized")
	}
	defer driver.Close()
	defer kafkaConsumer.Close()
	defer kafkaPartitionConsumer.Close()

	for {
		select {
		case msg := <-kafkaPartitionConsumer.Messages():
			if msg == nil {
				continue
			}

			spanData := string(msg.Value)
			if spanData == "" {
				spanData = "<empty-span>"
			}

			key := "<nil>"
			if msg.Key != nil {
				key = string(msg.Key)
			}

			sourceService := fmt.Sprintf("Service_%s", key)
			targetService := "Database_DUMMY"

			if err := createCausalRelationship(sourceService, targetService); err != nil {
				log.Printf("Error processing message (Key: %s): %v", key, err)
			} else {
				log.Printf("Processed span data '%s' for relationship: %s -> %s", spanData, sourceService, targetService)
			}
		case err := <-kafkaPartitionConsumer.Errors():
			if err != nil {
				log.Printf("Kafka Consumer Error: %v", err)
			}
		}
	}
}

func persistSpan(job *spanJob) error {
	session := driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	if job.SpanID == "" {
		return fmt.Errorf("span has empty ID, skipping persistence")
	}

	params := map[string]interface{}{
		"spanID":     job.SpanID,
		"parentID":   job.ParentID,
		"name":       job.Name,
		"durationNs": job.DurationNano,
		"latencyMs":  job.LatencyMs,
		"traceID":    job.TraceID,
		"tenantID":   job.TenantID,
	}

	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(spanUpsertCypher, params)
		if err != nil {
			return nil, err
		}
		return result.Consume()
	})
	return err
}

func publishSpanAnalytics(job *spanJob) error {
	if kafkaProducer == nil {
		return fmt.Errorf("kafka producer is not initialized")
	}

	flatSpan := flattenSpan(job)
	payload, err := json.Marshal(flatSpan)
	if err != nil {
		return fmt.Errorf("marshal span: %w", err)
	}

	key := job.TraceID
	if key == "" {
		key = job.SpanID
	}

	_, _, err = kafkaProducer.SendMessage(&sarama.ProducerMessage{
		Topic: spanAnalyticsTopic,
		Key:   sarama.StringEncoder(key),
		Value: sarama.ByteEncoder(payload),
	})
	return err
}

func flattenSpan(job *spanJob) map[string]interface{} {
	data := map[string]interface{}{
		"trace_id":             job.TraceID,
		"span_id":              job.SpanID,
		"parent_span_id":       job.ParentID,
		"name":                 job.Name,
		"tenant_id":            job.TenantID,
		"latency_ms":           job.LatencyMs,
		"duration_ns":          job.DurationNano,
		"start_time_unix_nano": job.StartTimeUnixNano,
		"end_time_unix_nano":   job.EndTimeUnixNano,
		"kind":                 job.Kind,
		"status_code":          job.StatusCode,
		"status_message":       job.StatusMessage,
	}

	for key, value := range job.ResourceAttributes {
		data["resource."+key] = value
	}

	for key, value := range job.ScopeAttributes {
		data["scope."+key] = value
	}

	for key, value := range job.Attributes {
		data["attr."+key] = value
	}

	return data
}

func main() {
	// 1. Setup Dependencies
	initNeo4j()
	initKafka()
	defer kafkaProducer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startSpanLinker(ctx, 30*time.Second)

	// 2. Start the OTLP Receiver in a GOROUTINE (Non-blocking)
	go func() {
		lis, err := net.Listen("tcp", ":4317")
		if err != nil {
			log.Fatalf("Failed to listen: %v", err)
		}
		s := grpc.NewServer()
		coltracepb.RegisterTraceServiceServer(s, &traceServer{})
		log.Println("OTLP gRPC Server started on :4317")
		if err := s.Serve(lis); err != nil {
			log.Fatalf("gRPC failed: %v", err)
		}
	}()

	// 3. Start the Kafka Consumer (This is the blocking call)
	log.Println("Graph Aggregator is listening on 'raw_spans_topic'...")
	startKafkaConsumer() // This will now block here, but OTLP is already running in the background!
}

func createCausalRelationship(source, target string) error {
	session := driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	cypher := `
        MERGE (source:Service {name: $source})
        MERGE (target:Resource {name: $target})
        MERGE (source)-[r:CALLS]->(target)
        ON CREATE SET r.creationTime = timestamp()
        ON MATCH SET r.lastCalled = timestamp()
        RETURN source, r, target
    `

	params := map[string]interface{}{
		"source": source,
		"target": target,
	}

	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(cypher, params)
		if err != nil {
			return nil, err
		}
		return result.Consume()
	})

	return err
}

func startSpanLinker(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 30 * time.Second
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				log.Println("Span linker stopped")
				return
			case <-ticker.C:
				linkMissingRelationships()
			}
		}
	}()
}

func linkMissingRelationships() {
	session := driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(missingLinkCypher, nil)
		if err != nil {
			return nil, err
		}
		record, err := result.Single()
		if err != nil {
			return nil, err
		}
		if record != nil {
			if links, ok := record.Get("linksCreated"); ok {
				if count, ok := links.(int64); ok && count > 0 {
					log.Printf("Linker created %d CALLS relationships", count)
				}
			}
		}
		return nil, nil
	})

	if err != nil {
		log.Printf("Span linker failed: %v", err)
	}
}
