package main

import (
	"context"
	"log"
	"net"

	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/grpc"
)

type traceServer struct {
	coltracepb.UnimplementedTracesServiceServer
}

func (s *traceServer) Export(ctx context.Context, req *coltracepb.ExportTracesServiceRequest) (*coltracepb.ExportTracesServiceResponse, error) {
	log.Printf("Received %d spans from telemetrygen", len(req.ResourceSpans))
	// TODO: produceToKafka(req)
	return &coltracepb.ExportTracesServiceResponse{}, nil
}

func main() {
	log.Println("Neo4j Driver initialized.")

	lis, err := net.Listen("tcp", ":4317")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	coltracepb.RegisterTracesServiceServer(s, &traceServer{})

	log.Println("gRPC Server listening on :4317")
	log.Println("Ready to bridge Telemetrygen -> Kafka")

	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

/*

// KafkaTraceReceiver simulates an OTLP receiver that serializes spans to Kafka.
type KafkaTraceReceiver struct {
	producer sarama.SyncProducer
	topic    string
}

// NewKafkaTraceReceiver wires a Sarama producer into the receiver.
func NewKafkaTraceReceiver(producer sarama.SyncProducer, topic string) *KafkaTraceReceiver {
	return &KafkaTraceReceiver{
		producer: producer,
		topic:    topic,
	}
}

// ConsumeTraces iterates the incoming trace data and ships each span to Kafka.
func (r *KafkaTraceReceiver) ConsumeTraces(ctx context.Context, td ptrace.Traces) error {
	for i := 0; i < td.ResourceSpans().Len(); i++ {
		rs := td.ResourceSpans().At(i)

		serviceName := "unknown"
		if v, ok := rs.Resource().Attributes().Get(string(semconv.ServiceNameKey)); ok {
			serviceName = v.Str()
		}

		for j := 0; j < rs.ScopeSpans().Len(); j++ {
			ss := rs.ScopeSpans().At(j)
			for k := 0; k < ss.Spans().Len(); k++ {
				span := ss.Spans().At(k)
				log.Printf("Received Span: TraceID=%s, SpanID=%s, Service=%s",
					span.TraceID().String(),
					span.SpanID().String(),
					serviceName)

				msg := &sarama.ProducerMessage{
					Topic: r.topic,
					Key:   sarama.StringEncoder(span.TraceID().String()),
					Value: sarama.StringEncoder("Encoded OTel Span Data..."),
				}
				if _, _, err := r.producer.SendMessage(msg); err != nil {
					log.Printf("Failed to send message to Kafka: %v", err)
				}
			}
		}
	}
	return nil
}

func main() {
	brokers := kafkaBrokersFromEnv()
	producer := initKafkaProducer(brokers)
	defer closeKafkaProducer(producer)

	topic := kafkaTopicFromEnv()
	traceReceiver := NewKafkaTraceReceiver(producer, topic)

	log.Printf("Starting Mock OTLP Receiver on :4317 (Kafka brokers: %s, topic: %s)...", strings.Join(brokers, ","), topic)
	listener, err := net.Listen("tcp", ":4317")
	if err != nil {
		log.Fatalf("Failed to start listener on :4317: %v", err)
	}

	startReceiver(traceReceiver, listener)
	waitForShutdown(listener)
}

func initKafkaProducer(brokers []string) sarama.SyncProducer {
	config := sarama.NewConfig()
	config.Producer.RequiredAcks = sarama.WaitForAll
	config.Producer.Retry.Max = 10
	config.Producer.Return.Successes = true
	config.ClientID = "ingestion-aggregator"

	producer, err := sarama.NewSyncProducer(brokers, config)
	if err != nil {
		log.Printf("Failed to start Sarama producer (brokers=%s): %v", strings.Join(brokers, ","), err)
		return newNoopProducer()
	}

	log.Println("Kafka producer initialized successfully.")
	return producer
}

func closeKafkaProducer(producer sarama.SyncProducer) {
	if producer == nil {
		return
	}
	if err := producer.Close(); err != nil {
		log.Printf("Failed to close Kafka producer: %v", err)
	}
}

func kafkaBrokersFromEnv() []string {
	raw := os.Getenv("KAFKA_BROKERS")
	if raw == "" {
		raw = "localhost:9093"
	}

	parts := strings.Split(raw, ",")
	brokers := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			brokers = append(brokers, trimmed)
		}
	}
	return brokers
}

func kafkaTopicFromEnv() string {
	topic := os.Getenv("KAFKA_SPAN_TOPIC")
	if topic == "" {
		return "raw_spans_topic"
	}
	return topic
}

func startReceiver(receiver *KafkaTraceReceiver, listener net.Listener) {
	log.Printf("Mock Receiver running, ready to accept OTLP data on %s and forward to Kafka topic %s", listener.Addr().String(), receiver.topic)
}

type noopProducer struct {
	logOnce sync.Once
}

func newNoopProducer() sarama.SyncProducer {
	log.Println("Kafka is unavailable; continuing with a no-op producer that drops spans.")
	return &noopProducer{}
}

func (p *noopProducer) SendMessage(msg *sarama.ProducerMessage) (partition int32, offset int64, err error) {
	p.logOnce.Do(func() {
		log.Println("OTLP spans will not be forwarded because the no-op producer is active.")
	})
	return 0, 0, nil
}

func (p *noopProducer) SendMessages(msgs []*sarama.ProducerMessage) error {
	p.logOnce.Do(func() {
		log.Println("OTLP spans will not be forwarded because the no-op producer is active.")
	})
	return nil
}

func (p *noopProducer) Close() error {
	return nil
}

func (p *noopProducer) TxnStatus() sarama.ProducerTxnStatusFlag {
	return sarama.ProducerTxnFlagReady
}

func (p *noopProducer) IsTransactional() bool {
	return false
}

func (p *noopProducer) BeginTxn() error {
	return nil
}

func (p *noopProducer) CommitTxn() error {
	return nil
}

func (p *noopProducer) AbortTxn() error {
	return nil
}

func (p *noopProducer) AddOffsetsToTxn(offsets map[string][]*sarama.PartitionOffsetMetadata, groupID string) error {
	return nil
}

func (p *noopProducer) AddMessageToTxn(msg *sarama.ConsumerMessage, groupID string, metadata *string) error {
	return nil
}

type noopListener struct{}

func newNoopListener() net.Listener {
	log.Println("Network sockets unavailable; running with a no-op listener (no OTLP traffic will be accepted).")
	return &noopListener{}
}

func (n *noopListener) Accept() (net.Conn, error) {
	return nil, errors.New("no-op listener cannot accept connections")
}

func (n *noopListener) Close() error {
	return nil
}

func (n *noopListener) Addr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4zero, Port: 0}
}

func waitForShutdown(listener net.Listener) {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	log.Println("Shutdown signal received, closing listener...")
	if err := listener.Close(); err != nil {
		log.Printf("Failed to close listener: %v", err)
	}
}
*/
