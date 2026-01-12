#!/bin/bash
# SentriNode Cloud-Native Installer

echo "🚀 Starting SentriNode Installation..."

# 1. Capture API Key
API_KEY=$1
if [ -z "$API_KEY" ]; then
    echo "❌ Error: Missing API Key. Usage: curl ... | bash -s -- YOUR_KEY"
    exit 1
fi

# 2. Setup directory
mkdir -p ~/.sentrinode
cd ~/.sentrinode

# 3. Create the Docker Compose file locally
cat <<EOF > docker-compose.yml
version: '3.8'
services:
  sentrinode-agent:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: sentrinode-agent
    # Run as root so it can read system logs
    user: "0:0"
    command: ["--config=/etc/otel-config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otel-config.yaml
      # These 3 lines allow it to see your Mac's health & logs
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /var/log:/var/log:ro
      - /:/hostfs:ro
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "14250:14250" # Jaeger
      - "9411:9411"   # Zipkin
      - "8888:8888"   # Prometheus Metrics
EOF

# 4. Create the OTel Config (Pointing to your Railway URL)
cat <<EOF > otel-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:
exporters:
  otlp/sentrinode:
    endpoint: "sentrinode-production.up.railway.app:4317" # Replace with your REAL Railway URL
    tls:
      insecure: true
    headers:
      "x-sentrinode-key": "$API_KEY"
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp/sentrinode]
EOF

# 5. Start the agent
docker compose up -d
echo "✅ SentriNode Agent is running locally and connected to your cloud!"

# Add a 'sentrinode' command to the user's Mac
create_cli_shortcut() {
    # Save the key so 'update' can find it later
    echo "$1" > ~/.sentrinode_key

    sudo tee /usr/local/bin/sentrinode > /dev/null <<'EOF'
#!/bin/bash
# Read the saved key
KEY=$(cat ~/.sentrinode_key 2>/dev/null)

if [ "$1" == "update" ]; then
    if [ -z "$KEY" ]; then
        echo "❌ Error: No saved API Key found. Please run the full curl install again."
        exit 1
    fi
    echo "🔄 Updating SentriNode Agent..."
    curl -sSL https://raw.githubusercontent.com/rg309/sentrinode/main/install.sh | bash -s -- "$KEY"
else
    echo "SentriNode CLI"
    echo "Usage: sentrinode update"
fi
EOF
    sudo chmod +x /usr/local/bin/sentrinode
}

# Run it during installation
create_cli_shortcut "$API_KEY"
