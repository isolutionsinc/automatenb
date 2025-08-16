#!/bin/bash

# CloudWatch Metrics Helper Script
# Functions to send custom metrics to CloudWatch

NAMESPACE="PM2/CodeBox"
REGION="us-east-1"

# Function to send metric to CloudWatch
send_metric() {
    local metric_name=$1
    local value=$2
    local unit=$3
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
    
    if command -v aws &> /dev/null; then
        aws cloudwatch put-metric-data \
            --namespace "$NAMESPACE" \
            --metric-data MetricName="$metric_name",Value="$value",Unit="$unit",Timestamp="$timestamp" \
            --region "$REGION" 2>/dev/null || true
    fi
}

# Function to send service status metric
send_service_status() {
    local status=$1  # 1 for online, 0 for offline
    send_metric "ServiceStatus" "$status" "None"
}

# Function to send health check failure metric
send_health_check_failure() {
    send_metric "HealthCheckFailures" "1" "Count"
}

# Function to send service restart metric
send_service_restart() {
    send_metric "ServiceRestarts" "1" "Count"
}

# Function to send response time metric
send_response_time() {
    local response_time=$1  # in milliseconds
    send_metric "ResponseTime" "$response_time" "Milliseconds"
}

# Function to check health endpoint and measure response time
check_health_with_timing() {
    local endpoint=$1
    local start_time=$(date +%s%3N)
    local response_code
    
    response_code=$(curl -s -w "%{http_code}" -o /dev/null -m 10 "$endpoint" 2>/dev/null)
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    if [ "$response_code" = "200" ]; then
        send_service_status 1
        send_response_time "$response_time"
        return 0
    else
        send_service_status 0
        send_health_check_failure
        return 1
    fi
}

# Export functions for use in other scripts
export -f send_metric
export -f send_service_status
export -f send_health_check_failure
export -f send_service_restart
export -f send_response_time
export -f check_health_with_timing
