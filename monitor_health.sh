#!/bin/bash

# PM2 Health Monitoring Script
# This script monitors the PM2 service and restarts it if unresponsive

# Source CloudWatch metrics functions
source /home/ubuntu/automatenb/cloudwatch_metrics.sh 2>/dev/null || true

# Configuration
PM2_APP_NAME="codebox"
HEALTH_ENDPOINT="http://localhost:5000/api-health"
LOG_FILE="/var/log/pm2-health-monitor.log"
MAX_RETRIES=3
RETRY_INTERVAL=10
RESTART_COOLDOWN=60

# Create log file if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || LOG_FILE="/tmp/pm2-health-monitor.log"

# Logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE" 2>/dev/null || echo "[$timestamp] [$level] $message"
}

# Function to check if PM2 process exists
check_pm2_process() {
    # First check if PM2 daemon is running
    if ! pgrep -f "PM2" > /dev/null 2>&1; then
        log_message "WARNING" "PM2 daemon not running, attempting to start"
        return 1
    fi
    
    # Check if our specific app is running
    local pm2_status
    pm2_status=$(pm2 describe "$PM2_APP_NAME" 2>/dev/null | grep -c "status.*online" || echo "0")
    
    if [ "$pm2_status" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to check health endpoint
check_health_endpoint() {
    if command -v check_health_with_timing &> /dev/null; then
        # Use CloudWatch-enabled health check
        check_health_with_timing "$HEALTH_ENDPOINT"
        return $?
    else
        # Fallback to simple check
        local response
        local http_code
        
        response=$(curl -s -w "%{http_code}" -o /dev/null -m 30 "$HEALTH_ENDPOINT" 2>/dev/null)
        http_code=$?
        
        if [ $http_code -eq 0 ] && [ "$response" = "200" ]; then
            return 0
        else
            return 1
        fi
    fi
}

# Function to check if process is consuming CPU (stuck in loop)
check_cpu_usage() {
    local pid
    # Get PID more reliably
    pid=$(pm2 list | grep "$PM2_APP_NAME" | awk '{print $10}' | head -1)
    
    if [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "-" ]; then
        local cpu_usage
        cpu_usage=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null | tr -d ' ' | cut -d'.' -f1)
        
        if [ -n "$cpu_usage" ] && [ "$cpu_usage" -gt 90 ]; then
            return 1
        fi
    fi
    return 0
}

# Function to restart PM2 service
restart_pm2_service() {
    log_message "WARNING" "Restarting PM2 service: $PM2_APP_NAME"
    
    # Send restart metric to CloudWatch
    if command -v send_service_restart &> /dev/null; then
        send_service_restart
    fi
    
    # Kill stuck process if needed
    local pid=$(pm2 jlist | jq -r ".[] | select(.name==\"$PM2_APP_NAME\") | .pid")
    if [ -n "$pid" ] && [ "$pid" != "null" ]; then
        log_message "INFO" "Killing stuck process PID: $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
    
    # Restart the PM2 service
    pm2 restart "$PM2_APP_NAME" --force
    local restart_result=$?
    
    if [ $restart_result -eq 0 ]; then
        log_message "INFO" "PM2 service restarted successfully"
        # Wait for service to fully start
        sleep 30
    else
        log_message "ERROR" "Failed to restart PM2 service"
        # Try to start if restart failed
        pm2 start "/home/ubuntu/automatenb/launchuvi.py" --name "$PM2_APP_NAME" --interpreter python3
        log_message "INFO" "Attempted to start PM2 service from scratch"
    fi
}

# Function to send alerts (customize as needed)
send_alert() {
    local message=$1
    log_message "ALERT" "$message"
    
    # Send to AWS CloudWatch if AWS CLI is configured
    if command -v aws &> /dev/null; then
        aws logs create-log-group --log-group-name "pm2-health-monitor" 2>/dev/null || true
        aws logs create-log-stream --log-group-name "pm2-health-monitor" --log-stream-name "$(hostname)" 2>/dev/null || true
        aws logs put-log-events --log-group-name "pm2-health-monitor" --log-stream-name "$(hostname)" --log-events timestamp=$(date +%s000),message="$message" 2>/dev/null || true
    fi
    
    # Send email if mail is configured
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "PM2 Health Alert - $(hostname)" root 2>/dev/null || true
    fi
}

# Main monitoring function
monitor_health() {
    local failed_checks=0
    local last_restart=$(date +%s)
    
    log_message "INFO" "Starting PM2 health monitoring for $PM2_APP_NAME"
    
    while true; do
        local current_time=$(date +%s)
        local health_ok=true
        
        # Check if PM2 process exists
        if ! check_pm2_process; then
            log_message "ERROR" "PM2 process $PM2_APP_NAME not found"
            health_ok=false
        # Check health endpoint
        elif ! check_health_endpoint; then
            log_message "WARNING" "Health endpoint check failed for $PM2_APP_NAME"
            health_ok=false
        # Check CPU usage for stuck processes
        elif ! check_cpu_usage; then
            log_message "WARNING" "Process appears to be stuck (high CPU usage)"
            health_ok=false
        fi
        
        if [ "$health_ok" = false ]; then
            failed_checks=$((failed_checks + 1))
            log_message "WARNING" "Health check failed ($failed_checks/$MAX_RETRIES)"
            
            if [ $failed_checks -ge $MAX_RETRIES ]; then
                # Check cooldown period to prevent restart loops
                local time_since_restart=$((current_time - last_restart))
                if [ $time_since_restart -ge $RESTART_COOLDOWN ]; then
                    send_alert "PM2 service $PM2_APP_NAME failed $MAX_RETRIES health checks. Restarting service."
                    restart_pm2_service
                    failed_checks=0
                    last_restart=$(date +%s)
                else
                    log_message "INFO" "Restart cooldown active. Waiting $((RESTART_COOLDOWN - time_since_restart)) seconds"
                fi
            fi
            
            sleep $RETRY_INTERVAL
        else
            if [ $failed_checks -gt 0 ]; then
                log_message "INFO" "Health check recovered for $PM2_APP_NAME"
            fi
            failed_checks=0
            sleep 30  # Normal check interval
        fi
    done
}

# Handle script termination
cleanup() {
    log_message "INFO" "Health monitoring stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check dependencies
if ! command -v pm2 &> /dev/null; then
    log_message "ERROR" "PM2 not found. Please install PM2."
    exit 1
fi

if ! command -v curl &> /dev/null; then
    log_message "ERROR" "curl not found. Please install curl."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_message "WARNING" "jq not found. Some features may not work. Installing jq..."
    sudo apt-get update && sudo apt-get install -y jq
fi

# Start monitoring
monitor_health
