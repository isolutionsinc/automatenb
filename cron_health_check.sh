#!/bin/bash

# Cron-based PM2 Health Check Script
# This script performs a quick health check and restart if needed
# Should be run every 5 minutes via cron

# Configuration
PM2_APP_NAME="codebox"
HEALTH_ENDPOINT="http://localhost:5000/api-health"
LOG_FILE="/var/log/pm2-cron-monitor.log"
LOCK_FILE="/tmp/pm2-health-check.lock"

# Create log file if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || LOG_FILE="/tmp/pm2-cron-monitor.log"

# Logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Check if another instance is running
if [ -f "$LOCK_FILE" ]; then
    local pid_in_lock=$(cat "$LOCK_FILE" 2>/dev/null)
    if kill -0 "$pid_in_lock" 2>/dev/null; then
        log_message "INFO" "Another health check is already running"
        exit 0
    else
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"

# Cleanup function
cleanup() {
    rm -f "$LOCK_FILE"
    exit $1
}

trap 'cleanup 0' EXIT
trap 'cleanup 1' INT TERM

# Function to check health endpoint
check_health() {
    local response
    response=$(curl -s -w "%{http_code}" -o /dev/null -m 10 "$HEALTH_ENDPOINT" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check PM2 process
check_pm2() {
    local status
    status=$(pm2 list 2>/dev/null | grep "$PM2_APP_NAME" | grep -c "online" || echo "0")
    
    if [ "$status" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to restart service
restart_service() {
    log_message "WARNING" "Attempting to restart $PM2_APP_NAME"
    
    # Stop the service first
    pm2 stop "$PM2_APP_NAME" 2>/dev/null || true
    sleep 5
    
    # Start the service
    pm2 start "$PM2_APP_NAME" 2>/dev/null || \
    pm2 start "/home/ubuntu/automatenb/launchuvi.py" --name "$PM2_APP_NAME" --interpreter python3
    
    # Wait a moment for startup
    sleep 10
    
    # Verify restart was successful
    if check_pm2 && check_health; then
        log_message "INFO" "Service restart successful"
        return 0
    else
        log_message "ERROR" "Service restart failed"
        return 1
    fi
}

# Main health check
main() {
    log_message "INFO" "Starting cron health check"
    
    # Check PM2 process first
    if ! check_pm2; then
        log_message "ERROR" "PM2 process not running"
        restart_service
        cleanup 0
    fi
    
    # Check health endpoint
    if ! check_health; then
        log_message "WARNING" "Health endpoint failed"
        # Wait and try again
        sleep 5
        if ! check_health; then
            log_message "ERROR" "Health endpoint failed twice"
            restart_service
        else
            log_message "INFO" "Health endpoint recovered on retry"
        fi
    else
        log_message "INFO" "Health check passed"
    fi
    
    cleanup 0
}

# Check dependencies
if ! command -v pm2 &> /dev/null; then
    log_message "ERROR" "PM2 not found"
    cleanup 1
fi

if ! command -v curl &> /dev/null; then
    log_message "ERROR" "curl not found"
    cleanup 1
fi

# Run main function
main
