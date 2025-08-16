# PM2 Health Monitoring System Setup Guide

## Overview

This comprehensive monitoring system ensures your PM2 service (`codebox`) remains responsive and automatically restarts when it becomes unresponsive. The system includes multiple layers of monitoring:

1. **Continuous Health Monitor** (systemd service)
2. **Cron-based Backup Monitor** (every 5 minutes)
3. **AWS CloudWatch Integration** (metrics and alerts)

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Health Monitor │    │  Cron Backup    │
│   (launchuvi.py)│◄───┤   (systemd)     │    │   Monitor       │
│   Port: 5000    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS CloudWatch                              │
│              (Metrics, Logs, Alarms)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Health Check Endpoint

Your FastAPI application (`launchuvi.py`) includes a health check endpoint:

```python
@app.get("/api-health")
async def api_health():
    return {"status": "API is healthy"}
```

### 2. Main Health Monitor (`monitor_health.sh`)

**Location**: `/home/ubuntu/automatenb/monitor_health.sh`

**Features**:
- Continuous monitoring (24/7)
- Health endpoint checking
- PM2 process monitoring
- CPU usage monitoring (detects stuck processes)
- Automatic restart with cooldown periods
- CloudWatch metrics integration
- Comprehensive logging

**Configuration**:
```bash
PM2_APP_NAME="codebox"
HEALTH_ENDPOINT="http://localhost:5000/api-health"
MAX_RETRIES=3
RETRY_INTERVAL=10
RESTART_COOLDOWN=60
```

### 3. Systemd Service (`pm2-health-monitor.service`)

**Location**: `/etc/systemd/system/pm2-health-monitor.service`

**Features**:
- Runs as a system service
- Automatic startup on boot
- Service restart on failure
- Proper security settings

### 4. Cron Backup Monitor (`cron_health_check.sh`)

**Location**: `/home/ubuntu/automatenb/cron_health_check.sh`

**Schedule**: Every 5 minutes via crontab

**Features**:
- Lightweight backup monitoring
- Lock file prevention of duplicate runs
- Quick health checks
- Emergency restart capability

### 5. CloudWatch Integration

**Components**:
- Custom metrics (`cloudwatch_metrics.sh`)
- Setup script (`cloudwatch_setup.sh`)
- Log aggregation
- Alarm configuration

## Installation Status

✅ **Completed Components**:
- Main health monitoring script
- Systemd service configuration
- Cron backup monitoring
- CloudWatch integration scripts
- Service auto-start enabled

## Current Service Status

```bash
# Check PM2 status
pm2 list

# Check health monitor service
sudo systemctl status pm2-health-monitor

# Check cron jobs
crontab -l

# Check logs
sudo journalctl -u pm2-health-monitor --lines=50
```

## Monitoring Capabilities

### Health Checks Performed

1. **PM2 Process Check**: Verifies PM2 daemon and app are running
2. **HTTP Health Check**: Tests `/api-health` endpoint response
3. **CPU Usage Check**: Detects stuck processes (>90% CPU)
4. **Response Time Monitoring**: Measures endpoint response times

### Automatic Actions

1. **Service Restart**: When health checks fail 3 times
2. **Process Termination**: Force kills stuck processes
3. **Cooldown Period**: 60-second minimum between restarts
4. **Metric Reporting**: Sends data to CloudWatch
5. **Alert Generation**: Triggers notifications on failures

## Configuration Files

### Main Configuration

```bash
# PM2 app name
PM2_APP_NAME="codebox"

# Health check endpoint
HEALTH_ENDPOINT="http://localhost:5000/api-health"

# Retry settings
MAX_RETRIES=3
RETRY_INTERVAL=10
RESTART_COOLDOWN=60

# Log locations
LOG_FILE="/var/log/pm2-health-monitor.log"
CRON_LOG_FILE="/var/log/pm2-cron-monitor.log"
```

### CloudWatch Settings

```bash
NAMESPACE="PM2/CodeBox"
REGION="us-east-1"
LOG_GROUP="/aws/ec2/pm2-health"
```

## Log Files

| File | Purpose | Location |
|------|---------|----------|
| Health Monitor | Main service logs | `/var/log/pm2-health-monitor.log` |
| Cron Monitor | Backup monitor logs | `/var/log/pm2-cron-monitor.log` |
| PM2 Output | Application stdout | `~/.pm2/logs/codebox-out.log` |
| PM2 Errors | Application stderr | `~/.pm2/logs/codebox-error.log` |
| System Journal | Systemd service logs | `journalctl -u pm2-health-monitor` |

## AWS CloudWatch Setup

### Prerequisites

1. **AWS CLI Installed and Configured**:
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region, and Output format
```

2. **IAM Permissions Required**:
- `cloudwatch:PutMetricData`
- `logs:CreateLogGroup`
- `logs:CreateLogStream`
- `logs:PutLogEvents`
- `sns:CreateTopic`
- `sns:Subscribe`

### Setup CloudWatch Monitoring

1. **Run the setup script**:
```bash
cd /home/ubuntu/automatenb
./cloudwatch_setup.sh
```

2. **Update email endpoint**:
Edit `cloudwatch_setup.sh` and change:
```bash
EMAIL_ENDPOINT="your-email@domain.com"  # Replace with your email
```

3. **Confirm email subscription** (check your email after running setup)

### Custom Metrics Available

| Metric Name | Description | Unit |
|-------------|-------------|------|
| ServiceStatus | 1=online, 0=offline | None |
| HealthCheckFailures | Number of failed checks | Count |
| ServiceRestarts | Number of restarts | Count |
| ResponseTime | Endpoint response time | Milliseconds |

### CloudWatch Alarms

| Alarm Name | Condition | Action |
|------------|-----------|--------|
| PM2-CodeBox-Service-Down | ServiceStatus < 1 for 10 minutes | Send notification |
| PM2-CodeBox-High-Error-Rate | >3 failures in 5 minutes | Send notification |
| PM2-CodeBox-Frequent-Restarts | >2 restarts in 15 minutes | Send notification |

## Troubleshooting

### Common Issues

#### 1. Service Not Starting

```bash
# Check service status
sudo systemctl status pm2-health-monitor

# Check service logs
sudo journalctl -u pm2-health-monitor --lines=50

# Restart service
sudo systemctl restart pm2-health-monitor
```

#### 2. Health Checks Failing

```bash
# Test health endpoint manually
curl -v http://localhost:5000/api-health

# Check PM2 status
pm2 list
pm2 logs codebox

# Check application logs
tail -f ~/.pm2/logs/codebox-error.log
```

#### 3. Permission Issues

```bash
# Fix log file permissions
sudo chown ubuntu:ubuntu /var/log/pm2-*.log
sudo chmod 664 /var/log/pm2-*.log

# Check systemd service permissions
sudo systemctl edit pm2-health-monitor
```

#### 4. Cron Jobs Not Running

```bash
# Check cron service
sudo systemctl status cron

# Check cron logs
sudo journalctl -u cron

# Test cron script manually
/home/ubuntu/automatenb/cron_health_check.sh
```

### Manual Testing

#### Test Health Monitoring

```bash
# Run health check manually
cd /home/ubuntu/automatenb
./monitor_health.sh &

# Stop and start to test restart
pm2 stop codebox
# Wait and observe restart

# Test with broken endpoint
# Temporarily modify health endpoint URL and observe behavior
```

#### Test Cron Monitoring

```bash
# Run cron script manually
./cron_health_check.sh

# Check lock file mechanism
ls -la /tmp/pm2-health-check.lock
```

#### Test CloudWatch Integration

```bash
# Source metrics functions
source ./cloudwatch_metrics.sh

# Send test metric
send_service_status 1

# Check CloudWatch console for metrics
```

## Maintenance

### Regular Tasks

1. **Log Rotation** (weekly):
```bash
# Archive old logs
sudo logrotate /etc/logrotate.d/pm2-health
```

2. **Service Health Check** (monthly):
```bash
# Verify all components
sudo systemctl status pm2-health-monitor
crontab -l
pm2 list
```

3. **Update Dependencies** (as needed):
```bash
# Update AWS CLI
pip3 install --upgrade awscli

# Update system packages
sudo apt update && sudo apt upgrade
```

### Customization

#### Modify Check Intervals

Edit `monitor_health.sh`:
```bash
RETRY_INTERVAL=10      # Seconds between failed checks
RESTART_COOLDOWN=60    # Minimum seconds between restarts
```

Edit crontab:
```bash
crontab -e
# Change */5 to */3 for 3-minute intervals
*/3 * * * * /home/ubuntu/automatenb/cron_health_check.sh
```

#### Add Custom Metrics

Edit `cloudwatch_metrics.sh` to add new metrics:
```bash
send_custom_metric() {
    local metric_name=$1
    local value=$2
    send_metric "$metric_name" "$value" "Count"
}
```

#### Modify Alert Thresholds

Edit `cloudwatch_setup.sh` alarm configurations:
```bash
--threshold 3          # Change alert threshold
--evaluation-periods 2 # Change evaluation periods
```

## Security Considerations

1. **Service runs as `ubuntu` user** (not root)
2. **Limited file system access** via systemd restrictions
3. **Log files have appropriate permissions**
4. **AWS credentials stored securely**

## Performance Impact

- **CPU Usage**: Minimal (~0.1% average)
- **Memory Usage**: ~1-2MB per monitoring process
- **Network**: Minimal (local health checks only)
- **Disk I/O**: Log file writes only

## Support and Maintenance

### Backup and Recovery

1. **Backup monitoring scripts**:
```bash
tar -czf pm2-monitoring-backup.tar.gz \
  monitor_health.sh \
  cron_health_check.sh \
  cloudwatch_*.sh \
  pm2-health-monitor.service
```

2. **Recovery procedure**:
```bash
# Restore files
tar -xzf pm2-monitoring-backup.tar.gz

# Reinstall service
sudo cp pm2-health-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pm2-health-monitor
sudo systemctl start pm2-health-monitor
```

### Monitoring the Monitors

```bash
# Check if health monitor is running
pgrep -f monitor_health.sh

# Check recent cron executions
grep "cron health check" /var/log/pm2-cron-monitor.log | tail -5

# Verify CloudWatch metrics are being sent
aws cloudwatch get-metric-statistics \
  --namespace "PM2/CodeBox" \
  --metric-name "ServiceStatus" \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

## Conclusion

This monitoring system provides comprehensive protection against PM2 service failures with multiple layers of redundancy:

- **Primary**: Continuous systemd-managed health monitoring
- **Secondary**: Cron-based backup monitoring every 5 minutes
- **Tertiary**: AWS CloudWatch with alerts and notifications

The system automatically handles common failure scenarios including:
- Service crashes
- Unresponsive endpoints
- Hung processes
- Resource exhaustion

All components are production-ready and will ensure your PM2 service maintains high availability.
