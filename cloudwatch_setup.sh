#!/bin/bash

# AWS CloudWatch Setup Script for PM2 Monitoring
# This script sets up CloudWatch custom metrics and alarms

# Configuration
REGION="us-east-1"  # Change to your AWS region
NAMESPACE="PM2/CodeBox"
LOG_GROUP="/aws/ec2/pm2-health"
ALARM_SNS_TOPIC="pm2-health-alerts"
EMAIL_ENDPOINT="your-email@domain.com"  # Change this to your email

# Function to check if AWS CLI is configured
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo "ERROR: AWS CLI not found. Please install AWS CLI."
        return 1
    fi
    
    if ! aws sts get-caller-identity &>/dev/null; then
        echo "ERROR: AWS CLI not configured. Please run 'aws configure'."
        return 1
    fi
    
    echo "AWS CLI is configured"
    return 0
}

# Function to create CloudWatch log group
create_log_group() {
    echo "Creating CloudWatch log group: $LOG_GROUP"
    
    aws logs create-log-group \
        --log-group-name "$LOG_GROUP" \
        --region "$REGION" 2>/dev/null || echo "Log group may already exist"
    
    # Set retention policy (14 days)
    aws logs put-retention-policy \
        --log-group-name "$LOG_GROUP" \
        --retention-in-days 14 \
        --region "$REGION" 2>/dev/null || true
}

# Function to create SNS topic for alerts
create_sns_topic() {
    echo "Creating SNS topic: $ALARM_SNS_TOPIC"
    
    local topic_arn
    topic_arn=$(aws sns create-topic \
        --name "$ALARM_SNS_TOPIC" \
        --region "$REGION" \
        --output text --query 'TopicArn' 2>/dev/null)
    
    if [ -n "$topic_arn" ]; then
        echo "SNS Topic ARN: $topic_arn"
        
        # Subscribe email to topic (you'll need to confirm subscription)
        if [ "$EMAIL_ENDPOINT" != "your-email@domain.com" ]; then
            aws sns subscribe \
                --topic-arn "$topic_arn" \
                --protocol email \
                --notification-endpoint "$EMAIL_ENDPOINT" \
                --region "$REGION" 2>/dev/null || true
            echo "Email subscription created. Check your email to confirm."
        fi
        
        echo "$topic_arn"
    else
        echo "Failed to create SNS topic"
        return 1
    fi
}

# Function to create CloudWatch alarms
create_cloudwatch_alarms() {
    local sns_topic_arn=$1
    
    echo "Creating CloudWatch alarms..."
    
    # Alarm for service downtime
    aws cloudwatch put-metric-alarm \
        --alarm-name "PM2-CodeBox-Service-Down" \
        --alarm-description "PM2 CodeBox service is down" \
        --metric-name "ServiceStatus" \
        --namespace "$NAMESPACE" \
        --statistic Average \
        --period 300 \
        --threshold 1 \
        --comparison-operator LessThanThreshold \
        --evaluation-periods 2 \
        --alarm-actions "$sns_topic_arn" \
        --ok-actions "$sns_topic_arn" \
        --region "$REGION" 2>/dev/null || echo "Failed to create service down alarm"
    
    # Alarm for high error rate
    aws cloudwatch put-metric-alarm \
        --alarm-name "PM2-CodeBox-High-Error-Rate" \
        --alarm-description "PM2 CodeBox service has high error rate" \
        --metric-name "HealthCheckFailures" \
        --namespace "$NAMESPACE" \
        --statistic Sum \
        --period 300 \
        --threshold 3 \
        --comparison-operator GreaterThanThreshold \
        --evaluation-periods 1 \
        --alarm-actions "$sns_topic_arn" \
        --region "$REGION" 2>/dev/null || echo "Failed to create high error rate alarm"
    
    # Alarm for service restarts
    aws cloudwatch put-metric-alarm \
        --alarm-name "PM2-CodeBox-Frequent-Restarts" \
        --alarm-description "PM2 CodeBox service is restarting frequently" \
        --metric-name "ServiceRestarts" \
        --namespace "$NAMESPACE" \
        --statistic Sum \
        --period 900 \
        --threshold 2 \
        --comparison-operator GreaterThanThreshold \
        --evaluation-periods 1 \
        --alarm-actions "$sns_topic_arn" \
        --region "$REGION" 2>/dev/null || echo "Failed to create frequent restarts alarm"
    
    echo "CloudWatch alarms created"
}

# Function to install CloudWatch agent
install_cloudwatch_agent() {
    echo "Installing CloudWatch agent..."
    
    # Download and install CloudWatch agent
    wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    sudo dpkg -i -E amazon-cloudwatch-agent.deb
    rm -f amazon-cloudwatch-agent.deb
    
    # Create CloudWatch agent configuration
    cat > /tmp/cloudwatch-config.json << 'EOF'
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/pm2-health-monitor.log",
                        "log_group_name": "/aws/ec2/pm2-health",
                        "log_stream_name": "{instance_id}-health-monitor"
                    },
                    {
                        "file_path": "/var/log/pm2-cron-monitor.log",
                        "log_group_name": "/aws/ec2/pm2-health",
                        "log_stream_name": "{instance_id}-cron-monitor"
                    },
                    {
                        "file_path": "/home/ubuntu/.pm2/logs/codebox-error.log",
                        "log_group_name": "/aws/ec2/pm2-health",
                        "log_stream_name": "{instance_id}-pm2-error"
                    },
                    {
                        "file_path": "/home/ubuntu/.pm2/logs/codebox-out.log",
                        "log_group_name": "/aws/ec2/pm2-health",
                        "log_stream_name": "{instance_id}-pm2-out"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "PM2/CodeBox",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF
    
    # Start CloudWatch agent with configuration
    sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
        -a fetch-config -m ec2 -s -c file:/tmp/cloudwatch-config.json
    
    # Enable CloudWatch agent to start on boot
    sudo systemctl enable amazon-cloudwatch-agent
    
    echo "CloudWatch agent installed and configured"
}

# Main setup function
main() {
    echo "Setting up AWS CloudWatch monitoring for PM2 CodeBox service..."
    
    # Check prerequisites
    if ! check_aws_cli; then
        echo "Please install and configure AWS CLI first"
        exit 1
    fi
    
    # Create log group
    create_log_group
    
    # Create SNS topic and get ARN
    local sns_arn
    sns_arn=$(create_sns_topic)
    
    if [ -n "$sns_arn" ]; then
        # Create CloudWatch alarms
        create_cloudwatch_alarms "$sns_arn"
    else
        echo "WARNING: Could not create SNS topic. Alarms will not send notifications."
    fi
    
    # Install CloudWatch agent
    if command -v amazon-cloudwatch-agent &> /dev/null; then
        echo "CloudWatch agent already installed"
    else
        install_cloudwatch_agent
    fi
    
    echo ""
    echo "CloudWatch setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Confirm your email subscription if you provided an email address"
    echo "2. Update the monitoring scripts to send metrics to CloudWatch"
    echo "3. Monitor the CloudWatch console for metrics and alarms"
    echo ""
    echo "CloudWatch Log Group: $LOG_GROUP"
    echo "CloudWatch Namespace: $NAMESPACE"
    if [ -n "$sns_arn" ]; then
        echo "SNS Topic ARN: $sns_arn"
    fi
}

# Check if running as root for some operations
if [ "$EUID" -eq 0 ]; then
    echo "Note: Some operations require sudo privileges"
fi

# Run main function
main
