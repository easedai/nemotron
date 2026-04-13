#!/bin/bash
# shellcheck disable=SC2129
set -euo pipefail

# ------------------------------------------------------------------
# ECS agent configuration
# ------------------------------------------------------------------
cat >> /etc/ecs/ecs.config <<ECS_CONFIG
ECS_CLUSTER=${cluster_name}

# Use locally cached image layers instead of pulling on every task launch.
# On first start the image is pulled and cached on the EBS root volume.
# Subsequent task starts (restarts, redeployments) reuse the cache —
# critical for the large vllm/vllm-openai image (~15 GB).
ECS_IMAGE_PULL_BEHAVIOR=prefer-cached

# Allow containers to assume IAM roles via task role credentials
ECS_ENABLE_TASK_IAM_ROLE=true

# Gracefully drain tasks when the EC2 Spot interruption notice arrives
ECS_ENABLE_SPOT_INSTANCE_DRAINING=true

# Give the vLLM container 120 s to finish in-flight requests on SIGTERM
ECS_CONTAINER_STOP_TIMEOUT=120s

# Reserve 512 MB for the OS + ECS agent (not offered to tasks)
ECS_RESERVED_MEMORY=512
ECS_CONFIG

# ------------------------------------------------------------------
# Mount the S3 Files filesystem via NFS 4.1
#
# The mount target IP is baked into this script at launch-template
# creation time by Terraform (aws_s3files_mount_target.models["0"].ip_address).
# Both AZ mount targets are reachable within the VPC; using a single IP
# here is acceptable — AWS routes NFS traffic intra-VPC without NAT.
#
# The mount exposes the S3 bucket root at /data/huggingface.
# Model files land at /data/huggingface/nemotron-nano-12b-vl-bf16/
# which is bind-mounted into the vLLM container as
# /root/.cache/huggingface/nemotron-nano-12b-vl-bf16.
# ------------------------------------------------------------------
S3FILES_IP="${s3files_mount_ip}"
mkdir -p /data/huggingface

# Retry loop: mount target may still be provisioning when the instance boots
MOUNTED=false
for i in $(seq 1 30); do
  if mount -t nfs4 \
      -o "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,_netdev" \
      "$${S3FILES_IP}":/ /data/huggingface 2>/dev/null; then
    echo "S3 Files filesystem mounted successfully on attempt $i"
    MOUNTED=true
    break
  fi
  echo "Mount attempt $i failed — retrying in 10 s..."
  sleep 10
done

if [ "$MOUNTED" = "true" ]; then
  # Persist mount across instance reboots
  echo "$${S3FILES_IP}:/ /data/huggingface nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,_netdev 0 0" >> /etc/fstab
  chmod 1777 /data/huggingface
else
  echo "ERROR: Failed to mount S3 Files filesystem after 30 attempts" >&2
  exit 1
fi
