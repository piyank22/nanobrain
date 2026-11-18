#!/bin/bash
# Stop this pod now (keeps volumes, releases the GPU). Uses the pod's own RUNPOD_API_KEY, which may stop its own
# pod over GraphQL (runpodctl get pod and REST v1 return unauthorized with that key).
#   bash stop.sh "reason"
W=${W:-/workspace}
echo "$(date -u +%FT%TZ) stop requested: $1" >> $W/progress.log
curl -s -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query":"mutation { podStop(input:{podId:\"'"$RUNPOD_POD_ID"'\"}) { id desiredStatus } }"}' \
  https://api.runpod.io/graphql >> $W/progress.log 2>&1
