if curl -s -o /dev/null -w "%{http_code}" http://localhost:10102/ | grep -q "405"; then
  echo "AWS Orchestrator Agent is healthy";
  exit 0;
fi;

# Unhealthy
exit 1;
