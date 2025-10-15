#!/bin/sh
# Copyright AWS Orchestrator Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Check if the server is responding on the expected port
# The application returns 405 for GET requests, which means it's running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:10102/ | grep -q "405"; then
  echo "AWS Orchestrator Agent is healthy";
  exit 0;
fi;

# Unhealthy
exit 1;
