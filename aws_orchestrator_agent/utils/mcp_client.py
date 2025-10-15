# Copyright (C) 2025 StructBinary
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from aws_orchestrator_agent.utils.exceptions import ConfigError, AWSOrchestratorAgentError


class MCPAdapterClient:
    """
    MCP client using langchain-mcp-adapter for LangChain/LangGraph compatibility.
    
    This client wraps the langchain-mcp-adapter's MultiServerMCPClient to provide
    a higher-level interface for agent discovery and health checking operations.
    """
    
    def __init__(self, host: str = 'localhost', port: str = '10100', transport: str = 'sse'):
        """
        Initialize the MCP adapter client.
        
        Args:
            host: The hostname of the MCP server (for SSE transport)
            port: The port of the MCP server (for SSE transport) 
            transport: The transport type ('sse' or 'stdio')
        """
        self.host = host
        self.port = port
        self.transport = transport
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        self._tool_map: Dict[str, BaseTool] = {}
        
        # Build MCP server configuration
        self.mcp_config = self._build_mcp_config()
    
    def _build_mcp_config(self) -> Dict[str, Any]:
        """Build MCP server configuration based on transport type."""
        if self.transport == 'sse':
            return {
                "agent_server": {
                    "url": f"http://{self.host}:{self.port}/sse",
                    "transport": "sse"
                }
            }
        elif self.transport == 'stdio':
            # Filter out None values from environment variables
            env = {
                key: value for key, value in {
                    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                }.items() if value is not None
            }
            
            return {
                "agent_server": {
                    "command": "uv",
                    "args": ["run", "-m", "agents_mcp_server"],
                    "transport": "stdio",
                    "env": env if env else None
                }
            }
        else:
            raise ValueError(
                f"Unsupported transport type: {self.transport}. Must be 'sse' or 'stdio'."
            )
    
    async def initialize(self) -> None:
        """Initialize the MCP client and load tools."""
        self.client = MultiServerMCPClient(self.mcp_config)
        
        # Use get_tools() method as recommended for version 0.1.0
        self.tools = await self.client.get_tools()
        
        # Create a mapping of tool names to tools for easy lookup
        self._tool_map = {tool.name: tool for tool in self.tools}
    
    async def close(self) -> None:
        """Close the MCP client connection."""
        # MultiServerMCPClient in v0.1.0 doesn't require explicit cleanup
        pass
    
    @asynccontextmanager
    async def session(self):
        """Context manager for MCP client session."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()
    
    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name from the loaded tools."""
        return self._tool_map.get(tool_name)
    
    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with the given arguments."""
        tool = self._get_tool(tool_name)
        if not tool:
            raise ConfigError(
                "No tools available. Ensure MCP server is running and tools are registered."
            )
        
        try:
            # Execute the tool using LangChain's tool interface
            result = await tool.ainvoke(kwargs)
            
            # If result is a string that looks like JSON, parse it
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result
            
            return result
        except Exception as e:
            raise AWSOrchestratorAgentError(f"Error executing tool '{tool_name}': {str(e)}")
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of all available agents from the MCP server.
        
        Returns:
            List of agent dictionaries with agent information
        """
        try:
            # Use the actual available tool for listing agents
            if 'find_a2a_agents' in self._tool_map:
                result = await self._execute_tool('find_a2a_agents', query="*", filters={})
                if isinstance(result, dict) and 'agents' in result:
                    return result['agents']
                return result if isinstance(result, list) else []
            
            # Fallback: try list_mcp_servers if available
            if 'list_mcp_servers' in self._tool_map:
                result = await self._execute_tool('list_mcp_servers')
                if isinstance(result, dict) and 'servers' in result:
                    return result['servers']
                return result if isinstance(result, list) else []
            
            # If no specific tools, return empty list
            return []
            
        except Exception as e:
            print(f"Warning: Failed to list agents: {e}")
            return []
    
    async def find_agent(self, query: str) -> Any:
        """
        Find agents matching the given query.
        
        Args:
            query: Natural language query for agent search
            
        Returns:
            Tool execution result containing matching agents
        """
        return await self._execute_tool('find_a2a_agents', query=query, filters={})
    
    async def check_agent_health(self, agent_id: str) -> Optional[str]:
        """
        Check the health status of a specific agent.
        
        Args:
            agent_id: The ID of the agent to check
            
        Returns:
            Health status string or None if check failed
        """
        try:
            # Use get_agent_details to check agent status/health
            result = await self._execute_tool('get_agent_details', agent_id=agent_id)
            
            if isinstance(result, dict):
                # Look for health/status fields in the agent details
                return result.get('status') or result.get('health') or 'unknown'
            elif isinstance(result, str):
                return result
            
            return None
            
        except Exception as e:
            print(f"Warning: Health check failed for agent {agent_id}: {e}")
            return None
    
    async def get_agents_with_health(self) -> List[Dict[str, Any]]:
        """
        Get all agents with their current health status.
        
        Returns:
            List of agents with health information added
        """
        agents = await self.list_agents()
        
        for agent in agents:
            agent_id = agent.get("id", "")
            if agent_id:
                health = await self.check_agent_health(agent_id)
                agent["health"] = health
            else:
                agent["health"] = None
        
        return agents
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tool_map.keys())
    
    def get_tools(self) -> List[BaseTool]:
        """Get the loaded LangChain tools for use in agents."""
        return self.tools


# Convenience function to create a client session
@asynccontextmanager
async def create_mcp_client(host: str = 'localhost', port: str = '10100', transport: str = 'sse'):
    """
    Create an MCP client session.
    
    Args:
        host: MCP server hostname
        port: MCP server port
        transport: Transport type ('sse' or 'stdio')
        
    Yields:
        Initialized MCPAdapterClient
    """
    client = MCPAdapterClient(host=host, port=port, transport=transport)
    async with client.session():
        yield client
