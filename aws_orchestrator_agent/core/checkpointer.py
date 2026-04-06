"""
Checkpointer configuration and management for HITL.

Supports both PostgreSQL (production) and MemorySaver (development) checkpointers.
"""

from typing import Optional, Literal
from langgraph.checkpoint.memory import MemorySaver
from aws_orchestrator_agent.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger

# Optional PostgreSQL checkpointer import
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore
    POSTGRES_AVAILABLE = False

# Create logger for AWS Checkpointer
hitl_logger = AgentLogger("AWSCheckPointer")

CheckpointerType = Literal["postgres", "memory"]


def get_database_uri(config: Optional[Config] = None) -> Optional[str]:
    """
    Get database URI from configuration.
    
    Args:
        config: Optional Config instance. If None, creates a new one.
        
    Returns:
        Database URI string or None if not configured.
    """
    if config is None:
        config = Config()
    
    # Check for database URI in config
    db_uri = getattr(config, "DATABASE_URI", None)
    
    # Also check environment variable
    import os
    db_uri = db_uri or os.getenv("DATABASE_URI") or os.getenv("POSTGRES_URI")
    
    return db_uri


def create_checkpointer(
    checkpointer_type: CheckpointerType = "memory",
    config: Optional[Config] = None,
    database_uri: Optional[str] = None,
    auto_setup: bool = True
):
    """
    Create a checkpointer instance for HITL workflows.
    
    Args:
        checkpointer_type: Type of checkpointer ("postgres" or "memory")
        config: Optional Config instance
        database_uri: Optional database URI (overrides config)
        auto_setup: Whether to automatically setup PostgreSQL tables
        
    Returns:
        Checkpointer instance (PostgresSaver or MemorySaver)
        
    Raises:
        ValueError: If checkpointer_type is "postgres" but no database URI provided
        Exception: If PostgreSQL setup fails
    """
    if checkpointer_type == "memory":
        hitl_logger.log_structured(
            level="INFO",
            message="Creating MemorySaver checkpointer (development mode)",
            extra={"checkpointer_type": "memory"}
        )
        return MemorySaver()
    
    elif checkpointer_type == "postgres":
        # Check if PostgreSQL checkpointer is available
        if not POSTGRES_AVAILABLE or PostgresSaver is None:
            hitl_logger.log_structured(
                level="WARNING",
                message="PostgreSQL checkpointer not available, falling back to MemorySaver",
                extra={
                    "checkpointer_type": "postgres",
                    "fallback": "memory",
                    "reason": "langgraph.checkpoint.postgres not installed"
                }
            )
            return MemorySaver()
        
        # Get database URI
        db_uri = database_uri or get_database_uri(config)
        
        if not db_uri:
            hitl_logger.log_structured(
                level="WARNING",
                message="No database URI found, falling back to MemorySaver",
                extra={
                    "checkpointer_type": "postgres",
                    "fallback": "memory"
                }
            )
            return MemorySaver()
        
        try:
            hitl_logger.log_structured(
                level="INFO",
                message="Creating PostgresSaver checkpointer",
                extra={
                    "checkpointer_type": "postgres",
                    "has_uri": bool(db_uri),
                    "auto_setup": auto_setup
                }
            )
            
            from psycopg_pool import ConnectionPool
            from psycopg import Connection
            from psycopg.rows import dict_row, DictRow
            from typing import cast
            
            # Create a long-lived connection pool 
            pool = ConnectionPool(
                conninfo=db_uri,
                max_size=20,
                kwargs={
                    "autocommit": True, 
                    "prepare_threshold": 0,
                    "row_factory": dict_row
                }
            )
            
            # Cast pool to the exact type PostgresSaver expects to satisfy pyright
            typed_pool = cast(ConnectionPool[Connection[DictRow]], pool)
            checkpointer = PostgresSaver(typed_pool)
            
            if auto_setup:
                checkpointer.setup()
                hitl_logger.log_structured(
                    level="INFO",
                    message="PostgreSQL checkpointer tables initialized",
                    extra={"checkpointer_type": "postgres"}
                )
            
            return checkpointer
            
        except Exception as e:
            hitl_logger.log_structured(
                level="ERROR",
                message=f"Failed to create PostgreSQL checkpointer: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Fallback to MemorySaver on error
            hitl_logger.log_structured(
                level="WARNING",
                message="Falling back to MemorySaver due to PostgreSQL error",
                extra={"checkpointer_type": "memory"}
            )
            return MemorySaver()
    
    else:
        raise ValueError(f"Unknown checkpointer type: {checkpointer_type}")


def get_checkpointer(
    config: Optional[Config] = None,
    prefer_postgres: bool = True
):
    """
    Get checkpointer based on configuration.
    
    Automatically selects PostgreSQL if available, otherwise falls back to MemorySaver.
    
    Args:
        config: Optional Config instance
        prefer_postgres: If True, prefer PostgreSQL over MemorySaver
        
    Returns:
        Checkpointer instance
    """
    if prefer_postgres:
        db_uri = get_database_uri(config)
        if db_uri:
            return create_checkpointer(
                checkpointer_type="postgres",
                config=config,
                database_uri=db_uri
            )
    
    # Fallback to memory
    return create_checkpointer(checkpointer_type="memory", config=config)
