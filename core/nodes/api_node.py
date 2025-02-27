"""Node implementation for accessing external APIs."""
from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
import aiohttp
import json
from dataclasses import dataclass, field
from core.base import Node
from .factory import register_node
from .protocol import APINodeProtocol

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for API nodes."""
    base_url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: Optional[int] = None
    rate_period: float = 60.0
    auth_type: Optional[str] = None
    auth_config: Dict[str, Any] = field(default_factory=dict)

@register_node("api")
class APINode(Node):
    """Node for making API requests with error handling and rate limiting."""
    
    def __init__(self, name: str, config: APIConfig):
        super().__init__(name=name)
        self.config = config
        self.session = None
        self._request_times: List[float] = []
        self._semaphore = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize API client and rate limiting."""
        if not self._initialized:
            # Create persistent session
            self.session = aiohttp.ClientSession(
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Set up rate limiting if configured
            if self.config.rate_limit is not None:
                self._semaphore = asyncio.Semaphore(self.config.rate_limit)
            
            # Set up authentication if configured
            if self.config.auth_type:
                await self._setup_auth()
            
            self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._initialized = False
    
    async def _setup_auth(self):
        """Set up authentication based on config."""
        if self.config.auth_type == "basic":
            username = self.config.auth_config.get("username")
            password = self.config.auth_config.get("password")
            if username and password:
                auth = aiohttp.BasicAuth(username, password)
                self.session = aiohttp.ClientSession(
                    headers=self.config.headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    auth=auth
                )
        elif self.config.auth_type == "bearer":
            token = self.config.auth_config.get("token")
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif self.config.auth_type == "api_key":
            key = self.config.auth_config.get("key")
            key_name = self.config.auth_config.get("key_name", "X-API-Key")
            key_in = self.config.auth_config.get("key_in", "header")
            if key:
                if key_in == "header":
                    self.session.headers.update({key_name: key})
                elif key_in == "query":
                    # Will be added to request params
                    pass
    
    async def _respect_rate_limit(self):
        """Enforce rate limiting if configured."""
        now = asyncio.get_event_loop().time()
        
        # If rate limited, use semaphore
        if self._semaphore:
            async with self._semaphore:
                pass
        
        # Clean up old request times
        period_start = now - self.config.rate_period
        self._request_times = [t for t in self._request_times if t >= period_start]
        
        # If at rate limit, delay until we can make another request
        if self.config.rate_limit and len(self._request_times) >= self.config.rate_limit:
            wait_time = self._request_times[0] + self.config.rate_period - now
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self._request_times.append(asyncio.get_event_loop().time())
    
    async def request(self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            data: Form data
            headers: Additional headers for this request
            json_data: JSON data (alternative to data)
            **kwargs: Additional arguments passed to request
            
        Returns:
            Response data
            
        Raises:
            aiohttp.ClientError: If request fails after retries
        """
        if not self._initialized:
            await self.initialize()
        
        # Build URL
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        # Merge request-specific headers with default headers
        merged_headers = {**self.session.headers, **(headers or {})}
        
        # Add API key to query params if configured that way
        if (self.config.auth_type == "api_key" and 
            self.config.auth_config.get("key_in") == "query"):
            key = self.config.auth_config.get("key")
            key_name = self.config.auth_config.get("key_name", "api_key")
            if key:
                if not params:
                    params = {}
                params[key_name] = key
        
        # Respect rate limits
        await self._respect_rate_limit()
        
        # Retry logic
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=merged_headers,
                    json=json_data,
                    **kwargs
                ) as response:
                    # Check if successful
                    response.raise_for_status()
                    
                    # Parse response based on content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        result = await response.json()
                    else:
                        text = await response.text()
                        result = {"text": text}
                    
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "data": result
                    }
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.config.max_retries:
                    retry_delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed ({str(e)}), retrying in {retry_delay:.2f}s (attempt {attempt+1}/{self.config.max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Request failed after {self.config.max_retries} retries: {str(e)}")
                    raise
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs by making an API request.
        
        Args:
            inputs: Dictionary containing at least 'method' and 'endpoint'
            
        Returns:
            Dictionary containing API response
            
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        method = inputs.get("method", "GET").upper()
        endpoint = inputs.get("endpoint")
        
        if not endpoint:
            raise ValueError("Input must contain 'endpoint'")
        
        # Extract additional parameters
        params = inputs.get("params")
        data = inputs.get("data")
        headers = inputs.get("headers")
        json_data = inputs.get("json")
        
        # Make the request
        response = await self.request(
            method=method,
            endpoint=endpoint,
            params=params,
            data=data,
            headers=headers,
            json_data=json_data
        )
        
        return {"response": response}

def create_api_node(
    name: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
    auth_type: Optional[str] = None,
    auth_config: Optional[Dict[str, Any]] = None,
    rate_limit: Optional[int] = None,
    **kwargs
) -> APINode:
    """
    Convenience function to create an API node with common configurations.
    
    Args:
        name: Node name
        base_url: Base URL for API
        headers: Default headers
        auth_type: Authentication type (None, 'basic', 'bearer', 'api_key')
        auth_config: Authentication configuration
        rate_limit: Rate limit (requests per minute)
        **kwargs: Additional API configuration parameters
        
    Returns:
        Configured APINode instance
    """
    config = APIConfig(
        base_url=base_url,
        headers=headers or {},
        auth_type=auth_type,
        auth_config=auth_config or {},
        rate_limit=rate_limit,
        **kwargs
    )
    return APINode(name=name, config=config)
