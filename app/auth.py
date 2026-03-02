"""
auth.py – API key authentication dependency.

Set the ROUTER_API_KEY environment variable to enable auth.
If the variable is unset or empty the API is open (development mode).
"""
import os

from fastapi import HTTPException, Request, status


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency: validates the X-API-Key header when a key is configured."""
    expected: str = os.environ.get("ROUTER_API_KEY", "").strip()
    if not expected:
        # No key configured → open access (dev / internal deployment)
        return
    key: str = request.headers.get("X-API-Key", "").strip()
    if key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it via the X-API-Key header.",
        )
