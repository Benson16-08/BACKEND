from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.logger import logger

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


# Mock auth for frontend compatibility
# TODO: Replace with real authentication system
@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate user and return JWT token",
)
async def login(request: LoginRequest) -> LoginResponse:
    # Mock authentication - replace with real auth
    if request.username == "dr.demo@mediassist.test" and request.password == "DemoPass123":
        logger.info(f"Login successful for user: {request.username}")
        return LoginResponse(
            token="mock-jwt-token-dr-demo-2026",
            user={
                "id": "user-demo-001",
                "name": "Dr. Demo",
                "role": "doctor",
                "email": request.username,
            },
        )

    logger.warning(f"Login failed for user: {request.username}")
    raise HTTPException(status_code=401, detail="Invalid credentials")