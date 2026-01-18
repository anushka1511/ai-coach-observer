"""
FastAPI Backend Server for AI Coaching Observer - WITH FILE SUPPORT
"""

import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

from backend.core.orchestrator import CoachingObserverSystem
from backend.schemas.data_models import SessionReport

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Coaching Observer API",
    description="Real-time coaching session monitoring and analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys from environment
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not ASSEMBLYAI_KEY:
    logger.warning("ASSEMBLYAI_API_KEY not found in environment variables")
if not GEMINI_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Initialize the coaching system
coaching_system = CoachingObserverSystem(
    assemblyai_key=ASSEMBLYAI_KEY or "",
    gemini_key=GEMINI_KEY or ""
)


# Request/Response Models
class SessionStartRequest(BaseModel):
    session_type: str = "live"  # "live" or "file"
    device_index: Optional[int] = None
    file_path: Optional[str] = None  # For file mode


class SessionStartResponse(BaseModel):
    session_id: str
    status: str = "started"


class SessionStatusResponse(BaseModel):
    session_id: Optional[str]
    active: bool
    duration: float = 0.0
    chunks_processed: int = 0


# API Routes
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "running",
        "service": "AI Coaching Observer API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new coaching session"""
    try:
        if not ASSEMBLYAI_KEY or not GEMINI_KEY:
            raise HTTPException(
                status_code=500,
                detail="API keys not configured. Please set ASSEMBLYAI_API_KEY and GEMINI_API_KEY environment variables."
            )
        
        session_id = await coaching_system.start_session(
            session_type=request.session_type,
            device_index=request.device_index,
            file_path=request.file_path
        )
        
        logger.info(f"Session started: {session_id} (type: {request.session_type})")
        return SessionStartResponse(session_id=session_id, status="started")
        
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"Error starting session (RuntimeError): {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error starting session: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/session/stop")
async def stop_session():
    """Stop the current session and return report"""
    try:
        if not coaching_system.session_active:
            raise HTTPException(status_code=400, detail="No active session to stop")
        
        report = await coaching_system.stop_session()
        
        # Convert Pydantic model to dict for JSON serialization
        if hasattr(report, 'model_dump'):
            report_dict = report.model_dump()
        elif hasattr(report, 'dict'):
            report_dict = report.dict()
        else:
            from dataclasses import asdict
            report_dict = asdict(report)
        
        logger.info(f"Session stopped: {coaching_system.session_id}")
        return report_dict
        
    except Exception as e:
        logger.error(f"Error stopping session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/status", response_model=SessionStatusResponse)
async def get_session_status():
    """Get current session status"""
    try:
        duration = 0.0
        if coaching_system.session_data.get("start_time"):
            from datetime import datetime
            duration = (datetime.now() - coaching_system.session_data["start_time"]).total_seconds() / 60
        
        return SessionStatusResponse(
            session_id=coaching_system.session_id,
            active=coaching_system.session_active,
            duration=duration,
            chunks_processed=len(coaching_system.session_data.get("chunks", []))
        )
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/devices/audio")
async def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        devices = coaching_system.get_available_audio_devices()
        return {"devices": devices}
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/feedback")
async def websocket_feedback(websocket: WebSocket):
    """WebSocket endpoint for real-time feedback updates"""
    await websocket.accept()
    coaching_system.websocket_clients.add(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(coaching_system.websocket_clients)}")
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        coaching_system.websocket_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(coaching_system.websocket_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        coaching_system.websocket_clients.discard(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )