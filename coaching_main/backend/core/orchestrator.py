"""
Core Orchestrator - Fixed for GeminiAnalyzer
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Set
import json

from backend.models.audio_processor import AudioProcessor
from backend.models.inference_engine import ModelInferenceEngine
from backend.models.gemini_analyzer import GeminiAnalyzer
from backend.models.file_audio_processor import FileAudioProcessor
from backend.schemas.data_models import AudioChunk, RealTimeFeedback, GROWPhase, SessionReport

logger = logging.getLogger(__name__)


class CoachingObserverSystem:
    """Main orchestrator for the AI Coaching Observer system"""

    def __init__(self, assemblyai_key: str, gemini_key: str):
        self.assemblyai_key = assemblyai_key
        self.gemini_key = gemini_key
        
        # Core components
        self.audio_processor: Optional[AudioProcessor] = None
        self.file_processor: Optional[FileAudioProcessor] = None
        self.inference_engine = ModelInferenceEngine()
        self.gemini_analyzer = GeminiAnalyzer(gemini_key) if gemini_key else None
        
        # Session state
        self.session_id: Optional[str] = None
        self.session_active = False
        self.session_data: Dict[str, Any] = {}
        
        # Audio queue for chunks
        self.audio_queue: Optional[asyncio.Queue] = None
        
        # WebSocket clients for real-time updates
        self.websocket_clients: Set = set()
        
        # Background tasks
        self.processing_task: Optional[asyncio.Task] = None

    async def start_session(self, session_type: str = "live", device_index: Optional[int] = None, file_path: Optional[str] = None) -> str:
        """Start a new coaching session
        
        Args:
            session_type: "live" for microphone or "file" for audio file
            device_index: Audio device index (for live mode)
            file_path: Path to audio file (for file mode)
        """
        if self.session_active:
            raise RuntimeError("A session is already active")
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        self.session_active = True
        
        # Initialize session data
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now(),
            "chunks": [],
            "feedback_history": [],
            "type": session_type,
            "file_path": file_path
        }
        
        logger.info(f"🚀 Starting session {self.session_id} (type: {session_type})")
        
        try:
            # Create audio queue
            self.audio_queue = asyncio.Queue()
            if session_type == "file":
                # File mode
                if not file_path:
                    raise ValueError("file_path required for file mode")
                
                self.file_processor = FileAudioProcessor(self.assemblyai_key)
                logger.info(f"📁 File mode: {file_path}")
                
                # Start processing pipeline
                self.processing_task = asyncio.create_task(self._processing_pipeline())
                logger.info("✅ Processing pipeline started")
                
                # Start file processing in background
                asyncio.create_task(self.file_processor.process_file(file_path, self.audio_queue))
                
            else:
                # Live mode
                self.audio_processor = AudioProcessor(self.assemblyai_key)
                
                # Start live transcription
                await self.audio_processor.start_live_transcription(
                    self.audio_queue,
                    device_index=device_index
                )
                
                logger.info("✅ Audio processor initialized and connected to AssemblyAI")
                
                # Start processing pipeline
                self.processing_task = asyncio.create_task(self._processing_pipeline())
                logger.info("✅ Processing pipeline started")
            
            return self.session_id
            
        except Exception as e:
            error_msg = f"Failed to start session: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            self.session_active = False
            # Clean up on error
            if self.audio_processor:
                try:
                    await self.audio_processor.stop_transcription()
                except:
                    pass
            raise RuntimeError(error_msg) from e

    async def _processing_pipeline(self):
        """Main processing pipeline for audio chunks"""
        logger.info("🔄 Pipeline started")
        
        while self.session_active:
            try:
                # Get chunk from queue with timeout
                chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                
                # Process the chunk
                await self._process_chunk(chunk)
                
            except asyncio.TimeoutError:
                # No chunk received, continue waiting
                continue
            except Exception as e:
                if self.session_active:
                    logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        
        logger.info("⏹️ Pipeline stopped")

    async def _process_chunk(self, chunk: AudioChunk):
        """Process a single audio chunk through the analysis pipeline"""
        try:
            logger.info(f"🔄 Processing chunk from {chunk.speaker}: {chunk.transcript[:50]}...")
            
            # Store chunk
            self.session_data["chunks"].append(chunk)
            
            # Run ML inference
            inferences = await self.inference_engine.process_chunk(chunk)
            
            # Generate GROW phase analysis
            grow_phase = await self._analyze_grow_phase(chunk, inferences)
            
            # Calculate engagement score (FIXED: use interest_level, not engagement_level)
            engagement_score = inferences.interest_level
            
            # Assess coaching quality
            coaching_quality = await self._assess_coaching_quality(chunk, inferences, grow_phase)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(chunk, inferences, grow_phase)
            
            # Create real-time feedback
            feedback = RealTimeFeedback(
                timestamp=chunk.timestamp,
                speaker=chunk.speaker,
                grow_phase=grow_phase,
                emotion_trend=inferences.emotion,
                engagement_score=engagement_score,
                coaching_quality=coaching_quality,
                suggestions=suggestions
            )
            
            # Store feedback
            self.session_data["feedback_history"].append(feedback)
            
            # Broadcast to WebSocket clients
            await self._broadcast_feedback(feedback)
            
        except Exception as e:
            logger.error(f"❌ Chunk error: {e}", exc_info=True)

    async def _analyze_grow_phase(self, chunk: AudioChunk, inferences) -> GROWPhase:
        """Analyze GROW model phase with enhanced detection"""
        try:
            transcript_lower = chunk.transcript.lower()
            
            # Enhanced keyword detection for each phase
            goal_keywords = ["goal", "want", "achieve", "objective", "aim", "target", "hope to", "wish", "aspire"]
            reality_keywords = ["currently", "now", "situation", "reality", "actually", "right now", "present", 
                              "today", "at the moment", "happening", "problem", "challenge", "issue", "struggle"]
            options_keywords = ["option", "could", "might", "alternative", "what if", "perhaps", "maybe", 
                               "possibility", "choice", "consider", "explore", "idea"]
            wayforward_keywords = ["will", "plan", "next", "action", "step", "commit", "going to", "shall", 
                                  "decide", "do", "start", "begin", "implement"]
            
            # Count matches and weight by speaker
            goal_score = sum(2 if kw in transcript_lower else 0 for kw in goal_keywords)
            reality_score = sum(2 if kw in transcript_lower else 0 for kw in reality_keywords)
            options_score = sum(2 if kw in transcript_lower else 0 for kw in options_keywords)
            way_score = sum(2 if kw in transcript_lower else 0 for kw in wayforward_keywords)
            
            # Questions about goals/wants
            if "?" in chunk.transcript:
                if any(word in transcript_lower for word in ["want", "goal", "achieve"]):
                    goal_score += 3
                elif any(word in transcript_lower for word in ["what if", "could you"]):
                    options_score += 3
            
            # Determine phase
            scores = {
                "Goal": goal_score,
                "Reality": reality_score,
                "Options": options_score,
                "Way Forward": way_score
            }
            
            max_phase = max(scores.items(), key=lambda x: x[1])
            phase = max_phase[0]
            raw_score = max_phase[1]
            
            # Calculate confidence based on score
            if raw_score >= 6:
                confidence = 0.9
            elif raw_score >= 4:
                confidence = 0.75
            elif raw_score >= 2:
                confidence = 0.6
            else:
                # Default to Reality if no clear match
                phase = "Reality"
                confidence = 0.4
            
            reasoning = f"Detected {raw_score} phase indicators"
            
            return GROWPhase(
                phase=phase,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"GROW analysis error: {e}")
            return GROWPhase(phase="Reality", confidence=0.3, reasoning="Error in analysis")

    async def _assess_coaching_quality(self, chunk: AudioChunk, inferences, grow_phase: GROWPhase) -> Dict[str, float]:
        """Assess coaching quality metrics"""
        try:
            # Basic quality assessment
            transcript_lower = chunk.transcript.lower()
            
            # Check for open-ended questions
            questioning_score = 0.7
            if "?" in chunk.transcript:
                if any(word in transcript_lower for word in ["what", "how", "why", "tell me"]):
                    questioning_score = 0.9
            
            # Check for listening indicators
            listening_score = 0.6
            if any(word in transcript_lower for word in ["understand", "hear", "sounds like"]):
                listening_score = 0.85
            
            # Overall based on engagement and phase confidence
            overall = (questioning_score + listening_score + grow_phase.confidence) / 3
            
            return {
                "overall": overall,
                "questioning": questioning_score,
                "listening": listening_score
            }
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {"overall": 0.5, "questioning": 0.5, "listening": 0.5}

    async def _generate_suggestions(self, chunk: AudioChunk, inferences, grow_phase: GROWPhase) -> list:
        """Generate coaching suggestions with AI assistance"""
        suggestions = []
        
        try:
            # Get conversation context
            recent_chunks = self.session_data["chunks"][-5:]
            
            # Only use Gemini every 5th chunk to save quota
            chunk_count = len(self.session_data["chunks"])
            use_ai = (chunk_count % 5 == 0) and chunk.speaker == "coach" and self.gemini_analyzer
            
            if use_ai:
                try:
                    context = "\n".join([f"{c.speaker}: {c.transcript}" for c in recent_chunks])
                    prompt = f"""You are an expert coaching advisor. Based on this conversation, give ONE brief, actionable suggestion for the coach.

Recent conversation:
{context}

Provide ONE specific, actionable suggestion (max 15 words)."""

                    response = await self.gemini_analyzer.model.generate_content_async(prompt)
                    ai_suggestion = response.text.strip().replace('\n', ' ')[:150]
                    suggestions.append(ai_suggestion)
                except Exception as e:
                    logger.warning(f"Gemini suggestion failed: {e}")
                    # Fall through to heuristic suggestions
            
            # Heuristic suggestions (always run as fallback or primary)
            if not suggestions:
                if chunk.speaker == "coach" and "?" not in chunk.transcript:
                    suggestions.append("Try using an open-ended question to explore deeper")
                
                if inferences.interest_level < 0.4:
                    suggestions.append("Engagement is low - pause and ask what's on their mind")
                
                if grow_phase.phase == "Goal" and grow_phase.confidence > 0.7:
                    suggestions.append("Great! Now explore the current reality")
                elif grow_phase.phase == "Reality":
                    suggestions.append("Consider moving to explore options together")
                elif grow_phase.phase == "Options":
                    suggestions.append("Help them commit to specific next steps")
            
            # For coachee turns
            if chunk.speaker == "coachee":
                transcript_lower = chunk.transcript.lower()
                if "don't know" in transcript_lower or "confused" in transcript_lower:
                    suggestions.append("Coachee seems uncertain - help clarify their thinking")
                elif "worry" in transcript_lower or "anxious" in transcript_lower:
                    suggestions.append("Acknowledge their feelings, then explore what's in their control")
                elif inferences.interest_level < 0.5:
                    suggestions.append("Coachee engagement dropping - check in on their energy")
            
        except Exception as e:
            logger.error(f"Suggestion generation error: {e}")
            suggestions.append("Continue with active listening")
        
        return suggestions[:2]

    async def _broadcast_feedback(self, feedback: RealTimeFeedback):
        """Broadcast feedback to all connected WebSocket clients"""
        if not self.websocket_clients:
            logger.debug("No WebSocket clients connected")
            return
        
        try:
            # Get the latest chunk for transcript
            latest_chunk = self.session_data["chunks"][-1] if self.session_data["chunks"] else None
            
            # Get learning style from recent feedback
            learning_style = "Unknown"
            if len(self.session_data.get("feedback_history", [])) > 0:
                # Aggregate VAK from recent interactions
                learning_style = "Visual"  # Placeholder - should aggregate from VAK inferences
            
            # Convert feedback to dict for JSON serialization
            feedback_dict = {
                "timestamp": feedback.timestamp,
                "speaker": feedback.speaker,
                "transcript": latest_chunk.transcript if latest_chunk else "",
                "grow_phase": {
                    "phase": feedback.grow_phase.phase,
                    "confidence": feedback.grow_phase.confidence,
                    "reasoning": feedback.grow_phase.reasoning
                },
                "emotion_trend": feedback.emotion_trend,
                "engagement_score": feedback.engagement_score,
                "coaching_quality": feedback.coaching_quality,
                "suggestions": feedback.suggestions,
                "learning_style": learning_style,
                "digression_level": 0.1  # Placeholder - would come from ML model
            }
            
            message = json.dumps(feedback_dict)
            logger.info(f"📤 Broadcasting feedback to {len(self.websocket_clients)} clients")
            
            # Send to all clients
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send_text(message)
                    logger.debug("✅ Sent to client")
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}", exc_info=True)

    async def stop_session(self):
        """Stop the current session and generate report"""
        if not self.session_active:
            raise RuntimeError("No active session to stop")
        
        logger.info(f"⏹️ Stopping session {self.session_id}")
        
        # Mark session as inactive FIRST to stop pipeline
        self.session_active = False
        
        # Give pipeline time to finish current chunk
        await asyncio.sleep(0.5)
        
        # Stop processing task
        if self.processing_task:
            try:
                await asyncio.wait_for(self.processing_task, timeout=3.0)
                logger.info("✅ Processing task completed")
            except asyncio.TimeoutError:
                logger.warning("Processing task timed out")
                self.processing_task.cancel()
        
        # Stop audio processor
        if self.audio_processor:
            try:
                await asyncio.wait_for(
                    self.audio_processor.stop_transcription(),
                    timeout=5.0
                )
                logger.info("✅ Audio processor stopped")
            except asyncio.TimeoutError:
                logger.warning("Audio processor stop timed out")
            except Exception as e:
                logger.error(f"Error stopping audio processor: {e}")
        
        # Generate final report
        try:
            report_data = {
                'session_id': self.session_id,
                'duration': (datetime.now() - self.session_data["start_time"]).total_seconds() / 60,
                'chunks': [
                    {
                        'speaker': chunk.speaker,
                        'transcript': chunk.transcript,
                        'timestamp': chunk.timestamp
                    }
                    for chunk in self.session_data["chunks"]
                ]
            }
            
            if self.gemini_analyzer:
                try:
                    report = await asyncio.wait_for(
                        self.gemini_analyzer.generate_session_report(report_data),
                        timeout=30.0
                    )
                    logger.info("✅ Report generated with Gemini")
                except asyncio.TimeoutError:
                    logger.warning("Gemini report timed out, using basic report")
                    report = self._generate_basic_report(report_data)
                except Exception as e:
                    logger.warning(f"Gemini report failed, using basic report: {e}")
                    report = self._generate_basic_report(report_data)
            else:
                report = self._generate_basic_report(report_data)
                logger.info("✅ Basic report generated")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            report = self._generate_basic_report({'session_id': self.session_id, 'duration': 0, 'chunks': []})
        
        # Store session (optional)
        try:
            await self._store_session(report)
        except Exception as e:
            logger.warning(f"⚠️ Failed to store: {e}")
        
        logger.info(f"✅ Session {self.session_id} completed")
        
        return report

    async def _store_session(self, report):
        """Store session data in ChromaDB"""
        # This can be enhanced to actually store in ChromaDB
        # For now, just log it
        pass

    def _generate_basic_report(self, report_data: Dict[str, Any]) -> SessionReport:
        """Generate basic report without Gemini"""
        chunks = report_data.get('chunks', [])
        coach_chunks = [c for c in chunks if c.get('speaker') == 'coach']
        coachee_chunks = [c for c in chunks if c.get('speaker') == 'coachee']
        
        return SessionReport(
            session_id=report_data.get('session_id', 'unknown'),
            duration_minutes=report_data.get('duration', 0),
            participants={
                "coach": {"engagement_avg": 0.5, "total_turns": len(coach_chunks)},
                "coachee": {"engagement_avg": 0.5, "total_turns": len(coachee_chunks)}
            },
            grow_phases=[],
            emotional_journey={"coach": [], "coachee": []},
            learning_style_analysis={"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34},
            key_insights=[f"Session completed with {len(chunks)} total interactions"],
            coaching_effectiveness={
                "overall": 0.5,
                "questioning": 0.5,
                "listening": 0.5
            },
            recommendations=["Continue coaching sessions for better insights"],
            transcript_summary=f"Session with {len(coach_chunks)} coach turns and {len(coachee_chunks)} coachee turns"
        )

    def get_available_audio_devices(self):
        """Get list of available audio input devices"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })
            
            p.terminate()
            return devices
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []