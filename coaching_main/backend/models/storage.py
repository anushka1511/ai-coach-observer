import asyncio
import json
import os
from typing import Dict

import chromadb
from backend.schemas.data_models import SessionReport


class ChromaDBStorage:
    """Handles ChromaDB integration for session report persistence."""

    def __init__(self, persist_directory: str = None):
        persist_directory = persist_directory or os.getenv("CHROMADB_PERSIST_DIR", "./.chromadb")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.sessions_collection = self.client.get_or_create_collection("coaching_sessions")

    async def store_session_report(self, report: SessionReport):
        """Store final session report summary & metadata"""
        raw = report.model_dump()
        metadata = {
            k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
            for k, v in raw.items()
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.sessions_collection.add(
                documents=[report.transcript_summary],
                metadatas=[metadata],
                ids=[report.session_id],
            ),
        )
