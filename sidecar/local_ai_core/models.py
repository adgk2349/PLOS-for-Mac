from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    HYBRID = "HYBRID"
    CONFIRM_BEFORE_EXTERNAL = "CONFIRM_BEFORE_EXTERNAL"


class WorkMode(str, Enum):
    GENERAL = "GENERAL"
    SUMMARY = "SUMMARY"
    RESEARCH = "RESEARCH"
    DEVELOPMENT = "DEVELOPMENT"
    WRITING = "WRITING"
    PLANNING = "PLANNING"
    STRICT_SEARCH = "STRICT_SEARCH"


class StartupProfile(str, Enum):
    FAST = "FAST"
    RECOMMENDED = "RECOMMENDED"
    DEEP = "DEEP"


class WorkspaceUpdateRequest(BaseModel):
    included_paths: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    startup_profile: StartupProfile = StartupProfile.RECOMMENDED
    default_mode: WorkMode = WorkMode.GENERAL


class WorkspaceResponse(WorkspaceUpdateRequest):
    updated_at: datetime


class IndexJobRequest(BaseModel):
    scope: Literal["incremental", "full"]


class IndexJobStatus(BaseModel):
    job_id: str
    scope: Literal["incremental", "full"]
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    processed_files: int = 0
    failed_files: int = 0
    stage: Literal["queued", "scan", "parse", "embed", "store", "done", "failed"] = "queued"
    error: str | None = None


class FailureItem(BaseModel):
    path: str
    reason: str
    last_attempt_at: datetime


class FailureListResponse(BaseModel):
    failures: list[FailureItem]


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    file_path: str
    snippet: str
    score: float
    modified_at: datetime


class LocalChatRequest(BaseModel):
    query: str
    mode: WorkMode = WorkMode.GENERAL
    conversation_id: str | None = None
    top_k: int | None = None


class LocalChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    mode: WorkMode
    used_profile: StartupProfile
    is_local: bool = True


class DeepAnalysisRequest(BaseModel):
    query: str
    mode: WorkMode = WorkMode.GENERAL
    provider: Literal["openai", "anthropic"]
    selected_citations: list[Citation] = Field(default_factory=list)
    user_confirmed: bool = False


class ExternalCallEvent(BaseModel):
    provider: str
    sent_chars: int
    approved_by_user: bool
    timestamp: datetime


class DeepAnalysisResponse(BaseModel):
    answer: str
    provider: str
    event: ExternalCallEvent
    is_local: bool = False


class SettingsModel(BaseModel):
    privacy_mode: PrivacyMode = PrivacyMode.HYBRID
    startup_profile: StartupProfile = StartupProfile.RECOMMENDED
    model_profile: str = "recommended"
    reindex_policy: Literal["filewatch_incremental", "manual_only", "scheduled_full"] = "filewatch_incremental"
    language: str = "ko-KR"
