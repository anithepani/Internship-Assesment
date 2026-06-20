# src/reports/database.py
"""
Module 4 (support) — Database layer
SQLite + SQLAlchemy for persistent compliance event storage.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column, String, DateTime, Text, Float, Integer, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "outputs" / "compliance.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class ViolationEvent(Base):
    __tablename__ = "violation_events"

    event_id        = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    clip_id         = Column(String, nullable=False)
    zone            = Column(String, default="Zone-1")
    behavior_class  = Column(String, nullable=False)
    policy_rule_ref = Column(String, nullable=False)
    event_description = Column(Text, nullable=False)
    severity        = Column(String, nullable=False)   # LOW/MEDIUM/HIGH/CRITICAL
    escalation_action = Column(String, nullable=False)
    confidence      = Column(Float, default=0.0)
    frame_number    = Column(Integer, default=0)
    frame_path      = Column(String, default="")       # path to saved frame image


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


def log_event(
    clip_id: str,
    behavior_class: str,
    policy_rule_ref: str,
    event_description: str,
    severity: str,
    escalation_action: str,
    zone: str = "Zone-1",
    confidence: float = 0.0,
    frame_number: int = 0,
    frame_path: str = "",
) -> ViolationEvent:
    """Write a violation event to the database. Returns the saved event."""
    init_db()
    session = get_session()
    try:
        event = ViolationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            clip_id=clip_id,
            zone=zone,
            behavior_class=behavior_class,
            policy_rule_ref=policy_rule_ref,
            event_description=event_description,
            severity=severity,
            escalation_action=escalation_action,
            confidence=confidence,
            frame_number=frame_number,
            frame_path=frame_path,
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event
    finally:
        session.close()


def get_all_events(
    severity_filter: list[str] | None = None,
    behavior_filter: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Retrieve events from DB with optional filters."""
    init_db()
    session = get_session()
    try:
        q = session.query(ViolationEvent)
        if severity_filter:
            q = q.filter(ViolationEvent.severity.in_(severity_filter))
        if behavior_filter:
            q = q.filter(ViolationEvent.behavior_class == behavior_filter)
        events = q.order_by(ViolationEvent.timestamp.desc()).limit(limit).all()
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                "clip_id": e.clip_id,
                "zone": e.zone,
                "behavior_class": e.behavior_class,
                "policy_rule_ref": e.policy_rule_ref,
                "event_description": e.event_description,
                "severity": e.severity,
                "escalation_action": e.escalation_action,
                "confidence": e.confidence,
                "frame_number": e.frame_number,
                "frame_path": e.frame_path,
            }
            for e in events
        ]
    finally:
        session.close()
