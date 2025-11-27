# db.py
import os
import uuid
from datetime import datetime

from sqlalchemy import create_engine, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# В docker будем хранить DB в volume: /app/data/chess.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chess.db")

engine = create_engine(
    DATABASE_URL,
    # для sqlite нужно это:
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    status: Mapped[str] = mapped_column(String(20), default="ongoing", nullable=False)  # ongoing/finished
    human_color: Mapped[str] = mapped_column(String(1), default="w", nullable=False)   # "w" or "b"
    result: Mapped[str | None] = mapped_column(String(10), nullable=True)              # "1-0", "0-1", "1/2-1/2"
    fen: Mapped[str] = mapped_column(Text, nullable=False)

    moves: Mapped[list["Move"]] = relationship(back_populates="game", cascade="all, delete-orphan")


class Move(Base):
    __tablename__ = "moves"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(36), ForeignKey("games.id"), index=True, nullable=False)

    ply: Mapped[int] = mapped_column(Integer, nullable=False)  # 1..N полуход
    uci: Mapped[str] = mapped_column(String(10), nullable=False)
    san: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    game: Mapped["Game"] = relationship(back_populates="moves")


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
