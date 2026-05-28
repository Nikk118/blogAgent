from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

from app.db.database import Base


class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    firebase_uid = Column(
        String,
        unique=True,
        nullable=False
    )

    email = Column(
        String,
        nullable=False
    )