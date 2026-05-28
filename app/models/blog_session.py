from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import Text

from sqlalchemy.dialects.postgresql import JSONB

from app.db.database import Base


class BlogSession(Base):

    __tablename__ = "blog_sessions"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    title = Column(Text)

    prompt = Column(Text)

    content = Column(JSONB)