# app/models/blog_image.py
from sqlalchemy import Column, Integer, ForeignKey, Text, LargeBinary
from app.db.database import Base

class BlogImage(Base):
    __tablename__ = "blog_images"

    id = Column(Integer, primary_key=True)
    blog_session_id = Column(Integer, ForeignKey("blog_sessions.id", ondelete="CASCADE"))  # ✅ lowercase
    filename = Column(Text)
    alt = Column(Text)
    caption = Column(Text)
    image_data = Column(LargeBinary)