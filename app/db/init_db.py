from app.db.database import Base
from app.db.database import engine

from app.models.user import User
from app.models.blog_session import BlogSession

Base.metadata.create_all(bind=engine)

print("TABLES CREATED ✅")