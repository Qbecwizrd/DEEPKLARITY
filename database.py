from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

## defining the PostgreSQL database URL(replace with your credentials)
database_url = "postgresql://postgres:12345@localhost:5432/resumedb"

##craeting the sqlalchemy engine
engine=create_engine(database_url)


##creating a session factory for database transactions
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)

##base for declaring ORM Models
Base=declarative_base()