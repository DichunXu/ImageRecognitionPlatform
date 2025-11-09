import os
import json
import argparse
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Iterator
from sqlalchemy import create_engine, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column, Session as SQLAlchemySession
from connect import build_mysql_url as _build_mysql_url, test_connection as _test_connection, connect_with_params as _connect_with_params
def _build_database_url_from_env() -> Optional[str]:
    host = os.getenv('DB_HOST')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    name = os.getenv('DB_NAME')
    port = os.getenv('DB_PORT') or '3306'
    if all([host, user, password, name]):
        return _build_mysql_url(host=host, port=port, user=user, password=password, dbname=name)
    return None
DATABASE_URL = os.getenv('DATABASE_URL') or os.getenv('MYSQL_URL') or _build_database_url_from_env()

Base = declarative_base()
_engine = None
SessionLocal: Optional[sessionmaker] = None
class UploadHistory(Base):
    __tablename__ = 'upload_history'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_path: Mapped[str] = mapped_column(String(512), nullable=True)
    model_label: Mapped[str] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    result_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    labels: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    def to_dict(self) -> dict:
        return {
            'id': int(self.id) if self.id is not None else None,
            'time': self.time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'model_label': self.model_label,
            'filename': self.filename,
            'result_filename': self.result_filename,
            'labels': (json.loads(self.labels) if self.labels else None) if isinstance(self.labels, str) else self.labels,
        }
class TrainRun(Base):
    __tablename__ = 'train_run'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    returncode: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    args_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True) 
    exp_dir: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    train_log: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    def args(self) -> Optional[dict]:
        try:
            return json.loads(self.args_json) if self.args_json else None
        except Exception:
            return None
def set_database_url(url: str):
    global DATABASE_URL, _engine, SessionLocal
    DATABASE_URL = url
    _engine = None
    SessionLocal = None
def setup_database_from_env(logger=None, *, echo: bool = False):
    global _engine, SessionLocal
    if not DATABASE_URL:
        return None, None
    if _engine is not None and SessionLocal is not None:
        return _engine, SessionLocal
    _engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, future=True, echo=echo)
    SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    if logger:
        logger.info(f"数据库已配置: {DATABASE_URL.split('@')[-1]}")
    return _engine, SessionLocal
def init_db(logger=None):
    engine, _ = setup_database_from_env(logger)
    if engine is None:
        if logger:
            logger.info('DATABASE_URL 未设置，跳过数据库初始化')
        return False
    Base.metadata.create_all(engine)
    if logger:
        logger.info('数据库表已初始化')
    return True
def build_mysql_url(*, host: str, port: str | int = 3306, user: str, password: str, dbname: str) -> str:
    return _build_mysql_url(host=host, port=port, user=user, password=password, dbname=dbname)
def connect_with_params(*, host: str, port: str | int = 3306, username: str, password: str, database: str, echo: bool = False, init: bool = False) -> bool:
    def _set_db(url: str):
        set_database_url(url)

    def _do_init():
        try:
            init_db()
        except Exception:
            pass
    return _connect_with_params(host=host, port=port, username=username, password=password, database=database, echo=echo, init=init, set_db_callback=_set_db, init_callback=_do_init)
def get_engine(echo: bool = False):
    engine, _ = setup_database_from_env(None, echo=echo)
    return engine
@contextmanager
def get_session(echo: bool = False) -> Iterator[SQLAlchemySession]:
    if DATABASE_URL is None:
        raise RuntimeError('数据库未配置（缺少 DATABASE_URL）')
    _, Session = setup_database_from_env(None, echo=echo)
    assert Session is not None
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
def test_connection(echo: bool = False) -> bool:
    if DATABASE_URL is None:
        return False
    return _test_connection(DATABASE_URL, echo=echo)
def _main():
    parser = argparse.ArgumentParser(description='DB utility for MySQL (SQLAlchemy)')
    parser.add_argument('--url', help='Database URL, e.g. mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4')
    parser.add_argument('--host')
    parser.add_argument('--port', default='3306')
    parser.add_argument('--user', dest='user')
    parser.add_argument('--username', dest='user', help='Alias of --user')
    parser.add_argument('--password')
    parser.add_argument('--db', dest='dbname')
    parser.add_argument('--echo', action='store_true', help='Echo SQL statements')
    parser.add_argument('--init', action='store_true', help='Create tables if not exist')
    args = parser.parse_args()
    url = args.url
    if not url and args.host and args.user and args.password and args.dbname:
        url = build_mysql_url(host=args.host, port=args.port, user=args.user, password=args.password, dbname=args.dbname)
    if url:
        set_database_url(url)
    if DATABASE_URL is None:
        print('DATABASE_URL 未配置，且未通过参数提供 --url/--host 等。')
        raise SystemExit(2)
    print(f'使用数据库: {DATABASE_URL}')
    ok = test_connection(echo=args.echo)
    print(f'连接测试: {"成功" if ok else "失败"}')
    if not ok:
        raise SystemExit(1)
    if args.init:
        created = init_db()
        print(f'初始化建表: {"完成" if created else "跳过"}')
if __name__ == '__main__':
    _main()
