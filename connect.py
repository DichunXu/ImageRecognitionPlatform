import os
from typing import Optional, Callable
from sqlalchemy import create_engine


def build_mysql_url(*, host: str, port: str | int = 3306, user: str, password: str, dbname: str) -> str:
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=utf8mb4"


def test_connection(url: str, *, echo: bool = False) -> bool:
    if not url:
        return False
    try:
        engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600, future=True, echo=echo)
        with engine.connect() as conn:
            conn.exec_driver_sql('SELECT 1')
        return True
    except Exception:
        return False


def connect_with_params(*, host: str, port: str | int = 3306, username: str, password: str, database: str, echo: bool = False, init: bool = False, set_db_callback: Optional[Callable[[str], None]] = None, init_callback: Optional[Callable[[], None]] = None) -> bool:
    url = build_mysql_url(host=host, port=port, user=username, password=password, dbname=database)
    try:
        if set_db_callback:
            set_db_callback(url)
    except Exception:
        pass
    ok = test_connection(url, echo=echo)
    if ok and init and init_callback:
        try:
            init_callback()
        except Exception:
            pass
    return ok


__all__ = ["build_mysql_url", "test_connection", "connect_with_params"]
