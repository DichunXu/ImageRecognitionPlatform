from typing import Tuple
import os


def attempt_database_setup(connect_with_params, init_db, logger) -> Tuple[bool, str]:
    DB_ENABLED = False
    msg = ''
    try:
        DB_DIRECT_HOST = os.getenv('DB_DIRECT_HOST')
        DB_DIRECT_PORT = os.getenv('DB_DIRECT_PORT')
        DB_DIRECT_USER = os.getenv('DB_DIRECT_USER')
        DB_DIRECT_PASS = os.getenv('DB_DIRECT_PASS')
        DB_DIRECT_NAME = os.getenv('DB_DIRECT_NAME')
        if connect_with_params is not None and all([DB_DIRECT_HOST, DB_DIRECT_USER, DB_DIRECT_PASS, DB_DIRECT_NAME]):
            enabled = bool(connect_with_params(
                host=DB_DIRECT_HOST,
                port=DB_DIRECT_PORT,
                username=DB_DIRECT_USER,
                password=DB_DIRECT_PASS,
                database=DB_DIRECT_NAME,
                echo=False,
                init=True,
            ))
            DB_ENABLED = enabled
            msg = f"使用直接参数连接数据库: {'成功' if enabled else '失败'}"
            if logger:
                logger.info(msg)
        elif init_db is not None:
            enabled = bool(init_db(logger))
            DB_ENABLED = enabled
            msg = f"init_db fallback: {'成功' if enabled else '失败'}"

    except Exception as e:
        msg = f"数据库初始化失败，继续以文件模式运行: {e}"
        if logger:
            logger.warning(msg)

    return DB_ENABLED, msg


__all__ = ["attempt_database_setup"]
