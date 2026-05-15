import sys

import uvicorn
from loguru import logger

from api.config import api_settings


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=api_settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )


def main() -> None:
    configure_logging()
    uvicorn.run(
        "api.app:app",
        host=api_settings.api_host,
        port=api_settings.api_port,
        reload=False,
        log_config=None,  # отдаём логи loguru
    )


if __name__ == "__main__":
    main()