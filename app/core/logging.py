import logging


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # uvicorn --reload + WatchFiles：IDE 保存会刷屏 "N changes detected"
    for name in ("watchfiles", "watchfiles.main"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("med_rag_graph")


logger = setup_logging()
