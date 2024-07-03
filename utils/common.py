import logging


logger = logging.getLogger()


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def postprocess(cls, input_str):
        input_str = input_str.replace("<h>", cls.HEADER)
        input_str = input_str.replace("<blue>", cls.OKBLUE)
        input_str = input_str.replace("<green>", cls.OKGREEN)
        input_str = input_str.replace("<yellow>", cls.WARNING)
        input_str = input_str.replace("<red>", cls.FAIL)
        input_str = input_str.replace("</>", cls.ENDC)
        input_str = input_str.replace("<b>", cls.BOLD)
        input_str = input_str.replace("<u>", cls.UNDERLINE)
        input_str = input_str.replace("<clean>", "")
        return input_str
