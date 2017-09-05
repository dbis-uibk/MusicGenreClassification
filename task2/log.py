import logging
import sys
import datetime
from os.path import join


class StreamToLogger():
    def __init__(self, logger, log_level=logging.DEBUG):
        self.logger = logger
        self.log_level = log_level
        self.linebuf=''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    timestamp = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M')
    fh = logging.FileHandler(join('logs', '%s__%s.log' % (name, timestamp)), 'w')
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger
