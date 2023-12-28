import os
import errno
import logging
import time


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def log_creater(log_dir,logger_name):
    make_sure_path_exists(log_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(log_dir, log_name)
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] ==> %(message)s')
    file.setFormatter(formatter)
    stream.setFormatter(formatter)
    log.addHandler(file)
    log.addHandler(stream)
    log.info('creating {}'.format(final_log_file))
    return log


def log_remove_handlers(logger):
    if logger is not None:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)


if __name__ == "__main__":
    logger = log_creater('../log/test')