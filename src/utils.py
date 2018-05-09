import logging

def get_logger(name=''):
    formatter = logging.Formatter('%(asctime)s %(app_name)s: %(message)s')
    logging.setFormatter(
    return logging.getLogger('twe.' + name)
