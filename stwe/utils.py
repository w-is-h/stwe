import logging
import time

FORMAT = logging.Formatter('%(asctime)s %(app_name)s: %(message)s')
logging.basicConfig(filename='/tmp/stwe.log', level=logging.INFO)

def get_logger(name=''):
    if len(name) > 0:
        name = "." + name
    logging.basicConfig(format=FORMAT)

    return logging.getLogger('stwe' + name)
