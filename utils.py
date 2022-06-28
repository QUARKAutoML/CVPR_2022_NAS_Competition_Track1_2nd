import paddle
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    # model.save(filename, True)
    paddle.save(model.state_dict(), filename)


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('cvpr2022_workshop')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path, mode="w+")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def compute_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    sub_center = sub_kernel_size // 2
    start = center - sub_center
    end = center + sub_center + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    assert isinstance(kernel_size, int)
    assert kernel_size % 2 > 0, "kernel size must be odd number"
    return kernel_size // 2


def convert_to_list(value, n):
    return [value, ] * n