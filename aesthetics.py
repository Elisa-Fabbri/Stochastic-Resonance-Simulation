import sys
from contextlib import contextmanager

def progress(iterator):
    current_iteration = 0
    total_iterations = len(iterator)
    update_interval = int(total_iterations/100) if total_iterations>100 else 1
    for element in iterator:
        current_iteration += 1
        if current_iteration % update_interval == 0:
            progress_percentage = int(current_iteration / total_iterations * 100)
            sys.stdout.write('\r[' + '-' * progress_percentage + ' ' * (100 - progress_percentage) + f'] {progress_percentage}%\r')
            sys.stdout.flush()
        yield element
    sys.stdout.write('\r[' + '-' * 100 + '] 100%\n')
    sys.stdout.flush()

@contextmanager
def green_text():
    print('\033[32m', end='')
    try:
        yield
    finally:
        print("\033[0m", end='')

@contextmanager
def orange_text():
    print('\033[33m', end='')
    try:
        yield
    finally:
        print("\033[0m", end='')

@contextmanager
def red_text():
    print('\033[91m', end='')
    try:
        yield
    finally:
        print("\033[0m", end='')

