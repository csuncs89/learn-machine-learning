"""
Decoration utilities
"""
import functools


def print_banner(title):

    def identity_wrapper(func):

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            print('-' * 79)
            print(title)
            ret = func(*args, **kwargs)
            print('-' * 79 + '\n')
            return ret

        return func_wrapper

    return identity_wrapper


@print_banner(title='title')
def foobar(*args):
    print(args)

'''
# Above code equals

def foobar(*args):
    print(args)
foobar = print_banner('title')(foobar)

# After calling print_banner

def foobar(*args):
    print(args)
foobar = identity_wrapper(foobar) # 'title' is in the calling frame (closure)

# After calling identity_wrapper

def foobar(*args):
    print(args)

# Suppose do not use functools.wraps
def func_wrapper(*args, **kwargs):
    print('-' * 79 + '\n')
    print(title)
    ret = func(*args, **kwargs)
    print('-' * 79 + '\n')
    return ret

foobar = func_wrapper # The func foobar is in the calling frame (closure)
'''


def main():
    foobar('a', 'b', 1, 2)

if __name__ == '__main__':
    main()
