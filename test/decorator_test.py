# -*- coding: utf-8 -*-
# -*- test the decorator -*-

from typing import Union
import re


def __dat_header_reformer__(func):
    def wrapper(*arg, **kwarg):
        raw_header = func(*arg, **kwarg)
        header: dict[str, Union[str, float]] = {}
        header_ls: list[list[str]] = []
        for i in range(len(raw_header)):
            header_ls.append(raw_header[i].strip('\n').strip('\t').split('\t'))
        header_ls = header_ls[:-1]  # remove last element
        raw_header = []  # release memory
        for i in range(len(header_ls)):
            try:
                header[header_ls[i][0]] = __is_number__(header_ls[i][1])
            except IndexError:
                header[header_ls[i][0]] = ''
        header_ls = []  # release memory
        return header

    return wrapper


@__dat_header_reformer__
def __dat_header_reader__(f_path):
    raw_header: list[str] = []
    with open(f_path, 'r') as f:
        header_end = False
        while not header_end:
            line = f.readline()
            if re.match(r'\[DATA\]', line):
                header_end = True
            else:
                raw_header.append(line)
    return raw_header


def __is_number__(s: str) -> Union[int, float, str]:
    """String converter.

    Args:
        s (str): Input string.

    Returns:
        Union[int, float, str]: Convert an input string into int or float
        if possible, or it would return the input string itself.
    """
    if s.isdigit():  # judge if str is an int.
        return int(s)
    else:
        try:
            return float(s)  # convert into float.
        except ValueError:

            return s  # return input string without change.


# from

# def timeit(func):
#     def wrapper(func):

#         func()

#     return wrapper

print(
    __dat_header_reader__(
        '/Users/hunfen/Documents/Experimental/STM1500_data/' +
        '2021/2021-12-14/Bias-Spectroscopy00013.dat'))
