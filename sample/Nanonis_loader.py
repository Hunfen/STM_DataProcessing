# -*- coding: utf-8 -*-
import os
import re

import numpy as np


def loader(f_path: str):
    """To load the data files created by Nanonis SPM controller. 
    
    Args:
        f_path : Path to the data file.
    
    Returns:
        {
            __Nanonis_sxm__ : if file type is .sxm.
            __Nanonis_dat__ : if file type is .dat.
            __Nanonis_3ds__ : if file type is .3ds.
        }
    
    Raises:
        KeyError : If file type is still not supported.

    """
    switch = {
        '.sxm': lambda x: __Nanonis_sxm__(x),
        # '.dat': lambda x: __Nanonis_dat__(x),
        # '.3ds': lambda x: __Nanonis_3ds__(x)
    }
    try:
        return switch[os.path.splitext(f_path)[1]](f_path)
    except KeyError:
        print('File type not supported.')


class __Nanonis_sxm__:
    """[summary]
    
    """
    def __init__(self, f_path: str):
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]

    def __sxm_header_reader__(self, f_path: str):
        """[summary]

        Args:
            f_path: [description]
        """
        __entry = ''
        __contents = ''
        __raw_header = {}
        with open(f_path, 'rb') as __f:
            __header_end = False
            while not __header_end:
                __line = __f.readline().decode(encoding='utf-8',
                                               errors='replace')
                if re.match(':SCANIT_END:\n', __line):
                    __header_end = True
                # ':.+:' is the regex of the Nanonis .sxm file header entry.
                elif re.match(':.+:', __line):
                    __entry = __line[1:-2]  # Read header_entry
                    __contents = ''  # Clear __contents
                else:
                    __contents += __line
                    __raw_header[__entry] = __contents.strip(
                        '\n')  # remove EOL
        self.__raw_header = __raw_header


# class __Nanonis_dat__:

#     def __init__(self, f_path):
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]

# class __Nanonis_3ds__:

#     def __init__(self, f_path):
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]
