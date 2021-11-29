# -*- coding: utf-8 -*-
import os
import re

import numpy as np
from typing import Union


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


def __is_number(s: str) -> Union[float, str]:
    """String converter.

    Args:
        s (str): Input string.

    Returns:
        Union[float, str]: Convert an input string into float if possible, or it would return the input string itself.
    """
    try:
        return float(s)
    except ValueError:
        return s


class __Nanonis_sxm__:
    """[summary]
    """
    file_path: str
    fname: str
    __raw_header: 'dict[str, str]'
    header: 'dict[str, Union[dict[str, Union[float, str]], dict[str, dict[str, Union[float, str]]], list[float], float, str]]'

    def __init__(self, f_path: str) -> None:
        """[summary]

        Args:
            f_path (str): [description]
        """
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__raw_header = self.__sxm_header_reader__(f_path) 
        self.header = self.__sxm_header_reform__(self.__raw_header)

    def __sxm_header_reader__(self, f_path: str) -> 'dict[str, str]':
        """[summary]

        Returns:
            [type]: [description]
        """
        entry: str = ''
        contents: str = ''
        raw_header: dict[str, str] = {}
        with open(f_path, 'rb') as f:  # read the .sxm file
            header_end = False
            while not header_end:
                line = f.readline().decode(encoding='utf-8', errors='replace')
                # if reach the end of header
                if re.match(':SCANIT_END:\n', line):
                    header_end = True
                # ':.+:' is the regex of the Nanonis .sxm file header entry.
                elif re.match(':.+:', line):
                    entry = line[1:-2]  # Read header_entry
                    contents = ''  # Clear contents
                else:  # Load entries & corresponding parameters into pre-defined dict
                    contents += line
                    raw_header[entry] = contents.strip('\n')  # remove EOL
        return raw_header

    def __sxm_header_reform__(
        self, raw_header: 'dict[str, str]'
    ) -> 'dict[str, Union[dict[str, Union[float, str]], dict[str, dict[str, Union[float, str]]], list[float], float, str]]': 
        """[summary]

        Returns:
            [type]: [description]
        """
        scan_info: list[str] = [  # Scan information
            'ACQ_TIME', 'BIAS', 'Scan>lines', 'Scan>pixels/line',
            'Scan>speed backw. (m/s)', 'Scan>speed forw. (m/s)', 'COMMENT',
            'REC_DATE', 'REC_TIME', 'SCAN_DIR', 'SCAN_FILE'
        ]
        # scan_field_key: list[str] = [
        # 'X_OFFSET', 'Y_OFFSET', 'X_RANGE', 'Y_RANGE', 'ANGLE'
        # ]  # The order of scan_field_key should not be changed
        # trash_bin: list[str] = [
        #     'NANONIS_VERSION', 'REC_TEMP', 'SCANIT_TYPE', 'SCAN_ANGLE',
        #     'SCAN_OFFSET', 'SCAN_PIXELS', 'SCAN_RANGE', 'SCAN_TIME',
        #     'Scan>channels'
        # ]

        header: dict[str, Union[dict[str, Union[float, str]],
                                dict[str, dict[str, Union[float, str]]],
                                list[float], float, str]] = {}
        entries: list[str] = list(raw_header.keys())

        for i in enumerate(entries):
            if i[1] in scan_info:  # float type header
                header[i[1]] = __is_number(raw_header[i[1]].strip(' '))
            elif i[1] == 'Z-CONTROLLER':  # Table type header: information of feedback z controller
                z_controller: dict[str, Union[float, str]] = {}
                z_controller_config: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[0].strip('\t').split('\t')
                z_controller_configVar: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[1].strip('\t').split('\t')
                for j, s in enumerate(z_controller_config):
                    z_controller[s] = __is_number(z_controller_configVar[j])
                header[i[1]] = z_controller
            elif i[1] == 'DATA_INFO':  # data channel information
                data_info: dict[str, dict[str, Union[float, str]]] = {}
                raw_data_info: list[str] = raw_header[i[1]].split('\n')
                config = raw_data_info[0].strip('\t').split('\t')
                for i in range(1, len(raw_data_info)):
                    channel_info: dict[str, Union[float, str]] = {
                    }  # Initialization of dict channel information
                    name: str = ''  # Initialization of dict channel name
                    for j in range(len(config)):
                        if config[j] == 'Name':
                            name = raw_data_info[i].strip('\t').split('\t')[j]
                        else:
                            channel_info[config[j]] = __is_number(
                                raw_data_info[i].strip('\t').split('\t')[j])
                    data_info[name] = channel_info
                header[i[1]] = data_info
            elif i[1] == 'Scan>Scanfield':
                # [X_OFFSET, Y_OFFSET, X_RANGE, Y_RANGE, ANGLE] in float type
                header[i[1]] = [float(j) for j in raw_header[i[1]].split(';')]
        return header
# TODO: Getting data & rotation

# class __Nanonis_dat__:
#     """

#     """

#     file_path: str
#     fname: str

#     def __init__(self, f_path: str) -> None:
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]

# class __Nanonis_3ds__:
#     """[summary]
#     """
#     def __init__(self, f_path: str) -> None:
#         self.file_path = os.path.split(f_path)[0]
#         self.fname = os.path.split(f_path)[1]
