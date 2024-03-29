# -*- coding: utf-8 -*-
import os
import re

import numpy as np
from typing import Literal, Union, Tuple


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


def __is_number__(s: str) -> Union[float, str]:
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
    def __init__(self, f_path: str) -> None:
        """[summary]

        Args:
            f_path (str): [description]
        """

        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__raw_header = self.__sxm_header_reader__(f_path)
        self.header = self.__sxm_header_reform__(self.__raw_header)
        self.data, self.channel_dir = self.__sxm_data_reader__(
            f_path, self.header)

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
    ) -> 'dict[str, Union[dict[str, Union[float, str]], dict[str, dict[str, Union[float, str]]], tuple[float], float, str]]':
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
                                tuple[float], float, str]] = {}
        entries: list[str] = list(raw_header.keys())
        for i in enumerate(entries):
            if i[1] in scan_info:  # float type header
                header[i[1]] = __is_number__(raw_header[i[1]].strip(' '))
            elif i[1] == 'Z-CONTROLLER':  # Table type header: information of feedback z controller
                z_controller: dict[str, Union[float, str]] = {}
                z_controller_config: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[0].strip('\t').split('\t')
                z_controller_configVar: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[1].strip('\t').split('\t')
                for j, s in enumerate(z_controller_config):
                    # TODO: regex to split value str with units
                    z_controller[s] = __is_number__(z_controller_configVar[j])
                header[i[1]] = z_controller
            elif i[1] == 'DATA_INFO':  # data channel information
                data_info: dict[str, dict[str, Union[float, str]]] = {}
                raw_data_info: list[str] = raw_header[i[1]].split('\n')
                config = raw_data_info[0].strip('\t').split('\t')
                for j in range(1, len(raw_data_info)):
                    channel_info: dict[str, Union[float, str]] = {
                    }  # Initialization of dict channel information
                    name: str = ''  # Initialization of dict channel name
                    for k in range(len(config)):
                        if config[k] == 'Name':
                            name = raw_data_info[j].strip('\t').split('\t')[k]
                        else:
                            channel_info[config[k]] = __is_number__(
                                raw_data_info[j].strip('\t').split('\t')[k])
                    data_info[name] = channel_info
                header[i[1]] = data_info
            elif i[1] == 'Scan>Scanfield':
                # (X_OFFSET, Y_OFFSET, X_RANGE, Y_RANGE, ANGLE) in float type
                header[i[1]] = tuple(
                    float(j) for j in raw_header[i[1]].split(';'))
        return header

    def __sxm_data_reader__(self, f_path: str,
                            header: dict) -> Tuple[np.ndarray, tuple]:

        with open(f_path, 'rb') as f:
            read_all = f.read()
            offset = read_all.find('\x1A\x04'.encode(encoding='utf-8'))
            f.seek(offset + 2)
            data = np.fromfile(f, dtype='>f')
        # Check the data dimensions
        channel_counts = 0
        channel_dir: list[bool] = []
        channel_ls = list(header['DATA_INFO'].keys())
        for i in range(len(channel_ls)):
            if header['DATA_INFO'][channel_ls[i]]['Direction'] == 'both':
                channel_counts += 2
                channel_dir.append(True)  # true for fwd
                channel_dir.append(False)  # false for bwd
            elif header['DATA_INFO'][
                    channel_ls[i]]['Direction'] == 'fwd':  # forward only
                channel_counts += 1
                channel_dir.append(True)
            else:  # backward only
                channel_counts += 1
                channel_dir.append(False)
        dim = (int(channel_counts), int(header['Scan>pixels/line']),
               int(header['Scan>lines']))
        data = data.reshape(dim)  # data reshaped
        if header['Scan>Scanfield'][4] % 90 == 0:  # mutiple 90˚, rotate
            for i in range(dim[0]):
                if channel_dir[i]:
                    data[i] = np.rot90(data[i],
                                       int(header['Scan>Scanfield'][4] // 90))
                else:
                    data[i] = np.fliplr(
                        np.rot90(data[i],
                                 int(header['Scan>Scanfield'][4] // 90)))
        else:  # arbitary angle, no rotation
            for i in range(dim[0]):
                if channel_dir[i]:
                    continue
                else:
                    data[i] = np.fliplr(data[i])
        return data, tuple(channel_dir)


# TODO: Getting .sxm data & rotation
# TODO: spectrum .dat class
# TODO: grid spectrum .3ds class

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