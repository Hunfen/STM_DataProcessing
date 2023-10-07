# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""
__all__ = ['loader']

from typing import Union, Tuple
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
        '.dat': lambda x: __Nanonis_dat__(x),
        '.3ds': lambda x: __Nanonis_3ds__(x)
    }
    try:
        return switch[os.path.splitext(f_path)[1]](f_path)
    except KeyError:
        print('File type not supported.')


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


class __Nanonis_sxm__:

    def __init__(self, f_path: str) -> None:
        """Nanonis .sxm file class.

        Args:
            f_path (str): Absolute path to the Nanonis .sxm file.

        Attributes:
            {
                file_path(str): Absolute path to the Nanonis .sxm file.
                fname(str): Name of the Nanonis .sxm file.
                header(dict): Nanonis .sxm file header.
                data(np.ndarray): .sxm data.
                channel_dir(tuple): Showing direction of every channel.
                                    True for forward, False for backward.
            }
        """
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__raw_header = self.__sxm_header_reader__(f_path)
        self.header = self.__sxm_header_reform__(self.__raw_header)
        self.data, self.channel_dir = self.__sxm_data_reader__(
            f_path, self.header)

    def __sxm_header_reader__(self, f_path: str) -> 'dict[str, str]':
        """Read the .sxm file header into dict.

        Returns:
            dict[str, str]: Header of the .sxm file,
                            including all the file attributes.
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
                # Load entries & corresponding parameters into pre-defined dict
                else:
                    contents += line
                    raw_header[entry] = contents.strip('\n')  # remove EOL
        return raw_header

    def __sxm_header_reform__(self, raw_header: 'dict[str, str]') -> dict:
        """Convert raw header into an accessible/readable dict.

        Returns:
            dict: Reformed header variable.
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
            # Table type header: information of feedback z controller
            elif i[1] == 'Z-CONTROLLER':
                z_controller: dict[str, Union[float, str]] = {}
                z_controller_config: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[0].strip('\t').split('\t')
                z_controller_configVar: list[str] = raw_header[
                    'Z-CONTROLLER'].split('\n')[1].strip('\t').split('\t')
                for j, s in enumerate(z_controller_config):
                    # FIXME: regex to split value str with units
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
        """Read the data of .sxm file.

        Args:
            f_path (str): Absolute path to the Nanonis .sxm file.
            header (dict): Reformed header variable.

        Returns:
            Tuple[np.ndarray, tuple]: Formated data matrix and
            the direction for every channel.
        """

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


class __Nanonis_dat__:

    def __init__(self, f_path: str) -> None:
        """Nanonis .dat file class.

        Args:
            f_path (str): Absolute path to the Nanonis .dat file.
        """
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        # self.raw_header = self.__dat_header_reader__(f_path)
        self.header = self.__dat_header_reformer__(
            self.__dat_header_reader__(f_path))
        self.data = self.__dat_data_reader__(f_path)

    def __dat_header_reader__(self, f_path: str) -> 'list[str]':
        """Read the .dat file header into list of strings.

        Returns:
            str: Header of the .dat file, including all the file attributes.
        """
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

    def __dat_header_reformer__(
            self, raw_header: 'list[str]') -> 'dict[str, Union[str, float]]':
        """Convert raw header into an accessible/readable dict.

        Args:
            raw_header (list[str]): .

        Returns:
            dict[str, str | float]: Reformed header variable.
        """
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

    def __dat_data_reader__(self, f_path: str) -> np.ndarray:
        """[summary]

        Args:
            f_path (str): Absolute path to the Nanonis .dat file.

        Returns:
            np.ndarray: data of Nanonis .dat file.
        """
        data_str: Union[str, list[str]] = ''
        data_list = []
        with open(f_path, 'r') as f:  # search data start positon
            while True:
                if re.match(r'\[DATA\]', f.readline()):
                    f.readline()
                    data_str = f.read()
                    break
                else:
                    continue
        data_str = data_str.split('\n')
        data_str = data_str[:-1]
        for i in range(len(data_str)):
            data_list.append(data_str[i].split('\t'))
        data_str = []  # release
        return np.array(data_list).astype(float)


class __Nanonis_3ds__:

    def __init__(self, f_path: str) -> None:
        """Nanonis .3ds file class.

        Args:
            f_path (str): Absolute path to the Nanonis .3ds file.
        """
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        # self.raw_header = self.__3ds_header_reader__(f_path)
        self.header = self.__3ds_header_reformer__(
            self.__3ds_header_reader__(f_path))
        self.Parameters, self.data = self.__3ds_data_reader__(
            f_path, self.header)

    def __3ds_header_reader__(self, f_path: str) -> 'dict[str, str]':
        """Read the .3ds file header into dict.

        Returns:
            dict[str, str]: Header of the .3ds file, including
            all the file attributes.
        """
        entry: str = ''
        contents: str = ''
        raw_header: dict[str, str] = {}
        with open(f_path, 'rb') as f:
            header_end = False
            while not header_end:
                line = f.readline().decode(encoding='utf-8', errors='replace')
                if re.match(':HEADER_END:', line):
                    header_end = True
                else:
                    entry, contents = line.split('=')
                    contents = contents.strip('"\r\n')
                raw_header[entry] = contents
        return raw_header

    def __3ds_header_reformer__(self, raw_header: 'dict[str, str]') -> dict:
        """Convert raw header into an accessible/readable dict.

        Returns:
            dict: Reformed header.
        """
        scan_info_tuple = ['Grid dim', 'Grid settings']
        scan_info_parameters = [
            'Experiment parameters', 'Channels', 'Fixed parameters'
        ]
        entries = list(raw_header.keys())
        header = {}
        for i in enumerate(entries):
            if i[1] in scan_info_tuple:
                if re.fullmatch(r'\d+\s[x]\s\d+', raw_header[i[1]]):
                    header[i[1]] = tuple(
                        int(j) for j in raw_header[i[1]].split(' x '))
                else:
                    header[i[1]] = tuple(
                        float(j) for j in raw_header[i[1]].split(';'))
            elif i[1] in scan_info_parameters:
                header[i[1]] = raw_header[i[1]].split(';')
            else:
                header[i[1]] = __is_number__(raw_header[i[1]])
        # FIXME: Defining type hints for header raises erorrs.
        header['# Parameters shape'] = tuple([
            header['Grid dim'][0] * header['Grid dim'][1],
            header['# Parameters (4 byte)']
        ])  # shape of the .Parameters
        header['data shape'] = tuple([
            header['Grid dim'][0] * header['Grid dim'][1],
            len(header['Channels']), header['Points']
        ])  # shape of the .data
        return header

    def __3ds_data_reader__(self, f_path: str,
                            header: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Read the data of .3ds file.

        Args:
            f_path (str): Absolute path to the Nanonis .3ds file.
            header (dict): Reformed header variable.

        Returns:
            Tuple[np.ndarray, np.ndarray]: spec attributes and the data matrix.
        """
        with open(f_path, 'rb') as f:
            read_all = f.read()
            offset = read_all.find(
                ':HEADER_END:\x0d\x0a'.encode(encoding='utf-8'))
            f.seek(offset + 14)
            data = np.fromfile(f, dtype='>f')
        Parameters = np.zeros(header['# Parameters shape'])
        spec_data = np.zeros((header['Grid dim'][0] * header['Grid dim'][1],
                              len(header['Channels']), header['Points']))
        data_size = header['Grid dim'][0] * header['Grid dim'][1] * (
            header['# Parameters (4 byte)'] +
            header['Experiment size (bytes)'] / 4)
        if not data.size == data_size:
            # If .3ds file is not integrated, dataset would be filled with 0.
            data = np.pad(data, (0, int(data_size - data.size)),
                          'constant',
                          constant_values=0)
        for i in range(header['Grid dim'][0] * header['Grid dim'][1]):
            # Read Parameters
            for j in range(header['# Parameters (4 byte)']):
                Parameters[i][j] = data[
                    i * int(header['# Parameters (4 byte)'] +
                            header['Experiment size (bytes)'] / 4) + j]
            # Read spec data
            for j in range(len(header['Channels'])):
                for k in range(header['Points']):
                    spec_data[i][j][k] = data[
                        int(i * (header['Experiment size (bytes)'] / 4 +
                                 header['# Parameters (4 byte)']) +
                            (j * header['Points'] +
                             header['# Parameters (4 byte)']) + k)]
        return Parameters, spec_data
