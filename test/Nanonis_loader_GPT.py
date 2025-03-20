# -*- coding: utf-8 -*-
"""
Python module that helps read the Nanonis files.
"""
__all__ = ['is_number']

import os
import re
from typing import Union, Tuple, Dict
import numpy as np


def is_number(s):
    """Convert string to float if possible, otherwise return the original string."""
    try:
        return float(s)
    except ValueError:
        return s


class NanonisFileLoader:
    def __init__(self, f_path: str) -> None:
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.header = {}
        self.data = None
        self.channel_dir = []

        if f_path.endswith('.sxm'):
            self.file_type = 'sxm'
            self.header = self.__reform_sxm_header__(self.__read_sxm_header__(f_path))
            self.data, self.channel_dir = self.__read_sxm_data__(f_path)
        elif f_path.endswith('.dat'):
            self.file_type = 'dat'
            self.header = self.__reform_dat_header__(self.__read_dat_header__(f_path))
            self.data = self.__read_dat_data__(f_path)
        elif f_path.endswith('.3ds'):
            self.file_type = '3ds'
            self.header = self.__reform_3ds_header__(self.__read_3ds_header__(f_path))
            self.Parameters, self.data = self.__read_3ds_data__(f_path, self.header)
        else:
            raise ValueError(f"Unsupported file type: {os.path.splitext(f_path)[1]}")

    def __read_sxm_header__(self, f_path: str) -> dict[str, str]:
        raw_header: dict[str, str] = {}
        with open(f_path, 'rb') as f:
            for line in f:
                decoded_line = line.decode(encoding='utf-8', errors='replace').strip()
                if re.match(':SCANIT_END:', decoded_line):
                    break
                elif re.match(':.+:', decoded_line):
                    entry = decoded_line[1:-2]
                    contents:str = ''
                else:
                    contents += decoded_line
                    raw_header[entry] = contents.strip('\n')
        return raw_header

    def __reform_sxm_header__(self, raw_header: dict[str, str]) -> dict:
        scan_info_keys = [
            'ACQ_TIME', 'BIAS', 'Scan>lines', 'Scan>pixels/line',
            'Scan>speed backw. (m/s)', 'Scan>speed forw. (m/s)', 'COMMENT',
            'REC_DATE', 'REC_TIME', 'SCAN_DIR', 'SCAN_FILE'
        ]

        header = {}
        entries = list(raw_header.keys())

        for entry in entries:
            value = raw_header[entry].strip(' ')
            if entry in scan_info_keys:
                header[entry] = is_number(value)
            elif entry == 'Z-CONTROLLER':
                z_controller_config, z_controller_configVar = raw_header['Z-CONTROLLER'].split('\n')
                z_controller_config = z_controller_config.strip('\t').split('\t')
                z_controller_configVar = z_controller_configVar.strip('\t').split('\t')
                z_controller = {key: is_number(val) for key, val in zip(z_controller_config, z_controller_configVar)}
                header[entry] = z_controller
            elif entry == 'DATA_INFO':
                data_info = {}
                raw_data_info_lines = raw_header['DATA_INFO'].strip().split('\n')
                config_keys = raw_data_info_lines[0].strip('\t').split('\t')
                for line in raw_data_info_lines[1:]:
                    channel_info = {key: is_number(val) for key, val in zip(config_keys, line.strip('\t').split('\t'))}
                    data_info[channel_info['Name']] = channel_info
                    del data_info[channel_info['Name']]['Name']
                header[entry] = data_info
            elif entry == 'Scan>Scanfield':
                header[entry] = tuple(map(float, raw_header['Scan>Scanfield'].split(';')))
        return header

    def __read_sxm_data__(self, f_path: str) -> Tuple[np.ndarray, list]:
        with open(f_path, 'rb') as f:
            read_all = f.read()
            offset = read_all.find(b'\x1A\x04')
            f.seek(offset + 2)
            data = np.fromfile(f, dtype='>f')

        channel_counts = 0
        channel_dir: list[bool] = []
        for info in self.header['DATA_INFO'].values():
            if info['Direction'] == 'both':
                channel_counts += 2
                channel_dir.extend([True, False])
            else:
                channel_counts += 1
                channel_dir.append(info['Direction'] == 'fwd')

        dim = (channel_counts, int(self.header['Scan>pixels/line']), int(self.header['Scan>lines']))
        data = data.reshape(dim)

        if self.header['Scan>Scanfield'][4] % 90 == 0:
            for i in range(dim[0]):
                rotations = int(self.header['Scan>Scanfield'][4] // 90)
                data[i] = np.rot90(data[i], k=rotations) if channel_dir[i] else np.fliplr(np.rot90(data[i], k=rotations))
        else:
            for i in range(dim[0]):
                if not channel_dir[i]:
                    data[i] = np.fliplr(data[i])
        return data, channel_dir

    def __read_dat_header__(self, f_path: str) -> list[str]:
        raw_header: list[str] = []
        with open(f_path, 'r',encoding='utf-8') as f:
            for line in f:
                if re.match(r'\[DATA\]', line):
                    break
                raw_header.append(line.strip('\n').strip('\t'))
        return raw_header

    def __reform_dat_header__(self, raw_header: list[str]) -> dict[str, Union[str, float]]:
        header = {}
        for line in raw_header:
            try:
                key, value = line.split('\t')
                header[key] = is_number(value)
            except ValueError:
                header[line] = ''
        return header

    def __read_dat_data__(self, f_path: str) -> np.ndarray:
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                if re.match(r'\[DATA\]', line):
                    break
            data_str = f.read().strip()
        data_list = [list(map(is_number, line.split('\t'))) for line in data_str.strip().split('\n')]
        return np.array(data_list)

    def __read_3ds_header__(self, f_path: str) -> dict[str, str]:
        raw_header: dict[str, str] = {}
        with open(f_path, 'rb') as f:
            for line in iter(lambda: f.readline().decode('utf-8', errors='replace').strip(), ':HEADER_END:'):
                if '=' not in line:
                    continue
                entry, contents = line.split('=')
                raw_header[entry] = contents.rstrip('\r\n')
        return raw_header

    def __reform_3ds_header__(self, raw_header: dict[str, str]) -> dict:
        def convert_type(s):
            if re.fullmatch(r'\d+\s[x]\s\d+', s):
                return tuple(map(int, s.split(' x ')))
            elif ';' in s:
                return tuple(map(float, s.split(';')))
            else:
                return is_number(s)

        header = {k: convert_type(v) for k, v in raw_header.items()}
        header['# Parameters shape'] = (header['Grid dim'][0] * header['Grid dim'][1], header['# Parameters (4 byte)'])
        header['data shape'] = (header['Grid dim'][0] * header['Grid dim'][1], len(header['Channels']), header['Points'])
        return header

    def __read_3ds_data__(self, f_path: str, header: dict) -> Tuple[np.ndarray, np.ndarray]:
        with open(f_path, 'rb') as f:
            start = f.read().find(':HEADER_END:\x0d\x0a'.encode('utf-8')) + 14
            f.seek(start)
            data = np.fromfile(f, dtype='>f')

        expected_data_size = header['Grid dim'][0] * header['Grid dim'][1] * 
                           (header['# Parameters (4 byte)'] +
                            header['Experiment size (bytes)'] / 4)
        if len(data) < expected_data_size:
            data = np.pad(data, (0, int(expected_data_size - len(data))), constant_values=0)

        parameters_shape = (header['Grid dim'][0] * header['Grid dim'][1], header['# Parameters shape'][1])
        Parameters = data[:parameters_shape[0] * parameters_shape[1]].reshape(parameters_shape)
        spec_data = data[parameters_shape[0] * parameters_shape[1]:].reshape(header['data shape'])

        return Parameters, spec_data


def get_header(self) -> dict:
    """Get the header of the file."""
    return self.header

def get_data(self) -> np.ndarray:
    """Get the data of the file."""
    if self.file_type == '3ds':
        return self.data
    else:
        return self.data

def get_channel_directions(self) -> list[bool]:
    """Get the channel directions for .sxm files."""
    if self.file_type == 'sxm':
        return self.channel_dir
    else:
        raise ValueError("Channel directions are only available for .sxm files.")
    