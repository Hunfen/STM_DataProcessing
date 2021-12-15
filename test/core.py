# -*- coding: utf-8 -*-
import os
import re

import numpy as np



# @profile
# def mem_test():
    # f_path = '/Users/hunfen/Documents/Experimental/STM1500_data/2021/2021-12-10/Grid Spectroscopy012.3ds'
    # a = loader(f_path)
    # b = a.header

    # with open(f_path, 'rb') as f:
    #     read_all = f.read()
    #     offset = read_all.find(':HEADER_END:\x0d\x0a'.encode(encoding='utf-8'))
    #     f.seek(offset + 14)
    #     data = np.fromfile(f, dtype='>f')
    # del f
    # Parameters = np.zeros(b['# Parameters shape'])
    # spec_data = np.zeros(
    #     (b['Grid dim'][0] * b['Grid dim'][1], len(b['Channels']), b['Points']))
    # if data.size == b['Grid dim'][0] * b['Grid dim'][1] * (
    #         b['# Parameters (4 byte)'] + b['Experiment size (bytes)'] / 4):
    #     for i in range(b['Grid dim'][0] * b['Grid dim'][1]):
    #         # Read Parameters
    #         for j in range(b['# Parameters (4 byte)']):
    #             Parameters[i][j] = data[i *
    #                                     int(b['# Parameters (4 byte)'] +
    #                                         b['Experiment size (bytes)'] / 4) +
    #                                     j]
    #         # Read spec data
    #         for k in range(len(b['Channels'])):
    #             for l in range(b['Points']):
    #                 spec_data[i][k][l] = data[int(
    #                     i * (b['Experiment size (bytes)'] / 4 +
    #                          b['# Parameters (4 byte)']) +
    #                     (k * b['Points'] + b['# Parameters (4 byte)']) + l)]
    # del data


# if __name__ == "__main__":
#     mem_test()

# def get_hmm():
#     """Get a thought."""
#     return 'hmmm...'

# def hmm():
#     """Contemplation..."""
#     if helpers.get_answer():
#         print(get_hmm())
