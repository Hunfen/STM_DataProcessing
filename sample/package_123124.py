# -*- coding: utf-8 -*-
"""
 
Created on 05.06.2012
 
@author: Felix Wählisch
 
Description:
============
This Script reads in .sxm files for the Version 2. It still does not care for 
up/down scan directions or something alike. I added some non-core functions as
example how to use the module 
        (   see:     if __name__=="__main__":    at end of script). 
 
To run the script I recommend PythonXY - the scientific oriented python enviroment
http://code.google.com/p/pythonxy/
it includes numpy, pylab, scipy - all you will need.
        
The module is not fully functional, in doublt doublecheck with gwyddion.
This is NOT intended to be a full solution but save you quite some time 
implementing the binary read-in. I hope it does exactly this for you.
By now, this script can for sure NOT handle correctly:
    * data that only were read in in one direction
    * up/down or down/up - I did not care for now
    * incomplete images
    * ...
    
See the class descriptions for more infos:
- NanonisAFM:
    handles the data in self.signals and the header in self.infos
    signals is an array of class Datachannel
    infos is a dictionary
- Datachannel
    data and informations concerning one channel
- Fit
    2D fit functions to flatten your AFM data - used by
    NanonisAFM.substract2Dfit()
    1D fit function to flatten your AFM data - used by
    NanonisAFM.substract1Dfit()
    PLEASE NOTE THAT THE DATA MANIPULATION IS DONE ON ALL DATACHANNELS FOR NOW!
    Please also note, that the manipulation is done IN PLACE (=replace old data)
 
Disclaimer:
===========
I am not responsible to any havoc, malevolence or inconvenience that happens to
you or your data while using this script (:
    
License:
========
"THE BEER-WARE LICENSE":
As long as you retain this notice you can do whatever you want with this stuff. 
If we meet some day, and you think this stuff is worth it, you can buy me 
a beer in return.
    - Felix Wählisch
    
"""


import numpy as N
import pylab as P
import os
import re
import string
import struct
from scipy import optimize


class NanonisAFM:
    ''' data contains:
        fname           filename
        infos           dictionary with all header entries
        datachannels    array of the datachannels in datachannel class format
    '''

    def __init__(self, fname=''):
        ''' creates empty class and asks for file to process if not given '''
        self.signals = []
        self.infos = {}
        self.fname = ''
        if fname == '':
            # your custom ask for file
            print "init(): please program me!"
        elif fname == '':
            print "selection aborted"
            return
        else:
            self.fname = fname
            print "file: ", os.path.split(self.fname)[1]
            self.readin()

    def _checkfile(self):
        '''inherit with some reality-checks that verify a good filename. It is recommended to run this after selecting self.fname'''
        if self.fname.endswith('.sxm'):
            return True
        else:
            print "wrong file ending (not .sxm)", self.fname
            return False

    def readin(self):
        if not self._checkfile():
            return -1
        self._read_header()
        self._read_body()

    def _read_body(self):
        '''The binary data begins after the header and is introduced by the (hex) code \1A\04. 
        According to SCANIT_TYPE the data is encoded in 4 byte big endian floats. 
        The channels are stored one after the other, forward scan followed by backward scan.
        The data is stored chronologically as it is recorded. 
        On an up-scan, the first point corresponds to the lower left corner 
        of the scanfield (forward scan). On a down-scan, it is the upper 
        left corner of the scanfield. 
        Hence, backward scan data start on the right side of the scanfield.'''
        # extract channes to be read in
        data_info = self.infos['DATA_INFO']
        lines = string.split(data_info, '\n')
        # headers: Channel    Name      Unit        Direction                Calibration               Offset
        lines.pop(0)
        names = []
        units = []
        for line in lines:
            entries = string.split(line)
            if len(entries) > 1:
                names.append(entries[1])
                units.append(entries[2])
                if entries[3] != 'both':
                    print "warning, only one direction recorded, expect a crash :D", entries
        print names
        # extract lines, pixels
        # xPixels = int(self.infos['Scan>pixels/line'])
        # yPixels = int(self.infos['Scan>lines'])
        xPixels, yPixels = string.split(self.infos['SCAN_PIXELS'])
        xPixels = int(xPixels)
        yPixels = int(yPixels)
        # find position in file
        fhandle = open(self.fname, 'rb')  # read binary
        read_all = fhandle.read()
        offset = read_all.find('\x1A\x04')
        print 'found start at', offset
        fhandle.seek(offset+2)  # data start 2 bytes afterwards
        # read in data
        fmt = '>f'  # float
        ItemSize = struct.calcsize(fmt)
        for i in range(len(names)*2):  # fwd+bwd
            if i % 2 == 0:
                direction = '_fwd'
            else:
                direction = '_bwd'
            bindata = fhandle.read(ItemSize*xPixels*yPixels)
            data = N.zeros(xPixels*yPixels)
            for j in range(xPixels*yPixels):
                data[j] = struct.unpack(
                    fmt, bindata[j*ItemSize: j*ItemSize+ItemSize])[0]
            data = data.reshape(yPixels, xPixels)
            data = N.rot90(data)
            if direction == '_bwd':
                data = data[::-1]
            channel = Datachannel(
                name=names[i/2]+direction, data=data, unit=units[i/2])
            print channel.name, channel.unit, channel.data.shape
            self.signals.append(channel)
        fhandle.close()

    def _read_header(self):
        ''' reads the header and adds to info dictionary - ready for further parsing as needed'''
        header_ended = False
        fhandle = open(self.fname, 'r')
        caption = re.compile(':*:')
        key = ''
        contents = ''
        while not header_ended:
            line = fhandle.readline()
            if line == ":SCANIT_END:\n":  # check for end of header
                header_ended = True
                self.infos[key] = contents
                # two blank lines
                fhandle.readline()
                fhandle.readline()
            else:
                if caption.match(line) != None:  # if it is a caption
                    if key != '':  # avoid 1st run problems
                        self.infos[key] = contents
                    key = line[1:-2]  # set new name
                    contents = ''  # reset contents
                else:  # if not caption, it is content
                    contents += (line)
        fhandle.close()

    def create_img(self, nametag, clim=(None, None)):
        '''puts out images of signals whose name contains the nametag.
        adjust your color bar by using clim(lower, upper)'''
        x_len, y_len = string.split(self.infos['SCAN_RANGE'])
        x_len = float(x_len)
        y_len = float(y_len)
        # if you change to nm, also change the labels further below (;
        x_len *= 1.0e6  # um
        y_len *= 1.0e6  # um
        for i in self.signals:
            if nametag in i.name:
                print "create_img(): creating", i.name, "image..."
                P.figure()
                ax = P.subplot(111)
                z = N.fliplr(N.rot90(i.data, k=1))
                if i.unit != 'V':
                    z = z*1.0e9
                    i.unit = 'n'+i.unit
                P.imshow(z, origin="lower", cmap=P.cm.YlOrBr_r, aspect='auto')
                # ticker adjustment
                (yi, xi) = z.shape
                # get old x-labels, create new ones
                x_ticks = N.int_(
                    N.round(N.linspace(0,  xi-1, len(ax.axes.get_xticklabels()))))
                x_tlabels = N.round(N.linspace(0, x_len, len(
                    ax.axes.get_xticklabels())), decimals=2)
                ax.axes.set_xticks(x_ticks)
                ax.axes.set_xticklabels(x_tlabels)
                P.xlim(0, xi-1)  # plots from 0 to p-1, so only show that
                P.xlabel("X [um]")
                # get old y-labels, create new ones, note reverse axis ticks
                y_ticks = N.int_(
                    N.round(N.linspace(0,  yi-1, len(ax.axes.get_yticklabels()))))
                y_tlabels = N.round(N.linspace(0, y_len,  len(
                    ax.axes.get_yticklabels())), decimals=2)
                ax.axes.set_yticks(y_ticks)
                ax.axes.set_yticklabels(y_tlabels)
                P.ylim(0, yi-1)  # plots from 0 to p-1, so only show that
                P.ylabel("Y [um]")
                if clim != (None, None):
                    P.clim(clim[0], clim[1])
                bar = P.colorbar(shrink=0.7)
                bar.set_label(i.name+' ['+i.unit+']')
                P.title(os.path.split(self.fname)[1][:-4])
                P.draw()
                P.savefig(self.fname[:-4]+'_'+i.name+'.png', transparent=True)
                P.close()

    def substract_1Dfit(self, deg=1):
        '''substracts a line by line fit from all data
        degree = 0 offset fit        
        degree = 1 linear fit
        degree = 2 ...
        '''
        Fit = fit()
        for i in self.signals:
            data = i.data
            res = Fit.poly_line_by_line(data, deg, axis=1)
            i.data = data-res
        return 0

    def substract_2Dfit(self, deg=2):
        '''substracts something like a 2D fit from all data
        degree: 1- plane substract
                2- parabolic substract'''
        if deg not in range(1, 3, 1):  # goes to n-1
            print "substract_2Dfit(): unknown degree of fit, abort..."
            return -1
        Fit = fit()
        for i in self.signals:
            data = i.data
            # fit parameters initial values
            if deg == 1:
                params = Fit.fitplane(data)
                fit_func = Fit.return_plane(params, data)
            if deg == 2:
                params = Fit.fitparabolic(data)
                fit_func = Fit.return_parabolic(params, data)
            i.data = data-fit_func
        return 0

    def friction_signal(self, ignore_large_img=True):
        ''' calculates Horiz._Deflection fwd - bwd / 2.0 and appends the Friction channel to the signals.
        Returns Friction.data. Ignores large images by default due to hysteresis.
        '''
        fwd = None
        bwd = None
        for i in self.signals:
            if i.name == "Horiz._Deflection_fwd":
                fwd = i.data
            if i.name == "Horiz._Deflection_bwd":
                bwd = i.data
                unit = i.unit  # unit of both channels is supposed to be same
            if i.name == 'Friction':
                print "friction_signal(): Friction channel already exists, aborting..."
                return -1
        if fwd == None or bwd == None:
            print "friction_signal(): could not find all signals needed, aborted."
            return -1
        # ignore large images due to hysteresis
        x_len, y_len = string.split(self.infos['SCAN_RANGE'])
        x_len = float(x_len)
        y_len = float(y_len)
        x_len *= 1.0e6  # um
        y_len *= 1.0e6  # um
        if x_len > 30.0 or y_len > 30.0:
            if not ignore_large_img:
                print "friction_signal(): warning, the friction signal might be shadowed due to large scan range and hysteresis!"
            else:
                print "friction_signal(): friction signal is not created due to image size"
                return -1
        print "friction_signal(): creating Friction channel."
        frict = Datachannel(data=(fwd-bwd)/2.0, name='Friction',
                            desc='horiz. deflection (fwd-bwd)/2', unit=unit)
        self.signals.append(frict)
        return frict.data


class Datachannel:
    '''data and their description...'''
    unit = ""  # unit of the channel
    name = ""  # name of the channel
    desc = ""  # description / additional info
    data = N.array([])  # data in the channel

    def __init__(self, unit="", name="", desc="", data=""):
        self.unit = str(unit)
        self.name = str(name)
        self.desc = str(desc)
        self.data = N.array(data)


class fit:
    """ 2D Fit Functions """

    def __init__(self):
        return

    ####################################
    """ taken from the Scipy Cookbook - gauss is not used from Nanonis AFM"""

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height*N.exp(
            -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def gaussmoments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = N.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = N.sqrt(abs((N.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = N.sqrt(abs((N.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.gaussmoments(data)
        def errorfunction(p): return N.ravel(self.gaussian(*p)(*N.indices(data.shape)) -
                                             data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    ####################################

    def parabolic(self, a0, a1, a2, b1, b2, x0, y0):
        '''could also do slope and plain - wow! used by substract_2Dfit'''
        return lambda x, y: a0 + a1*(x-x0) + a2*(x-x0)**2 + b1*(y-y0) + b2*(y-y0)**2

    def parabolicmoments(self, data):
        '''to be filled...'''
        a0 = abs(data).min()
        index = (data-a0).argmin()
        x, y = data.shape
        x0 = float(index / x)
        y0 = float(index % y)
        a1 = 0.0
        a2 = 0.0
        b1 = 0.0
        b2 = 0.0
        return a0, a1, a2, b1, b2, x0, y0

    def fitparabolic(self, data):
        params = self.parabolicmoments(data)
        def errorfunction(p): return N.ravel(
            self.parabolic(*p)(*N.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def return_parabolic(self, params, data):
        ''' returns an 2D array of the parabolic fit with the shape of data'''
        fit_data = self.parabolic(*params)
        return fit_data(*N.indices(data.shape))
    ####################################

    def plane(self, a0, a1, b1, x0, y0):
        return lambda x, y: a0 + a1*(x-x0) + b1*(y-y0)

    def planemoments(self, data):
        a0 = N.abs(data).min()
        index = (data-a0).argmin()
        x, y = data.shape
        x0 = float(index / x)
        y0 = float(index % y)
        a1 = 0.0
        b1 = 0.0
        return a0, a1, b1, x0, y0

    def fitplane(self, data):
        params = self.planemoments(data)
        def errorfunction(p): return N.ravel(
            self.plane(*p)(*N.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def return_plane(self, params, data):
        fit_data = self.plane(*params)
        return fit_data(*N.indices(data.shape))

    ####################################
    def poly_line_by_line(self, data, deg=1, axis=0):
        '''takes data, degree for polynomial line-by-line fitting, 
        axis to fit along
        returns fitted surface'''
        if axis == 1:  # turn data around
            data = N.rot90(data)

        surface = N.zeros(data.shape)
        x = range(data.shape[1])
        for i in range(len(data)):
            p = N.polyfit(x, data[i], deg)
            surface[i] = N.polyval(p, x)

        if axis == 1:  # turn results back around
            surface = N.rot90(surface, k=3)
        return surface
