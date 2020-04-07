# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:34:57 2020

@author: hartmann
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\clean_data').iloc[:,1:]



class Dataexplorer:
    def __init__(self, data):
        self.data = data
        self.time = data.iloc[:, 0]
        self.time_diff = np.diff(self.time)
        self.freq = 1/np.mean(self.time_diff)
        
    def timegraph(self):
        plt.plot(self.time)
        
    def find_timejumps(self):
        timejump = []
        i = 0
        for val in self.time_diff:
            if val <= 0:
                timejump.append(i)
            i += 1
        return timejump
        
    def data_plotter(self, sensorplots=False, timeplot=False, label=False, sensorpdf=False):
        if timeplot:
            timejump = self.find_timejumps()
            plt.figure(0)
            for xc in timejump:
                plt.axvline(xc, color = 'red')
            self.timegraph()
        if sensorplots:
            sensorchannel = 0
            for fig in range (1,17,1):
                if sensorchannel == 242:
                            break
                plt.figure(fig)
                for subplot in range(5):
                    plt.subplot(5,1,subplot+1)
                    for plot in range(3):
                        plt.plot(self.data.iloc[:,sensorchannel])
                        for xc in timejump:
                            plt.axvline(xc, color = 'red')
                        sensorchannel += 1
                        
        if sensorpdf:
            sensorchannel = 0
            for fig in range (100,115,1):
                if sensorchannel == 242:
                        break
                plt.figure(fig)
                for subplot in range(16):
                    plt.subplot(4,4,subplot+1)
                    plt.hist(self.data.iloc[:,sensorchannel], bins=20)
                    sensorchannel += 1
    
    def label_plotter(self):
        plt.figure(200)
        plt.plot(self.data.iloc[:,-1])
        unique_vals = self.data.iloc[:,-1].value_counts(normalize=True)
        plt.figure(201)
        plt.bar([i for i in range(len(unique_vals))], unique_vals)
                    

dataobj = Dataexplorer(data)
dataobj.label_plotter()

