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
        self.timejump = []
        i = 0
        for val in self.time_diff:
            if val <= 0:
                self.timejump.append(i)
            i += 1
        
    def split_at_timejumps(self):
        self.datalist = []
        self.datalist.append(self.data.iloc[0:timejump[0],:])
        for i in range(len(timejump)-1):
            split_of_data = self.data.iloc[timejump[i]:timejump[i+1],:]
            self.datalist.append(split_of_data)


dataobj = Dataexplorer(data)
dataobj.timegraph()
dataobj.find_timejumps()
timejump = dataobj.timejump
dataobj.split_at_timejumps()
