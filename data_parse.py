# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:16:15 2021

@author: M.N.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataParse:
    """
    Class that read csv files from folder, then parse files into one 
    Pandas dataframe
    """
    
    def __init__(self):
        pass
        
    def sort_folder(self, directory: str, start: int = 33, end: int = -4) -> list:
        """
        Sort files in directory based on number label
        
        Parameters
        ----------
        directory: str
            path of directory.
        """
        files = os.listdir(directory)
        l = [[int(file[start:end]), file] for file in files]
        l.sort()
        return [file for ele, file in l]
        
        
    def read_folder(self, directory: str, data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        """
        Read folder of csv files as formatted by Ben.
        
        Parameters
        ----------
        directory: str
            path of directory.
        data: pd.DataFrame
            initial data frame. Default empty.
        """
        files = self.sort_folder(directory)
        for file in files:
            if file.endswith('.csv'):
                f = open(directory + file, 'r')
                if data.empty:
                    data = pd.read_csv(f)
                else:
                    data = data.append( pd.read_csv(f), ignore_index=True )
                f.close()
        return data
    
    def filter_trade_hour(self, data: pd.DataFrame, start_time: int = 90000, 
                          end_time: int = 153000) -> pd.DataFrame:
        """
        Filter the dataframe when the trade happens.

        Parameters
        ----------
        start_time : Union[int, None], optional
            The starting time of the search in HHMMSS format. None means from the beginning of the file. The default is
            None. Inclusive.
        end_time : Union[int, None], optional
            The ending time of the search in HHMMSS format. None means from the beginning of the file. The default is
            None. Inclusive.

        Returns: pd.DataFrame where trades happen.
        -------
        """
        return data[(data['Time'] >= start_time) & 
             (data['Time'] <= end_time) & 
             (20000000 + data['EntryDate'] == data['Date'])] # Beware 20th century data