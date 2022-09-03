'''
||  evisualiser: an event-based to frame-based visual data converter  ||

Event-based neuromorphic visual systems are an alternative to traditional
frame-based visual data capture and processing systems. They offer a sparse and
asynchronous data capture and representation system, which is more temporally
precise (µm level precision) compared to most frame-based image capture systems.

Instead of capturing frames, the camera captures "events", which are pixel-level
changes in brightness above a certain threshold. In the address event
representation (AER) format, each event is defined by four parameters:

       [x-coordinate, y-coordinate, timestamp, polarity]

where polarity is a binary value indicating whether there has been a positive or
negative change in brightness.

The AEDAT (extension: .aedat) is a format often used to store event-based data.
This module helps convert event-based data stored in the AEDAT format to a
frame-based MPEG video. This can be used for many purposes such as, visualisation
of the data, training of machine learning models on the data, etc.

The module allows the user to enter a desired frame rate that will be used to
create the frames and play the MPEG video. For the required frame rate, a
corresponding time window will be calculated and events will be grouped into
time windows. Each frame will be composed of the events contained in a single
time window, with events of a positive polarity marked green and those with a
negative polarity marked red.

The final output will be saved in a .mp4 file with a user defined filename.
'''


import cv2 as cv
import tempfile
import time
import os
import shutil
import numpy as np
import pandas as pd
from aermanager.aerparser import load_events_from_file
from aermanager.parsers import parse_dvs_ibm
from typing import List

class Sequence:
    '''
    This class encapsulates all variables and methods involving the conversion
    of a visual sequence in the event-based format to frame-based video.
    '''
    def __init__(self) -> None:
        self.shape = None           # shape of data (horizontal pixels, vertical pixels)
        self.df_events = None       # pandas dataframe containing events in address-event representation ['x', 'y', 't', 'p']
        self.n_events = None        # total number of events in the sequence
        self.start_time_logs = 0    # start_time for logs

    def event_file_to_video(self, input_path:str, output_path:str, fps:int=25) -> None:
        '''
        Generates an MPEG video from events stored in an input path.

        Parameters:
            input_path (str): path containing the AEDAT file containing events.
            output_path (str): path where the MPEG output file should be saved.
            fps (int): frame rate (fps) of the MPEG output video.
        Returns: None.
        '''

        frame_width_us = 1000000//fps
        print("Loading events from input file ...", end="\r")
        self.load_df_from_file(input_path)
        self.generate_all_frames(output_path, fps=fps)

    def load_df_from_file(self, path:str) -> None:
        '''
        Loads AEDAT file from the path provided in the input onto a pandas
        dataframe (self.df_events).

        Parameters:
            path(str): path containing AEDAT file.
        Returns: None.
        '''
        self.shape, events = load_events_from_file(path, parser = parse_dvs_ibm)
        self.df_events = pd.DataFrame(data=events, columns={'x', 'y', 't', 'p'})      # timestamp in µs
        self.df_events = self.df_events[['x' ,'y', 't', 'p']]
        self.n_events = len(self.df_events.index)

    def generate_all_frames(self, output_path:str, fps:int) -> None:
        '''
        Divides all events into windows according to the required frame rate
        (fps) and writes them as PNG files in a temporary directory.

        Then uses the create_video method to create an MPEG video from the PNG
        files, and finally removes the temporary directory along with its
        content.

        Parameters:
            output_path (str): filename or path to save the MPEG video in.
            fps (int): frame rate (frames per second) for the MPEG video.
        Returns: None.
        '''

        frame_width_us = 1000000//fps     # timestamps are in microseconds
        temp_dir = tempfile.mkdtemp()

        start_time = self.df_events.iloc[0].t
        frame_number = 1
        current_event = 0

        start_time_logs = time.time()    # for showing progress

        while current_event < self.n_events:
            print('Generating frame ', frame_number, '. Progress: ', '{:.2f}'.format((current_event+1)*100/self.n_events), ' %', end='\r')
            end_time = start_time + frame_width_us    # the end time for this time-window for frame generation

            frame_start_index = current_event
            frame_end_index = frame_start_index

            # finding the index of the final event that can be allowed in this frame:
            while current_event < self.n_events and self.df_events.iloc[current_event].t < end_time:
                frame_end_index += 1
                current_event += 1

            # generating the frame:
            current_frame = self.create_frame(self.df_events.iloc[frame_start_index : frame_end_index][['x', 'y', 'p']].to_numpy())

            # saving it as a png file:
            path = temp_dir + "/frame_" + str(frame_number) + ".png"
            cv.imwrite(path, current_frame)

            start_time = end_time       # start time for the next frame
            frame_number += 1

        print(str(frame_number) + " frames generated in total.\n")
        self.create_video(output_path, fps=fps, frame_dir=temp_dir)
        shutil.rmtree(temp_dir)

    def create_frame(self, events_in:pd.DataFrame) -> np.ndarray:
        '''
        Creates a 2D frame from comprising of events that are within a given
        time-frame. A positive polarity (p==1) is marked green and a negative
        polarity (p==0) is marked red.

        Parameters:
            events_in(pd.DataFrame): Contains the events that should be used to
            create the new frame. Its columns are 'x', 'y' and 'p' in this order,
            representing the x-coordinate, the y-coordinate and the polarity of
            the event respectively.
        Returns:
            np.ndarray: A numpy array containing the new frame in BGR format.
        '''
        # note: can you make events_in a 2d array?

        new_frame = np.zeros([self.shape[0], self.shape[1], 3], np.uint8)

        for i in range(len(events_in)):                                                 # if p == 1: value=(0,255,0), if p==0: value=(0,0,255)
            new_frame[events_in[i][1]][events_in[i][0]][1] = events_in[i][2]*255           # frame[y][x][1] (green) = polarity*255
            new_frame[events_in[i][1]][events_in[i][0]][2] = (1-events_in[i][2])*255       # frame[y][x][1] (red) = (1-polarity)*255

        return new_frame

    def create_video(self, out:str, fps:int, frame_dir:str) -> None:
        '''
        Uses the ffmpeg package to create and save an MPEG video from frames
        saved in PNG format.

        Parameters:
            out (str): the path or filename to be used for the output file.
            fps (int): frame rate (frames per second) for the output file.
            frame_dir (str): the directory where the frames are stored.
        Returns: None.
        '''

        out = os.path.join(os.getcwd(), out)
        command = "ffmpeg -i {}/frame_%d.png -r {} {}".format(frame_dir, str(int(fps)), out)
        os.system(command)

    def get_eta(self, current_event:int) -> str:
        '''
        Returns an ETA (time remaining) using the number of events that have been
        processed and the total number of events, assuming that each event takes
        the same amount of time to be processed.

        Parameters:
            current_event (int): a number given to the current event that is
                being processed
        Returns:
            str: a string in H:M:S format showing ETA
        '''
        if current_event == 0:
            return "-"
        eta = (time.time() - self.start_time_logs)*(self.n_events - current_event) / current_event
        return time.strftime('%H:%M:%S', time.gmtime(eta))
