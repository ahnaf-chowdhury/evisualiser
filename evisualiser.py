import cv2 as cv
import tempfile
import time
import os
import shutil
import numpy as np
import pandas as pd
from aermanager.aerparser import load_events_from_file
from aermanager.parsers import parse_dvs_ibm

class Sequence:
    def __init__(self) -> None:
        self.shape = None
        self.df_events = None
        self.n_events = None
        self.start_time_logs = 0    #start_time for logs

    def load_df_from_file(self, path) -> None:
        self.shape, events = load_events_from_file(path, parser = parse_dvs_ibm)

        self.df_events = pd.DataFrame(data=events, columns={'x', 'y', 't', 'p'})      # timestamp in µs
        self.df_events = self.df_events[['x' ,'y', 't', 'p']]
        self.n_events = len(self.df_events.index)
        del events

    def create_frame(self, frame_in) -> np.ndarray:
        # frame in should be a 2d numpy array with columns representing x, y, p
        new_frame = np.zeros([self.shape[0], self.shape[1], 3], np.uint8)

        for i in range(len(frame_in)):
            new_frame[frame_in[i][1]][frame_in[i][0]][1] = frame_in[i][2]*255           # frame[y][x][1] (green) = polarity*255
            new_frame[frame_in[i][1]][frame_in[i][0]][2] = (1-frame_in[i][2])*255       # frame[y][x][1] (red) = (1-polarity)*255

        return new_frame

    def generate_all_frames(self, frame_width, output_path, fps) -> None:
        temp_dir = tempfile.mkdtemp()

        start_time = self.df_events.iloc[0].t
        frame_number = 1
        current_event = 0

        start_time_logs = time.time()  # for logs (eta)

        while current_event < self.n_events:
            print('Generating frame ', frame_number, '. Progress: ', '{:.2f}'.format((current_event+1)*100/self.n_events), ' %', end='\r')
            end_time = start_time + frame_width

            frame_start_index = current_event
            frame_end_index = frame_start_index

            while current_event < self.n_events and self.df_events.iloc[current_event].t < end_time:
                frame_end_index += 1
                current_event += 1

            current_frame = self.create_frame(self.df_events.iloc[frame_start_index : frame_end_index][['x', 'y', 'p']].to_numpy())

            path = temp_dir + "/frame_" + str(frame_number) + ".png"
            cv.imwrite(path, current_frame)

            start_time = end_time       # for the next frame
            frame_number += 1

        print(str(frame_number) + " frames generated in total.\n")
        self.create_video(output_path, fps=fps, frame_dir=temp_dir)
        shutil.rmtree(temp_dir)

    def create_video(self, out, fps, frame_dir):
        out = os.path.join(os.getcwd(), out)
        command = "ffmpeg -i {}/frame_%d.png -r {} {}".format(frame_dir, str(int(fps)), out)
        os.system(command)

    def event_file_to_video(self, input_path, output_path, fps=25):
        frame_width_us = 1000000//fps
        print("Loading events from input file ...", end="\r")
        self.load_df_from_file(input_path)
        self.generate_all_frames(frame_width_us, output_path, fps=fps)

    def get_eta(self, current_event):
        if current_event == 0:
            return "-"
        eta = (time.time() - self.start_time_logs)*(self.n_events - current_event) / current_event
        return time.strftime('%H:%M:%S', time.gmtime(eta))
