from evisualiser import Sequence

if __name__ == "__main__":

    my_input_path =   # Enter input path ending .aedat
    my_output_path =  # Enter output path ending .mp4
    fps =             # Enter fps (integer) for output file

    seq = Sequence()
    seq.event_file_to_video(input_path=my_input_path, output_path=my_output_path, fps=fps)
