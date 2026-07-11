import sys
import os
import imageio.v2 as imag




# finds the folder with frames from script inputs

frame_dir = sys.argv[1]
output_file = ( 
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(frame_dir, "movie.mp4"))


# sorts png frames in order

frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])



# create video: fps= frames per second, joins frames together

with imag.get_writer(output_file, fps=10) as writer:

    for f in frames:

        image = imag.imread(os.path.join(frame_dir, f))

        writer.append_data(image)


print("Video saved as movie.mp4 in", output_file)