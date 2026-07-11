import imageio

import os


# folder with frames

frame_dir = "."


# collect frames in order

frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])


# create video

with imageio.get_writer("movie.mp4", fps=10) as writer:

    for f in frames:

        image = imageio.imread(os.path.join(frame_dir, f))

        writer.append_data(image)


print("Video saved as movie.mp4")
