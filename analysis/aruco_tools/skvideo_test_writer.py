import skvideo.io
import numpy as np

outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
outputdata = outputdata.astype(np.uint8)

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
for i in range(5):
        writer.writeFrame(outputdata[i, :, :, :])
writer.close()