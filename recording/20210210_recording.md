On raspberry pi

```bash
raspivid -w 4056 -h 3040 -fps 2 -t 20000 -cd MJPEG -b 25000000 -o ${OUTPUT}
```

Copy with

```bash
scp pi@{IP}:/home/pi/git/bee-box/${VIDEO} ./
```

Convert with:

```bash
ffmpeg -r 1 -i ${INPUT} -framerate 1 -c:v libx265 ${OUTPUT}
```

Finally, snip a picture of the image and run ARucoTag on that image with:


```bash
python detect_aruco_image.py --image ${IMAGE} --type DICT_5X5_100
```
