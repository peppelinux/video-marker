Video Marker
------------

Video Marker is a python application based on OpenCV that can automate:

- a video record from a webcam and save to a file
- prepend a video before your video record starts
- append a video after your video record ends
- add a watermark image

## Setup

````
pip install video-marker
````

## Documentation
````
video_marker -h
````

## Example
````
video_marker --watermark-fpath 'assets/logo.png' --monitor --pre-media assets/intro.mp4  --post-media assets/intro.mp4  --watermark-size 50
````

## License

ses [LICENSE](LICENSE) file.

## Author

Giuseppe De Marco
