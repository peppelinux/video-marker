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

## Usage examples

Monitor your webcam
````
video_marker --video-cap 0 --monitor
````

Add watermark to your webcam
````
video_marker --watermark-fpath 'assets/logo.png' --monitor
````

A quite complete example
````
video_marker --watermark-fpath 'assets/logo.png' --monitor --pre-media assets/intro.mp4  --post-media assets/intro.mp4  --watermark-size 50
````

## License

ses [LICENSE](LICENSE) file.

## Author

Giuseppe De Marco
