
import pyrealsense2 as rs
import numpy as np
import os
import argparse

def run(name: str, outdir: str):

    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    bagFile = "realsense_" + name + ".bag"

    # Declare pointcloud object, for calculating pointclouds and texture mappings
    pc = rs.pointcloud()
    # We want the points object to be persistent so we can display the last cloud when a frame drops
    points = rs.points()

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Start streaming with chosen configuration
    rs.config.enable_device_from_file(config, bagFile)

    pipe.start(config)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        for n in range(0, 1000):
            print("Frame {:d}".format(n))
            # Wait for the next set of frames from the camera
            frames = pipe.wait_for_frames()
            aligned_depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.flip(np.array(color_frame.get_data()), 2)

            pc.map_to(color_frame)
            points = pc.calculate(aligned_depth_frame)

            # This had swapped the rgb around ...
            # We manually edit the PLY header to correct as that is easier
            points.export_to_ply(
                os.path.join(outdir, "rs_frame_{:d}.ply".format(n)), color_frame)
            print("Done")
    finally:
        pipe.stop()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--name',
        help='The number of frames to capture',
        required=True)
    parser.add_argument(
        '--outdir',
        help='The Maximum distance to render out to',
        required=False,
        default="kea_cloud")

    args = parser.parse_args()
    run(str(args.name), str(args.outdir))


if __name__ == '__main__':
    main()
