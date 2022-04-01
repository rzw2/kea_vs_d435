
import numpy as np
import cv2
import chronoptics.tof as tof
import pyrealsense2 as rs
from matplotlib import cm as colormap
import argparse

def no_rot(frame):
    return frame


def colourFrame(ToFframe: np.array, climit: list, rot_func, cm="jet", badpixColor=[0, 0, 0]) -> np.ndarray:

    frame = ToFframe
    cmap = colormap.get_cmap(cm)

    blank_pix = rot_func(np.isnan(frame) | (frame == 0))
    cm_frame = rot_func(np.copy(frame))

    cm_frame[cm_frame < climit[0]] = climit[0]
    cm_frame[cm_frame > climit[1]] = climit[1]

    cm_frame = (cm_frame - climit[0]) / (climit[1] - climit[0])

    color_frame = cmap(cm_frame)

    out_frame = np.flip(np.uint8(color_frame[:, :, 0:3] * 255), axis=2)
    out_frame[blank_pix] = badpixColor
    return out_frame


def run(name: str, dmin: int, dmax: int):
    csfFile = "kea_" + name + ".csf"
    bagFile = "realsense_" + name + ".bag"

    kea_mpeg = "kea_" + name + "_depth.avi"
    rs_mpeg = "realsense_" + name + "_depth.avi"

    cam = tof.CsfCamera(tof.ProcessingConfig(), csfFile)

    # Setup the processing pipeline for the Kea camera.
    cam_config = cam.getCameraConfig()
    proc = cam_config.defaultProcessing()
    proc.setAmpThresholdMin(0.25)
    proc.setFlyingDistance(250)
    proc.setLocalMeansEnabled(True)
    proc.setLocalMeansSize(7)
    proc.setMedianEnabled(False)
    proc.setIntensityScale(5.0)
    proc.setTemporalSigma(2.0)

    cam.setProcessConfig(proc)

    # tof.FrameType.INTENSITY, tof.FrameType.BGR_PROJECTED

    tof.selectStreams(
        cam, [tof.FrameType.Z])

    cam.start()

    roi = cam_config.getRoi(0)
    nrows = roi.getImgRows()
    ncols = roi.getImgCols()
    fps = 30.0

    out_kea = cv2.VideoWriter(
        kea_mpeg, cv2.VideoWriter_fourcc(
            "M", "J", "P", "G"), fps, (ncols, nrows)
    )
    out_rs = cv2.VideoWriter(
        rs_mpeg, cv2.VideoWriter_fourcc(
            "M", "J", "P", "G"), fps, (ncols, nrows)
    )

    # The intel realSense
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bagFile)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("RealSense Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Kea Depth", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    #colorizer = rs.colorizer()

    cnt = 0
    nframes = cam.getFrameNumber()
    # Streaming loop
    for n in range(0, nframes):
        # Get frameset of depth
        get_frame = cnt % nframes
        cnt = cnt + 1

        # Get the cameras frames
        kea_frames = cam.getFramesAt(get_frame)
        frames = pipeline.wait_for_frames()

        # Get depth frames
        depth_frame = frames.get_depth_frame()
        kea_depth = np.asarray(kea_frames[0])
        depth_image = np.asanyarray(depth_frame.get_data())

        kea_color_frame = colourFrame(
            kea_depth, [dmin, dmax], np.flipud, cm='jet')
        depth_colormap = colourFrame(
            depth_image, [dmin, dmax], np.fliplr, cm='jet')

        # Convert depth_frame to numpy array to render image in opencv
        out_kea.write(kea_color_frame)
        out_rs.write(depth_colormap)

        # Render image in opencv window
        cv2.imshow("RealSense Depth", depth_colormap)
        cv2.imshow("Kea Depth", kea_color_frame)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

    out_kea.release()
    out_rs.release()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--name',
        help='The number of frames to capture',
        required=True)
    parser.add_argument(
        '--dmax',
        help='The Maximum distance to render out to',
        required=False,
        default=1500)
    parser.add_argument(
        '--dmin',
        help='The Minimum distance to render out to',
        required=False,
        default=0)

    args = parser.parse_args()
    run(str(args.name), int(args.dmin), int(args.dmax))


if __name__ == '__main__':
    main()
