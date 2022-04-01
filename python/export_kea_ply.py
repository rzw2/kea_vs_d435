import numpy as np
import cv2
import chronoptics.tof as tof
from matplotlib import cm as colormap
import os
import argparse

from plyfile import PlyElement
from plyfile import PlyData


def get_frame(frames, frame_type):
    for frame in frames:
        if frame.frameType() == frame_type:
            return frame


def tof_to_ply(x, y, z, bgr):
    """
    Converts tof_chrono frames to PlyElement from plyfile

    Parameters
    ----------
    tof_frames : List
        List of frames returned by getFrames() by a tof_chrono camera

    Returns
    ----------
    el
        The PLY
    """

    nrows, ncols = np.shape(x)

    x_vec = np.reshape(x, (-1, 1))
    y_vec = np.reshape(y, (-1, 1))
    z_vec = np.reshape(z, (-1, 1))
    rgb_vec = np.reshape(bgr, (nrows*ncols, 3))

    rm_pts = ~np.isnan(z_vec) | ~(z_vec == 0)

    npts = np.sum(rm_pts)

    ply_pts = np.zeros(
        npts,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    ind = 0
    for n in range(0, npts):
        if rm_pts[n]:
            ply_pts[ind] = (x_vec[n], y_vec[n], z_vec[n],
                            rgb_vec[n, 2], rgb_vec[n, 1], rgb_vec[n, 1])
            ind = ind + 1
    el = PlyElement.describe(ply_pts, "vertex")
    return el


def run(name: str, outdir: str):

    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    csfFile = "kea_" + name + ".csf"

    cam = tof.CsfCamera(tof.ProcessingConfig(), csfFile)

    cam_config = cam.getCameraConfig()
    proc = cam_config.defaultProcessing()
    proc = cam_config.defaultProcessing()
    proc.setAmpThresholdMin(0.25)
    proc.setFlyingDistance(250)
    proc.setLocalMeansEnabled(True)
    proc.setLocalMeansSize(7)
    proc.setMedianEnabled(False)
    proc.setIntensityScale(5.0)
    proc.setTemporalSigma(2.0)

    cam.setProcessConfig(proc)

    tof.selectStreams(
        cam, [tof.FrameType.Z, tof.FrameType.X, tof.FrameType.Y, tof.FrameType.BGR_PROJECTED])

    cam.start()

    nframes = cam.getFrameNumber()

    for n in range(0, nframes):
        print("Frame {:d}".format(n))
        kea_frames = cam.getFramesAt(n)

        x = get_frame(kea_frames, tof.FrameType.X)
        y = get_frame(kea_frames, tof.FrameType.Y)
        z = get_frame(kea_frames, tof.FrameType.Z)
        bgr = get_frame(kea_frames, tof.FrameType.BGR_PROJECTED)

        el = tof_to_ply(np.asarray(x)/1000.0, np.asarray(y)/1000.0,
                        -1.0*np.asarray(z)/1000.0, np.asarray(bgr))

        PlyData([el]).write(
            os.path.join(outdir, "kea_cloud_frame_{:d}.ply".format(n)))

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
