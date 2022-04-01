# %%
import chronoptics.tof as tof
import pyrealsense2 as rs
import cv2
import numpy as np
import argparse

# %%


def run(nframes: int, name: str, serial: str):
    csfFile = "kea_" + name + ".csf"
    bagFile = "realsense_" + name + ".bag"

    # Setting up the Kea camera
    """
    user_config = tof.UserConfig()
    user_config.setEnvironment(tof.ImagingEnvironment.INSIDE)
    user_config.setFps(30)
    user_config.setMaxDistance(15.0)
    user_config.setIntegrationTime(tof.IntegrationTime.MEDIUM)
    user_config.setStrategy(tof.Strategy.BALANCED)
    """

    user_config = tof.UserConfig()
    user_config.setEnvironment(tof.ImagingEnvironment.INSIDE)
    user_config.setFps(30)
    user_config.setMaxDistance(10)
    user_config.setIntegrationTime(tof.IntegrationTime.MEDIUM)
    user_config.setStrategy(tof.Strategy.BALANCED)

    cam = tof.KeaCamera(tof.ProcessingConfig(), serial)
    cam_config = user_config.toCameraConfig(cam)

    cam.setCameraConfig(cam_config)
    proc = cam_config.defaultProcessing()
    proc.setIntensityScale(10.0)
    cam.setProcessConfig(proc)

    cam.setOnCameraProcessing(False)

    # We capture RAW because RADIAL has issues right now.
    tof.selectStreams(cam, [tof.FrameType.RAW, tof.FrameType.BGR])

    writer = tof.createCsfWriterCamera(csfFile, cam)

    cam.start()

    # Warming up the Kea camera.
    for n in range(0, 50):
        kea_frames = cam.getFrames()

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    config.enable_record_to_file(bagFile)

    pipeline.start(config)

    try:
        for n in range(0, nframes):
            frames = pipeline.wait_for_frames()
            kea_frames = cam.getFrames()
            for frame in kea_frames:
                writer.writeFrame(frame)

            print("Frame {:d}".format(n))

    finally:
        pipeline.stop()
        cam.stop()
    return


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--nframes',
        help='The number of frames to capture',
        required=False,
        default=100)
    parser.add_argument(
        '--name',
        help='The name to capture',
        required=True,
        default="capture")
    parser.add_argument(
        '--serial',
        help='The Kea camera serial number to capture from',
        required=True,
        default="202002a")

    args = parser.parse_args()
    run(int(args.nframes), str(args.name), str(args.serial))


if __name__ == '__main__':
    main()
