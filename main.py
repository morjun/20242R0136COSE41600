from lib import PointProcessor
import argparse

def main():
    parser = argparse.ArgumentParser(description='COSE416 HW1: Human Detection with a sequence of LiDAR sensor data(.pcd)')
    parser.add_argument("scenario", nargs='?', type=int, default=1, help="Select scenario number (1~7)")
    args = parser.parse_args()

    pcd_processor = PointProcessor(args)
    pcd_processor.run()


if __name__ == "__main__":
    main()
