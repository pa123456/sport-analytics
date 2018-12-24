import math
import cv2
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
n = 0
scale = 8
video_scale = 20

def get_axis(path, bbox):
    # Set up tracker.

    sentinel = True
    initial_threshold = 0
    displacement_nodes = []

    # Check if the initial threshold was exceeded by the projectile
    threshold_crossed = False


    tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(path)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Uncomment the line below to select a different bounding box
    if args.select:
        bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while sentinel:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            displacement_nodes.append((int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2)))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            if len(displacement_nodes) > 1:
                initial_threshold = displacement_nodes[0][1] - 15
                if displacement_nodes[-1][0] <= 2:
                    sentinel = False
                if displacement_nodes[-2][1] < displacement_nodes[-1][1] and not threshold_crossed and len(displacement_nodes) > 30:
                    sentinel = False
                for i in range(len(displacement_nodes) - 1):
                    cv2.line(frame, displacement_nodes[i], displacement_nodes[i + 1], (0, 255, 0), 3)
            if int(bbox[1] + bbox[3] // 2) < initial_threshold:
                cv2.putText(frame, "Crossed Threshold Upwards", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255),
                            2)
                threshold_crossed = True
            else:
                threshold_crossed = False
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "CSRT Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        if args.show:
            # Display result
            cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    df = pd.DataFrame(displacement_nodes)
    return df


def graph(df):
    xa, _ = np.polyfit(np.arange(len(df)), df.X, 1)
    ya, yb, _ = np.polyfit(np.arange(len(df)), df.Y0, 2)
    za, _ = np.polyfit(np.arange(len(df)), df.Z, 1)

    t = np.arange(int(max(np.roots((ya, yb, 0))) + 1))
    x = xa * t
    y = -(ya * t**2) - yb * t
    z = za * t

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, 90)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.axis("equal")

    def init():
        ax.plot(x, y, z)
        # ax.plot((x[0], xa + x[0]), (y[0], 2 * ya * 0 + yb), (z[0], za + z[0]))
        return fig,


    def animate(i):
        global n

        if n >= len(t):
            return fig,

        ax.plot((x[n], (scale * xa) + x[n]), (y[n], y[n]), (z[n], z[n]), c=(0.9, 0.2, 0.2, 0.4))
        ax.plot((x[n], x[n]), (y[n], scale * (2 * ya * n + yb) + y[n]), (z[n], z[n]), c=(0.2, 0.9, 0.2, 0.4))
        ax.plot((x[n], x[n]), (y[n], y[n]), (z[n], (scale * za) + z[n]), c=(0.2, 0.2, 0.9, 0.4))

        n += 1
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=False)

    plt.show()


def video(df_coord, df, path):
    xa, _ = np.polyfit(np.arange(len(df)), df.X, 1)
    ya, yb, _ = np.polyfit(np.arange(len(df)), df.Y0, 2)
    za, _ = np.polyfit(np.arange(len(df)), df.Z, 1)
    x = None
    y = None
    filename = None

    if args.side_view_path[0] == path:
        filename = "processed/side.mp4"
        # x: Z, y: Y0
        x = np.ones(len(df_coord)) * video_scale * za
        y = video_scale * (2 * ya * np.arange(len(df_coord)) + yb)
    else:
        filename = "processed/front.mp4"
        # x: X, y: Y0
        x = np.ones(len(df_coord)) * video_scale * xa
        y = video_scale * (2 * ya * np.arange(len(df_coord)) + yb)

     # Read video
    video = cv2.VideoCapture(path)


    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (960,540))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    frame_num = 0

    while video.isOpened():
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print("exit")
            break
        try:
            # draw x vector on graph
            point = tuple(df_coord.iloc[frame_num, :].values)
        except:
            break

        cv2.arrowedLine(frame, point, (int(point[0] + x[frame_num]), point[1]),
                 (0, 0, 255, 255), 2)
        cv2.arrowedLine(frame, point, (point[0], int(point[1] + y[frame_num])),
                 (0, 0, 255, 255), 2)

        instruction = None
        if args.front_view_path[0] == path:
            if xa < -0.5:
                instruction = "Kick a bit to the left"
            elif xa > 0.5:
                instruction = "Kick a bit to the right"
            else:
                "You are kicking accurately in the x direction"
        else:
            instruction = str(round(90 - math.tan(y[0] / x[0]) * 100)) + \
            ' degrees' + "  Try to aim for 45 degrees"

        cv2.putText(frame, instruction, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 100, 255), 2)


        if args.show:
            # Display result
            cv2.imshow("Tracking", frame)

        out.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        frame_num += 1

    out.release()






def main():
    global paser, args
    parser = argparse.ArgumentParser(description="Indicate two videos' \
                                     directory")
    parser.add_argument('--side', dest='side_view_path', type=str, nargs='+',
                        help='Directoy for side way view video file')
    parser.add_argument('--front', dest='front_view_path', type=str, nargs='+',
                        help='Directoy for front view video file')
    parser.add_argument('--show', dest='show', type=bool, nargs='+',
                        help='show graph and videos')
    parser.add_argument('--select', dest='select', type=bool, nargs='+',
                        help='select ball manually')
    args = parser.parse_args()
    df_1 = get_axis(args.side_view_path[0], (792, 430, 41, 34))
    df_1.columns = ["Z", "Y0"]
    df_2 = get_axis(args.front_view_path[0], (470, 400, 40, 40))
    df_2.columns = ["X", "Y1"]
    # print(df_1)
    # print(df_2)

    df = pd.concat([df_1, df_2], axis=1)
    df.dropna(inplace=True)
    #print(df)
    if args.show:
        graph(df)
    video(df_1, df, args.side_view_path[0])
    video(df_2, df, args.front_view_path[0])


if __name__ == '__main__':
    main()
