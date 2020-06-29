import cv2


def read_img(path):
    return cv2.imread(path)


def show_img(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def read_video(path):
    return cv2.VideoCapture(path)


def show_video(video):
    while True:
        ok, img = video.read()
        if ok:
            cv2.imshow("video", img)
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break


def read_web_camera():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)

    return camera


if __name__ == "__main__":
    show_img("cat", read_img("resources/cat.jpg"))