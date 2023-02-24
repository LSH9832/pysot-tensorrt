from cv2 import *



class DoubleEyesCamera(VideoCapture):

    def __init__(self, *args, **kwargs):
        super(DoubleEyesCamera, self).__init__(*args, **kwargs)

        self.size = (640, 480)
        self.set(CAP_PROP_FRAME_WIDTH, self.size[0] * 2)
        self.set(CAP_PROP_FRAME_HEIGHT, self.size[1])

    def setsize(self, width, height):
        self.size = (width, height)
        self.set(CAP_PROP_FRAME_WIDTH, width * 2)
        self.set(CAP_PROP_FRAME_HEIGHT, height)

    def readleft(self):
        success, frame = self.read()
        if success:
            return success, frame[:, 0:self.size[0]]
        else:
            return success, None

    def readright(self):
        success, frame = self.read()
        if success:
            return success, frame[:, self.size[0]:2*self.size[0]]
        else:
            return success, None

    def readall(self):
        success, frame = self.read()
        if success:
            return success, frame[:, 0:self.size[0]], frame[:, self.size[0]:2*self.size[0]]
        else:
            return success, None, None


class __ImageType:
    JPG = 0
    PNG = 1
    BMP = 2
    JPEG = 3
    GIF = 4
    ALL = -1


IMAGE_TYPE = __ImageType()


class ImageDatasetCapture(object):
    """
    read images from one dir
    """

    __image_type_list = ['jpg', 'png', 'bmp', 'jpeg', 'gif']
    __image_list = []
    __image_type = IMAGE_TYPE.ALL

    def __init__(self, path: str or list or None = None, image_type=IMAGE_TYPE.ALL):
        if path is not None:
            self.open(path, image_type)

    def open(self, path: str or list, image_type=IMAGE_TYPE.ALL):
        self.release()
        import os
        from glob import glob

        pathlist = [path] if isinstance(path, str) else path
        for path in pathlist:
            ori_path = path
            if not os.path.isdir(path):
                assert '*' in path, 'Wrong input format!'
                TYPE_DATA = path.split('/')[-1]
                # path = path[:-len(TYPE_DATA)]
                if TYPE_DATA == '*':
                    self.__image_type = IMAGE_TYPE.ALL

                else:
                    IS_SUPPORT_TYPE = False
                    for i, support_type in enumerate(self.__image_type_list):
                        if support_type in TYPE_DATA:
                            self.__image_type = i
                            IS_SUPPORT_TYPE = True

                    assert IS_SUPPORT_TYPE, 'Type is not supported!'
                self.__image_list += sorted(glob(ori_path))
            else:
                self.__image_type = image_type

                if not path.endswith('/'):
                    path += '/'
                if self.__image_type < 0:
                    for this_type in self.__image_type_list:
                        self.__image_list += sorted(glob('%s*.%s' % (path, this_type)))

                else:
                    self.__image_list += sorted(glob('%s*.%s' % (path, self.__image_type)))
        self.__image_list = sorted(self.__image_list)

    def isOpened(self):
        return bool(len(self.__image_list))

    def read(self):
        success, image = False, None
        if self.isOpened():
            image = imread(self.__image_list[0])
            success = True
            del self.__image_list[0]
        return success, image

    def release(self):
        self.__image_list = []


class __Fps:
    from time import time
    __time = time
    __source = None
    __ts = []

    def __call__(self, ID: int):
        assert ID >= 0

        while len(self.__ts) <= ID:
            self.__ts.append(self.__time())

        t1 = self.__time()
        try:
            now_fps = 1. / (t1 - self.__ts[ID])
        except:
            now_fps = 0
        self.__ts[ID] = t1
        return now_fps


fps = __Fps()


def OpenSource(source: str or int, double_eyes=False):

    """
    choose source automatically
    """
    if isinstance(source, str):
        if source.isdigit():
            source = int(source)

    is_string, is_int, is_list = isinstance(source, str), isinstance(source, int), isinstance(source, list)
    assert is_string or is_int or is_list, 'wrong input!'
    if is_list:
        return ImageDatasetCapture(source)
    elif is_string:
        if source.startswith("rtmp://") or source.startswith("rtsp://") or source.startswith("http://") or source.split(".")[-1] in ["mp4", "mkv", "rmvb", "ts", "avi", "flv"]:
            return VideoCapture(source)
        else:
            return ImageDatasetCapture(source)
    elif double_eyes:
        return DoubleEyesCamera(source)
    else:
        return VideoCapture(source)


# __all__ = [DoubleEyesCamera, IMAGE_TYPE, ImageDatasetCapture, fps, OpenSource]
