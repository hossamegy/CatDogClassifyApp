from enum import Enum

class ResponseSignal(Enum):
    FILE_TYPE = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    FILE_TYPE_NOT_SUPPORTED='FILE_TYPE_NOT_SUPPORTED'