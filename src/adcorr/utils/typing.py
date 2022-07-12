from typing import Tuple, TypeVar

from numpy import dtype, ndarray

#: The underlying data type of a frame.
FrameDType = TypeVar("FrameDType", bound=dtype)
#: The number of frames in a stack of frames.
NumFrames = TypeVar("NumFrames", bound=int)
#: The number of pixels spanning the width of a frame.
FrameWidth = TypeVar("FrameWidth", bound=int)
#: The number of pixels spanning the height of a frame.
FrameHeight = TypeVar("FrameHeight", bound=int)

#: The shape of a frame; Comprising of a width and height.
FrameShape = Tuple[FrameWidth, FrameHeight]
#: The shape of a stack of frames; Comprising a count, width and height.
StackShape = Tuple[NumFrames, FrameWidth, FrameHeight]

#: A frame; Comprising a shape and a data type
Frame = ndarray[FrameShape, FrameDType]
#: A stack of frames; Comprising a shape and a data type
Frames = ndarray[StackShape, FrameDType]
