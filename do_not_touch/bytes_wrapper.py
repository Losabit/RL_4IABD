from ctypes import *


class WrappedData(Structure):
    _fields_ = [
        ("byte_count", c_uint64),
        ("data", c_void_p)
    ]


WrappedDataPtr = POINTER(WrappedData)


def get_bytes(pointer_to_wrapped_data: WrappedDataPtr):
    wrapped_data = WrappedData.from_buffer(pointer_to_wrapped_data.contents, 0)
    return (c_uint8 * wrapped_data.byte_count).from_address(wrapped_data.data)
