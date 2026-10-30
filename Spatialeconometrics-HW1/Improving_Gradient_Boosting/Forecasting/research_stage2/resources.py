"""Read available physical memory without optional packages."""
import ctypes
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
        ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
def available():
    value=Memory();value.length=ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.available/2**30
if __name__=='__main__':print('AVAILABLE_MEMORY_GB',round(available(),2))
