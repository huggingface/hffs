from hffs import HfFileSystem

fs = HfFileSystem()
print(fs.glob("hf://**"))
