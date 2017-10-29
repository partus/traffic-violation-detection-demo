import cv2
import numpy as np
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
from ui import LineSelectionList

cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/taiwan.mp4")

builder = Gtk.Builder()
builder.add_from_file("test.glade")

from linetools import LineStorage
lineStorage = LineStorage()
class Handler:
    def onDeleteWindow(self, *args):
        Gtk.main_quit(*args)
        tracking.stop()
    def toggleGreyscale(self, *args):
        global greyscale
        greyscale = ~ greyscale
    def onPixelClicked (self,box, event):
        point = np.array([event.x, event.y],dtype=np.int32)
        line = lineStorage.clickMatch(point)
        if not line is None:
            lineSelectionList.add(4)
            lineSelectionList.show_all()
            print(point)

window = builder.get_object("window1")
image = builder.get_object("image")
lineSelectionList = LineSelectionList()
builder.get_object("lineMenu").add(lineSelectionList)
window.show_all()
builder.connect_signals(Handler())


def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pb = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                        GdkPixbuf.Colorspace.RGB,
                                        False,
                                        8,
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[2]*frame.shape[1])
    image.set_from_pixbuf(pb.copy())

from tracking import Tracking
tracking = Tracking(show_frame, lineStorage)
GLib.idle_add(tracking)
# GLib.idle_add(show_frame)
Gtk.main()
