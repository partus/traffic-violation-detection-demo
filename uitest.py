import cv2
import numpy as np
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
from ui import LineSelectionList

cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/taiwan.mp4")

builder = Gtk.Builder()
builder.add_from_file("test.glade")

greyscale = False

class Handler:
    def onDeleteWindow(self, *args):
        Gtk.main_quit(*args)

    def toggleGreyscale(self, *args):
        global greyscale
        greyscale = ~ greyscale
    def onPixelClicked (box, event):
       print event.x, event.y

window = builder.get_object("window1")
image = builder.get_object("image")
builder.get_object("lineMenu").add(LineSelectionList)
window.show_all()
builder.connect_signals(Handler())

def show_frame(*args):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    if greyscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pb = GdkPixbuf.Pixbuf.new_from_data(frame.tostring(),
                                        GdkPixbuf.Colorspace.RGB,
                                        False,
                                        8,
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[2]*frame.shape[1])
    image.set_from_pixbuf(pb.copy())
    return True

# from tracking import tracking
<<<<<<< HEAD
# GLib.idle_add(tracking)
GLib.idle_add(show_frame)
=======
GLib.idle_add(tracking)
# GLib.idle_add(show_frame)
>>>>>>> 36c1dbe5eaef4ac773ee119a65bafec9bdcb45c9
Gtk.main()
