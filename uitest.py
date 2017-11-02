import cv2
import numpy as np
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
from ui import LineSelectionList

import asyncio,gbulb
gbulb.install(gtk=True)

gbulb.get_event_loop()



builder = Gtk.Builder()
builder.add_from_file("test.glade")

from linetools import LineStorage
lineStorage = LineStorage()
class Handler:
    def onDeleteWindow(self, *args):
        Gtk.main_quit(*args)
        tracking.stop()
    def onPixelClicked (self,box, event):
        point = np.array([event.x, event.y],dtype=np.int32)
        id = lineStorage.clickMatch(point)
        if not id is None:
            lineSelectionList.add(id)
            lineSelectionList.show_all()
            print(point)

window = builder.get_object("window1")
image = builder.get_object("image")
lineSelectionList = LineSelectionList({
    'onRemove': lineStorage.remove,
    'onTypeChoose': lineStorage.onTypeChoose
})
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
    image.show_all()


# from tracking import Tracking
# tracking = Tracking(show_frame, lineStorage,"/data/livetraffic/2017-08-27/3/tokyo.mp4",0.5)
# tracking = Tracking(show_frame, lineStorage,"/data/livetraffic/2017-07-18/taiwan.mp4",1)
# GLib.idle_add(tracking)
# GLib.idle_add(show_frame)
# Gtk.main()

loop.run_forever()
