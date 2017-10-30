import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from ui2 import LineTypeSelector


class ListBoxRowWithData(Gtk.ListBoxRow):
    def __init__(self, data):
        super(Gtk.ListBoxRow, self).__init__()
        self.data = data
        self.add(Gtk.Label(data))

class LineSelectionList(Gtk.Box):
    def __init__(self,cbs):
        self.cbs = cbs
        super(Gtk.Box, self).__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.list = {}
        box_outer = self
        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        box_outer.pack_start(listbox, True, True, 0)

        self.listbox = listbox
        for i in range(2):
            self.add(i)
    def add(self,id):
        row = Gtk.ListBoxRow()
        if id in self.list:
            self.list[id].remove()
        self.list[id] = row
        row.add(LineTypeSelector(id,{
            'onRemove': self.onItemRemove
        }))
        print(self.list)
        self.listbox.add(row)
    def getSmallest(self):
        i= 0
        while i in self.list:
            i+=1
        return i
    def onItemRemove(self,id):
            self.cbs['onRemove'](id)
            self.list[id].remove()
        True
    def onRadioChoose(self,id,type):
        True

class ListBoxWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Line Type Selector")
        self.set_border_width(10)

        box_outer = LineSelectionList()
        self.add(box_outer)

def startUi():
    win = ListBoxWindow()
    win.connect("delete-event", Gtk.main_quit)
    win.show_all()
    Gtk.main()

if __name__ == '__main__':
    startUi()
