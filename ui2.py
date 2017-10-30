# import gi
# gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk
#
# win = Gtk.Window()
# win.connect("delete-event", Gtk.main_quit)
# win.show_all()
# Gtk.main()


import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk,Gdk

class LineTypeSelector(Gtk.Box):
    def __init__(self,id,cbs):
        # self.cbs = {
        #     'onRemove':
        #     'onTypeChoose':
        # }
        self.id = id
        self.cbs = cbs
        super(Gtk.Box, self).__init__(spacing=6)
        hbox = self
        self.hasLightBox = False
        close = Gtk.Button.new_with_label("Click Me")
        close.connect("clicked", self.on_close_click)
        hbox.pack_start(close, False, False, 0)

        button1 = Gtk.RadioButton.new_with_label_from_widget(None, "Dividing")
        button1.connect("toggled", self.on_button_toggled, "par" )
        hbox.pack_start(button1, False, False, 0)

        button2 = Gtk.RadioButton.new_from_widget(button1)
        button2.set_label("Front")
        button2.connect("toggled", self.on_button_toggled, "front" )
        hbox.pack_start(button2, False, False, 0)
        self.hbox = hbox
        # hbox.pack_start(LineColorSelector(), False, False, 0)
    def on_button_toggled(self, button, name):
        if button.get_active():
            state = "on"
            if name == "front":
                print("front")
                self.hasLightBox = True
                self.lightBox = LineColorSelector()
                self.hbox.pack_start(self.lightBox, False, False, 0)
                self.hbox.show_all()
            if name == "par":
                if(self.hasLightBox):
                    print("destroy")
                    self.lightBox.destroy()
        else:
            state = "off"
        print("Button", name, "was turned", state)

    def on_close_click(self,button):
        if 'onRemove' in self.cbs:
            self.cbs['onRemove'](self.id)
        True
    def remove(self):
        self.destroy()
class LineColorSelector(Gtk.Box):
    def get_color(self,rgb):
        colorh=rgb
        color=Gdk.RGBA()
        color.parse(colorh)
        color.to_string()
        return color
    def __init__(self):
        super(Gtk.Box, self).__init__(spacing=6)
        hbox = self
        button1 = Gtk.RadioButton.new_with_label_from_widget(None, "R")
        # button1.modify_bg(Gtk.StateType.PRELIGHT, Gdk.color_parse('#234fdb'))

        button1.override_background_color(Gtk.StateFlags.NORMAL, self.get_color("#FF0000"))
        button1.connect("toggled", self.on_button_toggled, "1")
        hbox.pack_start(button1, False, False, 0)

        button2 = Gtk.RadioButton.new_with_mnemonic_from_widget(button1,"Y")
        button2.override_background_color(Gtk.StateFlags.NORMAL, self.get_color("#FFFF00"))
        button2.connect("toggled", self.on_button_toggled, "2")
        hbox.pack_start(button2, False, False, 0)

        button3 = Gtk.RadioButton.new_with_mnemonic_from_widget(button1,"G")
        button3.override_background_color(Gtk.StateFlags.NORMAL, self.get_color("#00FF00"))
        button3.connect("toggled", self.on_button_toggled, "3")
        hbox.pack_start(button3, False, False, 0)

    def on_button_toggled(self, button, name):
        if button.get_active():
            state = "on"
        else:
            state = "off"
        print("Button", name, "was turned", state)
