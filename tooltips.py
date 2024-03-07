import tkinter as tk

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.hidetip)
        self.widget.bind("<ButtonPress>", self.hidetip)

    def display_tip(self):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tip_window, text=self.text, background="#f9e37c", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack()

    def schedule(self, event):
        self.x, self.y = event.x, event.y
        self.id = self.widget.after(500, self.display_tip)

    def hidetip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None