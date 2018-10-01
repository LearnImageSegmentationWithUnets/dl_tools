# coding: utf-8

"""
Utilities for processing inputs and viewing and editing inputs and outputs interactivly in Jupyter lab

based on ipywidgets
"""
import os
import matplotlib as mpl
from ipywidgets import widgets
from pathlib import Path


class LabelsEditor(object):

    def __init__(self, labels_fname='enter file path here', colors_fname='enter file path here'):
        if labels_fname != 'enter file path here':
            self.labels_fname = labels_fname
        else:
            self.labels_fname = ""

        if colors_fname != 'enter file path here':
            self.colors_fname = colors_fname
        else:
            self.colors_fname = ""

        self.cc = mpl.colors.ColorConverter()
        self.labels_file = widgets.Text(
            value=self.labels_fname,
            placeholder='Type something',
            description='Labels file:',
            disabled=False
        )

        self.colors_file = widgets.Text(
            value=self.colors_fname,
            placeholder='Type something',
            description='Colors file:',
            disabled=False
        )
        self.files_widget = widgets.HBox([self.labels_file, self.colors_file])
        self.items_widget = widgets.VBox()

        self.save_btn = widgets.Button(
            description='Save to Files',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='save'
        )
        self.save_btn.on_click(self.save_one)

        self.add_btn = widgets.Button(
            description='Add Class',
            disabled=False,
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='plus')
        self.add_btn.on_click(self.add_one)

        self.del_btn = widgets.Button(
            description='Delete last Class',
            disabled=False,
            button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='minus')
        self.del_btn.on_click(self.del_last)

        self.load_btn = widgets.Button(
            description='Load From File',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon=''
        )
        self.del_btn.on_click(self.load)

        self.labels_widget = widgets.VBox([])

        self.controls = widgets.HBox([self.load_btn, self.save_btn, self.add_btn, self.del_btn])
        self.label_editor = widgets.VBox([self.files_widget, self.labels_widget, self.controls])

        if os.path.exists(self.labels_fname) and \
           os.path.exists(self.colors_fname):
            self.load()

    def del_last(self, b):
        self.labels_widget.children = (*self.labels_widget.children[:-1], )

    def add_one(self, b):
        new_label = self.make_label('new class...')
        new_color = self.make_colorpicker('red')
        new_item = widgets.HBox([new_label, new_color])

        self.labels_widget.children += (new_item, )

    def load(self):
        self.labels_fname = self.labels_file.value
        self.colors_fname = self.colors_file.value

        labels = Path(self.labels_fname).read_bytes().split()
        colors = Path(self.colors_fname).read_bytes().split()

        items = []
        for label, color in zip(labels, colors):
            label_text = self.make_label(label)
            color_picker = self.make_colorpicker(mpl.colors.to_hex(self.cc.to_rgb(color.decode("utf-8"))))
            item = widgets.HBox([label_text, color_picker])

            items.append(item)

            self.labels_widget.children = (*items, )

    @staticmethod
    def make_colorpicker(color):
        return widgets.ColorPicker(
            concise=False,
            description='Pick a color',
            value=color,
            disabled=False)

    @staticmethod
    def make_label(label):
        return widgets.Text(
            value=label,
            placeholder='Type something',
            description='Class:',
            disabled=False
        )

    def save_one(self, b):
        label_fname = self.labels_file.value
        color_fname = self.colors_file.value

        label_f = open(label_fname, 'w')
        color_f = open(color_fname, 'w')

        labels = [item.children[0].value for item in self.labels_widget.children[1:-1]]
        colors = [item.children[1].value for item in self.labels_widget.children[1:-1]]

        label_f.write('\n'.join(labels))
        color_f.write('\n'.join(colors))

        label_f.close()
        color_f.close()
