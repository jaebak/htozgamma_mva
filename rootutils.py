import ROOT

def new_canvas(name = "", size = 500):
  canvases = ROOT.gROOT.GetListOfCanvases()
  iCanvas = canvases.GetEntries()
  if name == "":
    canvas_name = f"c_g_{iCanvas}"
  else: canvas_name = name
  return ROOT.TCanvas(canvas_name, canvas_name, size, size)

def get_max_th1():
  pad_list = ROOT.gPad.GetListOfPrimitives()
  maximum = 0
  for iobj, obj in enumerate(pad_list):
    class_name = obj.ClassName()
    if "TH1" in class_name:
      t_max = obj.GetMaximum()
      if t_max>maximum or iobj == 0: maximum = t_max;
  return maximum

def set_max(maximum):
  pad_list = ROOT.gPad.GetListOfPrimitives()
  for obj in pad_list:
    class_name = obj.ClassName()
    if 'TH1' in class_name: obj.SetMaximum(maximum)
    if 'THStack' in class_name: obj.SetMaximum(maximum)
  ROOT.gPad.Modified()
  ROOT.gPad.Update()

def set_max_th1(max_fraction = 1.05):
  maximum = get_max_th1() * max_fraction
  set_max(maximum)

import unicodedata
import re
def slugify(string):
  return re.sub(r'[-\s]+', '-',re.sub(r'[^\w\s-]', '',
                  unicodedata.normalize('NFKD', string)).strip().lower())
