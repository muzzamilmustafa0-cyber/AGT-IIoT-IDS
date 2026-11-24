# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Figure 1 (revision): same card style as the submitted figure, smaller canvas so text is legible at print size
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
import os, sys

BLUE = RGBColor(0x2E, 0x6D, 0xB4); ORANGE = RGBColor(0xE0, 0x77, 0x2A)
GREEN = RGBColor(0x2E, 0x9E, 0x4F); PURPLE = RGBColor(0x6B, 0x2F, 0xA0)
GOLD = RGBColor(0xE3, 0x9A, 0x00); INK = RGBColor(0x22, 0x33, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); CARDLN = RGBColor(0xC2, 0xCB, 0xD6)
GREYTX = RGBColor(0x40, 0x4A, 0x58); ENCBG = RGBColor(0xF4, 0xF7, 0xFB); LILAC = RGBColor(0xF3, 0xEC, 0xFA)
ICN = lambda k: os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons"), k + ".png")

prs = Presentation(); prs.slide_width = Inches(9.0); prs.slide_height = Inches(6.3)
sl = prs.slides.add_slide(prs.slide_layouts[6]); shp = sl.shapes

def tbox(x, y, w, h, text, size=13, color=INK, bold=False, italic=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tb = shp.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, Pt(1))
    for j, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = "Calibri"
    return tb

def rrect(x, y, w, h, fill, line=None, lw=1.25):
    s = shp.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08; s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(lw)
    s.shadow.inherit = False; return s

def header(x, y, w, text, color, size=14, hh=0.40):
    h = rrect(x, y, w, hh, color); tf = h.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    for m in ("margin_left", "margin_right"): setattr(tf, m, Pt(2))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = "Calibri"

def icon(x, y, s, key):
    if os.path.exists(ICN(key)): shp.add_picture(ICN(key), Inches(x), Inches(y), Inches(s), Inches(s))

def card(x, y, w, h, title, color, key, sub, size=13, isz=0.55):
    rrect(x, y, w, h, WHITE, line=CARDLN); header(x, y, w, title, color)
    icon(x + 0.12, y + 0.40 + (h - 0.40 - isz) / 2, isz, key)
    tbox(x + 0.72, y + 0.44, w - 0.78, h - 0.48, sub, size=size, color=GREYTX, align=PP_ALIGN.LEFT)

def arrow(x1, y1, x2, y2, color=INK, w=2.0):
    c = shp.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(w)
    ln = c.line._get_or_add_ln(); ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))

# row A: acquisition -> preprocessing -> graph
yA, hA, wA, gA = 0.15, 1.35, 2.66, 0.36
xs = [0.15, 0.15 + wA + gA, 0.15 + 2 * (wA + gA)]
card(xs[0], yA, wA, hA, "Traffic acquisition", BLUE, "traffic", "flow, host and\nconnectivity attributes")
card(xs[1], yA, wA, hA, "Preprocessing", BLUE, "preprocess", "impute, encode, scale\n(training statistics only)")
card(xs[2], yA, wA, hA, "Inductive k-NN graph", ORANGE, "knn", "exact GPU search:\nk reference flows\nper flow")
for i in range(2): arrow(xs[i] + wA, yA + hA / 2 + 0.18, xs[i + 1], yA + hA / 2 + 0.18)

# row B: encoder
yB, hB = 1.95, 1.95
rrect(0.15, yB, 8.7, hB, ENCBG, line=CARDLN); header(0.15, yB, 8.7, "AGT encoder  (residual connections, layer normalization, dropout)", PURPLE)
blocks = [("Entity\nembedding +\nprojection", ORANGE, "input_proj"), ("Local graph\nattention", GREEN, "local_attn"),
          ("Landmark\nattention:\npool \u2192 read", PURPLE, "global_attn"), ("Local graph\nattention", GREEN, "local_attn"),
          ("MLP head\n\u2192 class\nlogits", ORANGE, "mlp")]
bw, bg = 1.50, 0.285; bx0 = 0.15 + (8.7 - (5 * bw + 4 * bg)) / 2; by = yB + 0.55; bh = 1.25
for i, (t, c, k) in enumerate(blocks):
    x = bx0 + i * (bw + bg); rrect(x, by, bw, bh, c)
    icon(x + (bw - 0.36) / 2, by + 0.06, 0.36, k)
    tbox(x + 0.04, by + 0.42, bw - 0.08, bh - 0.46, t, size=12.5, color=WHITE, bold=True)
    if i < 4: arrow(x + bw, by + bh / 2, x + bw + bg, by + bh / 2, w=1.5)
arrow(xs[2] + wA / 2, yA + hA, xs[2] + wA / 2, yB)

# row C: detection, alerts, adaptation
yC, hC = 4.35, 1.85
card(5.95, yC, 2.9, 0.9, "Detection", GREEN, "detection", "benign + 9 attack\nfamilies", size=12.5)
rrect(5.95, yC + 1.1, 2.9, 0.75, WHITE, line=CARDLN); header(5.95, yC + 1.1, 2.9, "Alerts", GOLD)
tbox(6.0, yC + 1.5, 2.8, 0.34, "analyst triage and labelling", size=12.5, color=GREYTX)
arrow(7.4, yB + hB, 7.4, yC)
rrect(0.15, yC, 5.3, hC, LILAC, line=PURPLE); header(0.15, yC, 5.3, "Rehearsal-based incremental adaptation", PURPLE)
icon(0.3, yC + 0.75, 0.6, "self_evolve")
tbox(1.05, yC + 0.45, 4.3, hC - 0.5,
     "per window: predict first, then a labelled subset A\u209c\n"
     "+ replay buffer B (M flows) \u2192 k-NN graph over A\u209c \u222a B\n"
     "\u2192 S gradient steps; B keeps the most recent M flows",
     size=12.5, color=INK, align=PP_ALIGN.LEFT)
arrow(5.95, yC + 1.62, 5.45, yC + 1.62, color=PURPLE)
tbox(5.36, yC + 1.2, 0.7, 0.3, "labels", size=11, color=PURPLE, italic=True)
arrow(2.8, yC, 2.8, yB + hB, color=PURPLE)
tbox(2.88, yB + hB + 0.04, 1.6, 0.36, "parameter update", size=11, color=PURPLE, italic=True, align=PP_ALIGN.LEFT)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure1_AGT_R1.pptx")
prs.save(out); print("saved", out)
