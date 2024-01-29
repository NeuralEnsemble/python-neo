"""
This generates a diagram in .png and .svg from neo.core objects


Authors: Samuel Garcia, Julia Sprenger, Andrew Davison
"""

from datetime import datetime

import numpy as np
import quantities as pq
from matplotlib import pyplot
from matplotlib.patches import Rectangle, ArrowStyle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

import neo

line_height = 0.22
fontsize = 10.5
left_text_shift = 0.1
dpi = 100


def get_rect_height(name, obj):
    """
    calculate rectangle height
    """
    nlines = 1.5
    nlines += len(getattr(obj, "_all_attrs", []))
    nlines += len(getattr(obj, "_child_objects", []))
    if name == "Group":
        # because we represent RegionOfInterest only once
        # not the individual subclasses
        nlines -= 2
    return nlines * line_height


def annotate(ax, coord1, coord2, connectionstyle, color, alpha):
    arrowprops = dict(
        arrowstyle="simple",
        shrinkA=0.3,
        shrinkB=0.3,
        fc=color,
        ec=color,
        connectionstyle=connectionstyle,
        alpha=alpha,
    )
    bbox = dict(boxstyle="square", fc="w")
    a = ax.annotate(
        "",
        coord1,
        coord2,
        ha="right",
        va="center",
        size=fontsize,
        arrowprops=arrowprops,
        bbox=bbox,
    )
    a.set_zorder(-4)


def calc_coordinates(pos, height):
    x = pos[0]
    y = pos[1] + height - line_height * 0.5

    return x, y


def initialize_dummy_neo(name):
    if name in ["Block", "Segment", "Group", "Event", "Epoch"]:
        return getattr(neo, name)()
    elif name in ["ChannelView"]:
        sig = neo.AnalogSignal([] * pq.V, sampling_rate=1 * pq.Hz)
        return neo.ChannelView(sig, [0])
    elif name in ["ImageSequence"]:
        return neo.ImageSequence(
            np.array([1]).reshape((1, 1, 1)) * pq.V, spatial_scale=1 * pq.m, sampling_rate=1 * pq.Hz
        )
    elif name in ["AnalogSignal"]:
        return neo.AnalogSignal([] * pq.V, sampling_rate=1 * pq.Hz)
    elif name in ["SpikeTrain"]:
        return neo.SpikeTrain([] * pq.s, 0 * pq.s)
    elif name in ["IrregularlySampledSignal"]:
        return neo.IrregularlySampledSignal([] * pq.s, [] * pq.V)
    elif name in ["RegionOfInterest"]:
        imgseq = initialize_dummy_neo("ImageSequence")
        return neo.core.regionofinterest.CircularRegionOfInterest(imgseq, 0, 0, 0)
    else:
        raise ValueError(f"Unknown neo object: {name}")


def generate_diagram(rect_pos, rect_width, figsize):
    rw = rect_width

    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    all_h = {}
    objs = {}
    for name in rect_pos:
        objs[name] = initialize_dummy_neo(name)
        all_h[name] = get_rect_height(name, objs[name])

    # draw connections
    color = ["c", "m", "r"]
    alpha = [0.8, 0.8, 0.5]
    CHILD_OBJECTS = 0
    CHILD_PROPERTIES = 1
    CUSTOM_RELATIONS = 2
    for name, pos in rect_pos.items():
        obj = objs[name]
        relationships = [getattr(obj, "_child_objects", []), getattr(obj, "_child_properties", [])]

        # additional references
        custom_relations = []
        for attr in obj._all_attrs:
            attr_name, attr_type = attr[0], attr[1]
            if isinstance(attr_type, str) and hasattr(neo, attr_type):
                custom_relations.append(attr_type)
            elif isinstance(attr_type, tuple):
                for attr_t in attr_type:
                    if isinstance(attr_t, str) and hasattr(neo, attr_t):
                        custom_relations.append(attr_t)
        relationships.append(custom_relations)

        for r in (CHILD_OBJECTS, CHILD_PROPERTIES, CUSTOM_RELATIONS):
            for child_name in relationships[r]:
                if ("RegionOfInterest") in child_name:
                    child_name = "RegionOfInterest"
                if child_name not in rect_pos:
                    continue
                x1, y1 = calc_coordinates(rect_pos[child_name], all_h[child_name])
                x2, y2 = calc_coordinates(pos, all_h[name])

                # autolink
                if child_name == name:
                    rad = 1
                    y1 += 0.2
                    x1 += 0.1
                else:
                    rad = (y1 - y2) / 50

                if r == CUSTOM_RELATIONS:
                    rad = -0.1

                if r in (CHILD_OBJECTS, CHILD_PROPERTIES, CUSTOM_RELATIONS):
                    x2 += rect_width
                connectionstyle = f"arc3,rad={rad}"

                annotate(
                    ax=ax,
                    coord1=(x1 - 0.02, y1 - 0.05),
                    coord2=(x2 - 0.05, y2),
                    connectionstyle=connectionstyle,
                    color=color[r],
                    alpha=alpha[r],
                )

    # draw boxes
    for name, pos in rect_pos.items():
        htotal = all_h[name]
        obj = objs[name]
        allrelationship = []
        for item in getattr(obj, "_child_containers", []):
            if item not in allrelationship:
                # this is to avoid having "regionsofinterest" appear three times in Group
                # for the three subclasses
                allrelationship.append(item)

        rect = Rectangle(pos, rect_width, htotal, facecolor="w", edgecolor="k", linewidth=2.0)
        ax.add_patch(rect)

        # title green
        pos2 = pos[0], pos[1] + htotal - line_height * 1.5
        rect = Rectangle(pos2, rect_width, line_height * 1.5, facecolor="g", edgecolor="k", alpha=0.5, linewidth=2.0)
        ax.add_patch(rect)

        # single relationship
        relationship = list(getattr(obj, "_child_objects", []))
        if name == "Group":
            relationship.remove("RectangularRegionOfInterest")
            relationship.remove("PolygonRegionOfInterest")
        pos2 = pos[1] + htotal - line_height * (1.5 + len(relationship))
        rect_height = len(relationship) * line_height

        rect = Rectangle((pos[0], pos2), rect_width, rect_height, facecolor="c", edgecolor="k", alpha=0.5)
        ax.add_patch(rect)

        # necessary attr
        pos2 = pos[1] + htotal - line_height * (1.5 + len(allrelationship) + len(obj._necessary_attrs))
        rect = Rectangle(
            (pos[0], pos2), rect_width, line_height * len(obj._necessary_attrs), facecolor="r", edgecolor="k", alpha=0.5
        )
        ax.add_patch(rect)

        # name
        if hasattr(obj, "_quantity_attr"):
            post = "* "
        else:
            post = ""
        ax.text(
            pos[0] + rect_width / 2.0,
            pos[1] + htotal - line_height * 1.5 / 2.0,
            name + post,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fontsize + 2,
            fontproperties=FontProperties(weight="bold"),
        )

        # relationship
        for i, relat in enumerate(allrelationship):
            ax.text(
                pos[0] + left_text_shift,
                pos[1] + htotal - line_height * (i + 2),
                relat + ": list",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=fontsize,
            )
        # attributes
        for i, attr in enumerate(obj._all_attrs):
            attrname, attrtype = attr[0], attr[1]
            if hasattr(obj, "_quantity_attr") and obj._quantity_attr == attrname:
                t1 = attrname + " (object itself)"
            else:
                t1 = attrname

            if attrtype == pq.Quantity:
                if attr[2] == 0:
                    t2 = "Quantity scalar"
                else:
                    t2 = "Quantity %dD" % attr[2]
            elif attrtype == np.ndarray:
                t2 = "np.ndarray %dD dt='%s'" % (attr[2], attr[3].kind)
            elif attrtype == datetime:
                t2 = "datetime"
            elif type(attrtype) == tuple:
                t2 = str(attrtype).replace("(", "").replace(")", "")
                t2 = t2.replace("'", "").replace(", ", " or ")
            else:
                t2 = attrtype.__name__

            t = t1 + " :  " + t2

            # abbreviating lines to match in rectangles
            char_limit = int(rect_width * 13.4)  # 13.4 = arbitrary calibration
            if len(t) > char_limit:
                t = t[: char_limit - 3] + "..."
            ax.text(
                pos[0] + left_text_shift,
                pos[1] + htotal - line_height * (i + len(allrelationship) + 2),
                t,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=fontsize,
            )

    xlim, ylim = figsize
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def generate_diagram_simple():
    figsize = (18, 12)
    rw = rect_width = 3.0
    bf = 1.5
    rect_pos = {
        #  col 0
        "Block": (0.5 + rw * bf * 0, 7),
        #  col 1
        "Segment": (0.5 + rw * bf * 0.9, 8.5),
        "Group": (0.5 + rw * bf * 0.9, 3.7),
        #  col 2
        "ChannelView": (0.5 + rw * bf * 1.8, 2.5),
        "RegionOfInterest": (0.5 + rw * bf * 1.8, 1.0),
        # col 3
        "AnalogSignal": (0.5 + rw * bf * 3, 10),
        "IrregularlySampledSignal": (0.5 + rw * bf * 3, 8.4),
        "ImageSequence": (0.5 + rw * bf * 3, 6.35),
        "SpikeTrain": (0.5 + rw * bf * 3, 3.8),
        "Event": (0.5 + rw * bf * 3, 2.1),
        "Epoch": (0.5 + rw * bf * 3, 0.3),
    }

    fig = generate_diagram(rect_pos, rect_width, figsize)
    fig.savefig("simple_generated_diagram.png", dpi=dpi)
    fig.savefig("simple_generated_diagram.svg", dpi=dpi)


if __name__ == "__main__":
    generate_diagram_simple()
    pyplot.show()
