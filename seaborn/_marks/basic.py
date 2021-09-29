from __future__ import annotations
import numpy as np
from .base import Mark


class Point(Mark):

    supports = ["color"]

    def __init__(self, jitter=None, **kwargs):

        super().__init__(**kwargs)
        self.jitter = jitter  # TODO decide on form of jitter and add type hinting

    def _adjust(self, df):

        if self.jitter is None:
            return df

        x, y = self.jitter  # TODO maybe not format, and do better error handling

        # TODO maybe accept a Jitter class so we can control things like distribution?
        # If we do that, should we allow convenient flexibility (i.e. (x, y) tuple)
        # in the object interface, or be simpler but more verbose?

        # TODO note that some marks will have multiple adjustments
        # (e.g. strip plot has both dodging and jittering)

        # TODO native scale of jitter? maybe just for a Strip subclass?

        rng = np.random.default_rng()  # TODO seed?

        n = len(df)
        x_jitter = 0 if not x else rng.uniform(-x, +x, n)
        y_jitter = 0 if not y else rng.uniform(-y, +y, n)

        # TODO: this fails if x or y are paired. Apply to all columns that start with y?
        return df.assign(x=df["x"] + x_jitter, y=df["y"] + y_jitter)

    def _plot_split(self, keys, data, ax, mappings, kws):

        # TODO can we simplify this by modifying data with mappings before sending in?
        # Likewise, will we need to know `keys` here? Elsewhere we do `if key in keys`,
        # but I think we can (or can make it so we can) just do `if key in data`.

        # Then the signature could be _plot_split(ax, data, kws):  ... much simpler!

        # TODO since names match, can probably be automated!
        if "color" in data:
            c = mappings["color"](data["color"])
        else:
            # TODO prevents passing in c. But do we want to permit that?
            # I think if we implement map_color("identity"), then no
            c = None

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        points = ax.scatter(x=data["x"], y=data["y"], c=c, **kws)

        # TODO what to do about color= for marks that use facecolor and edgecolor?
        # a) defer to matplotlib?
        # b) implement matplotlib's (or matplotlib-like) rules in seaborn?
        # c) make color= set both face and edge color?
        # d) don't support color in those marks
        # e) not have a facecolor semantic, always use color
        # TODO also fix this whole nasty logic
        for var in ["facecolor", "edgecolor"]:
            getter = getattr(points, f"get_{var}")
            setter = getattr(points, f"set_{var}")
            if var in data:
                setter(mappings[var](data[var]))
            else:
                val, *others = getter()
                if not others:
                    val = [val] * len(data["x"])
                setter(val)  # TODO always n?

        if "marker" in data:
            markers = mappings["marker"](data["marker"])
            paths = [m.get_path().transformed(m.get_transform()) for m in markers]
            points.set_paths(paths)

        # TODO implement fill through facecolor or by setting marker fillstyle?
        # (Note that alpha=0 doesn't work on all backends)
        # TODO note also we need to deal with edges being invisible by default,
        # or alternately using the (non-colored) by default
        if "fill" in data:
            state = mappings["fill"](data["fill"])
            fc = points.get_facecolors()
            fc[~state] = (0, 0, 0, 0)
            points.set_facecolors(fc)

        # TODO note that when scaling this up we'll need to catch the artist
        # and update its attributes (like in scatterplot) to allow marker variation


class Line(Mark):

    # TODO how to handle distinction between stat groupers and plot groupers?
    # i.e. Line needs to aggregate by x, but not plot by it
    # also how will this get parametrized to support orient=?
    # TODO will this sort by the orient dimension like lineplot currently does?
    grouping_vars = ["color", "marker", "dash"]
    supports = ["color"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "color" in keys:
            kws["color"] = mappings["color"](keys["color"])
        if "dash" in keys:
            kws["linestyle"] = mappings["dash"](keys["dash"])

        ax.plot(data["x"], data["y"], **kws)


class Area(Mark):

    grouping_vars = ["color"]
    supports = ["color"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "color" in keys:
            # TODO as we need the kwarg to be facecolor, that should be the mappable?
            kws["facecolor"] = mappings["color"](keys["color"])

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if self.orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
