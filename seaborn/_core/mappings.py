from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb

from seaborn._compat import MarkerStyle
from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn.utils import get_color_cycle, remove_na
from seaborn.palettes import QUAL_PALETTES, color_palette

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Tuple
    from pandas import Series
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.scale import Scale
    from seaborn._core.typing import PaletteSpec

    DashPattern = Tuple[float, ...]
    DashPatternWithOffset = Tuple[float, Optional[DashPattern]]


# TODO I think we want map_semantic to accept _order/_norm parameters.
# But that forces some decisions:
# - Which takes precedence? (i.e. .map_ vs .scale_)?
# - Does Plot.map_ internally call self.scale_ or hand off to the Semantic?


class Semantic:
    ...


class BinarySemantic(Semantic):
    ...


class DiscreteSemantic(Semantic):

    def _default_values(self, n: int) -> list:
        """Return n unique values."""
        raise NotImplementedError

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> LookupMapping:

        provided = self._provided
        order = None if scale is None else scale.order
        levels = categorical_order(data, order)

        if provided is None:
            mapping = dict(zip(levels, self._default_values(len(levels))))
        # TODO generalize these input checks in Semantic
        elif isinstance(provided, dict):
            missing = set(data) - set(provided)
            if missing:
                formatted = ", ".join(map(repr, sorted(missing, key=str)))
                err = f"Missing {self._semantic} for following value(s): {formatted}"
                raise ValueError(err)
            mapping = provided
        elif isinstance(provided, list):
            if len(provided) > len(levels):
                msg = " ".join([
                    f"The {self._semantic} list has fewer values ({len(provided)})",
                    f"than needed ({len(levels)}) and will cycle, which may",
                    "produce an uninterpretable plot."
                ])
                warnings.warn(msg, UserWarning)
            mapping = dict(zip(levels, itertools.cycle(provided)))

        return LookupMapping(mapping)


class ContinuousSemantic(Semantic):

    norm: Normalize
    values: tuple[float, float]

    def __init__(
        self,
        values: tuple[float, float] | list[float] | dict[Any, float] | None,
    ):

        self._values = values

    def _infer_map_type(
        self,
        scale: Scale,
        provided: tuple[float, float] | list[float] | dict[Any, float] | None,
        data: Series,
    ) -> VarType:
        """Determine how to implement the mapping."""
        map_type: VarType
        if scale is not None:
            return scale.type
        else:
            map_type = variable_type(data, boolean_type="categorical")
        return map_type

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> NormedMapping:

        values = self._values
        # norm = None if scale is None else scale.norm
        # order = None if scale is None else scale.order
        map_type = self._infer_map_type(scale, values, data)  # TODO use norm/order?

        if map_type == "numeric":

            ...

        elif map_type == "categorical":

            ...

        elif map_type == "datetime":

            ...

    def _setup_categorical(data, values, order):
        ...


# ==================================================================================== #


class FillSemantic(BinarySemantic):
    ...


class ColorSemantic(Semantic):

    def __init__(self, palette: PaletteSpec = None):

        self._palette = palette

    def __call__(self, x):  # TODO types; will need to overload

        # TODO we are missing numeric maps and lots of other things
        if isinstance(x, pd.Series):
            if x.dtype.name == "category":  # TODO! possible pandas bug
                x = x.astype(object)
            # TODO where is best place to ensure that LUT values are rgba tuples?
            return np.stack(x.map(self.dictionary).map(to_rgb))
        else:
            return to_rgb(self.dictionary[x])

    def _infer_map_type(
        self,
        scale: Scale,
        palette: PaletteSpec,
        data: Series,
    ) -> VarType:
        """Determine how to implement the mapping."""
        map_type: VarType
        if scale is not None:
            return scale.type
        elif palette in QUAL_PALETTES:
            map_type = VarType("categorical")
        elif isinstance(palette, (dict, list)):
            map_type = VarType("categorical")
        else:
            map_type = variable_type(data, boolean_type="categorical")
        return map_type

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> ColorMapping:
        """Infer the type of mapping to use and define it using this vector of data."""
        palette: PaletteSpec = self._palette
        cmap: Colormap | None = None

        norm = None if scale is None else scale.norm
        order = None if scale is None else scale.order

        # TODO We need to add some input checks ...
        # e.g. specifying a numeric scale and a qualitative colormap should fail nicely.

        map_type = self._infer_map_type(scale, palette, data)

        if map_type == "categorical":

            # TODO what are we doing with levels now?
            levels, mapping = self._setup_categorical(
                data, palette, order,
            )
            return LookupMapping(mapping)

        elif map_type == "numeric":

            # TODO

            data = pd.to_numeric(data)
            levels, dictionary, norm, cmap = self._setup_numeric(
                data, palette, norm,
            )

        # --- Option 3: datetime mapping

        elif map_type == "datetime":
            # TODO this needs actual implementation
            cmap = norm = None
            levels, dictionary = self._setup_categorical(
                # Casting data to list to handle differences in the way
                # pandas and numpy represent datetime64 data
                list(data), palette, order,
            )

        # TODO do we need to return and assign out here or can the
        # type-specific methods do the assignment internally

        # TODO I don't love how this is kind of a mish-mash of attributes
        # Can we be more consistent across SemanticMapping subclasses?
        self.dictionary = dictionary
        self.palette = palette
        self.levels = levels
        self.norm = norm
        self.cmap = cmap

        return self

    def _setup_categorical(
        self,
        data: Series,
        palette: PaletteSpec,
        order: list | None,
    ) -> tuple[list, dict]:
        """Determine colors when the mapping is categorical."""
        # -- Identify the order and name of the levels

        levels = categorical_order(data, order)
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            dictionary = palette

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)  # TODO downgrade this to a warning?
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            dictionary = dict(zip(levels, colors))

        return levels, dictionary


class MarkerSemantic(DiscreteSemantic):

    # TODO This may have to be a parameters? (e.g. for color/edgecolor)
    _semantic = "marker"

    def __init__(self, shapes: list | dict | None = None):  # TODO full types

        # TODO fill or filled parameter?
        # allow singletons? e.g. map_marker(shapes="o", filled=[True, False])?
        # allow full matplotlib fillstyle API?

        if isinstance(shapes, list):
            shapes = [MarkerStyle(s) for s in shapes]
        elif isinstance(shapes, dict):
            shapes = {k: MarkerStyle(v) for k, v in shapes.items()}

        self._provided = shapes

    def _default_values(self, n):  # TODO or have this as an infinite generator?
        """Build an arbitrarily long list of unique marker styles for points.

        Parameters
        ----------
        n : int
            Number of unique marker specs to generate.

        Returns
        -------
        markers : list of string or tuples
            Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
            All markers will be filled.

        """
        # Start with marker specs that are well distinguishable
        markers = [
            "o",
            "X",
            (4, 0, 45),
            "P",
            (4, 0, 0),
            (4, 1, 0),
            "^",
            (4, 1, 45),
            "v",
        ]

        # Now generate more from regular polygons of increasing order
        s = 5
        while len(markers) < n:
            a = 360 / (s + 1) / 2
            markers.extend([
                (s + 1, 1, a),
                (s + 1, 0, a),
                (s, 1, 0),
                (s, 0, 0),
            ])
            s += 1

        # TODO use filled (maybe have different defaults depending on fill/nofill?)
        markers = [MarkerStyle(m) for m in markers]

        return markers[:n]


class DashSemantic(DiscreteSemantic):

    _semantic = "dash pattern"

    def __init__(self, styles: list | dict | None = None):  # TODO full types

        # TODO fill or filled parameter?
        # allow singletons? e.g. map_marker(shapes="o", filled=[True, False])?
        # allow full matplotlib fillstyle API?

        if isinstance(styles, list):
            styles = [self._get_dash_pattern(s) for s in styles]
        elif isinstance(styles, dict):
            styles = {k: self._get_dash_pattern(v) for k, v in styles.items()}

        self._provided = styles

    def _default_values(self, n: int) -> list[DashPatternWithOffset]:
        """Build an arbitrarily long list of unique dash styles for lines.

        Parameters
        ----------
        n : int
            Number of unique dash specs to generate.

        Returns
        -------
        dashes : list of strings or tuples
            Valid arguments for the ``dashes`` parameter on
            :class:`matplotlib.lines.Line2D`. The first spec is a solid
            line (``""``), the remainder are sequences of long and short
            dashes.

        """
        # Start with dash specs that are well distinguishable
        dashes: list[str | DashPattern] = [
            "-",  # TODO do we need to handle this elsewhere for backcompat?
            (4, 1.5),
            (1, 1),
            (3, 1.25, 1.5, 1.25),
            (5, 1, 1, 1),
        ]

        # Now programmatically build as many as we need
        p = 3
        while len(dashes) < n:

            # Take combinations of long and short dashes
            a = itertools.combinations_with_replacement([3, 1.25], p)
            b = itertools.combinations_with_replacement([4, 1], p)

            # Interleave the combinations, reversing one of the streams
            segment_list = itertools.chain(*zip(
                list(a)[1:-1][::-1],
                list(b)[1:-1]
            ))

            # Now insert the gaps
            for segments in segment_list:
                gap = min(segments)
                spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
                dashes.append(spec)

            p += 1

        return [self._get_dash_pattern(d) for d in dashes[:n]]

    @staticmethod
    def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
        """Convert linestyle to dash pattern."""
        # Copied and modified from Matplotlib 3.4
        # go from short hand -> full strings
        ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
        if isinstance(style, str):
            style = ls_mapper.get(style, style)
            # un-dashed styles
            if style in ['solid', 'none', 'None']:
                offset = 0
                dashes = None
            # dashed styles
            elif style in ['dashed', 'dashdot', 'dotted']:
                offset = 0
                dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])

        elif isinstance(style, tuple):
            if len(style) > 1 and isinstance(style[1], tuple):
                offset, dashes = style
            elif len(style) > 1 and style[1] is None:
                offset, dashes = style
            else:
                offset = 0
                dashes = style
        else:
            raise ValueError(f'Unrecognized linestyle: {style}')

        # normalize offset to be positive and shorter than the dash cycle
        if dashes is not None:
            dsum = sum(dashes)
            if dsum:
                offset %= dsum

        return offset, dashes


class AreaSemantic(ContinuousSemantic):
    ...


# ==================================================================================== #

class SemanticMapping:
    ...


class LookupMapping(SemanticMapping):

    def __init__(self, mapping: dict):

        self.mapping = mapping

    def __call__(self, x: Any) -> Any:  # Possibly to type output based on lookup_table?

        if isinstance(x, pd.Series):
            if x.dtype.name == "category":
                # https://github.com/pandas-dev/pandas/issues/41669
                x = x.astype(object)
            return x.map(self.mapping)
        else:
            return self.mapping[x]


class NormedMapping(SemanticMapping):

    def __init__(self, norm: Normalize, transform: Callable[float, Any]):

        self.norm = norm
        self.transform = transform

    def __call__(self, x: Any) -> Any:

        # TODO can we work out whether transform is vectorized and use it that way?
        # (Or ensure that it is, since we control it?)
        # TODO note that matplotlib Normalize is going to return a masked array
        # maybe this is fine since we're handing the output off to matplotlib?
        return self.transform(self.norm(x))

# ==================================================================================== #


class SemanticMapping:
    """Base class for mappings between data and visual attributes."""

    levels: list  # TODO Alternately, use keys of dictionary?

    def setup(self, data: Series, scale: Scale | None) -> SemanticMapping:
        # TODO why not just implement the GroupMapping setup() here?
        raise NotImplementedError()

    def __call__(self, x):  # TODO types; will need to overload (wheee)
        # TODO this is a hack to get things working
        if isinstance(x, pd.Series):
            if x.dtype.name == "category":  # TODO! possible pandas bug
                x = x.astype(object)
            # TODO where is best place to ensure that LUT values are rgba tuples?
            # TODO may need to move below line to ColorMapping
            # return np.stack(x.map(self.dictionary))
            return x.map(self.dictionary)
        else:
            return self.dictionary[x]


# TODO Currently, the SemanticMapping objects are also the source of the information
# about the levels/order of the semantic variables. Do we want to decouple that?

# TODO Perhaps the setup method should not add attributes and return self, but rather
# return an object that is initialized to do the mapping. This would make it so that
# Plotter._setup_mappings() won't mutate attributes on the Plot that generated it.
# Think about this a bit more but I think it's the way forward. Decrease state!
# Also if __init__ is just going to store information, can we abstract that in
# a nice way while also having a method with a signature/docstring we can use to
# attach map_{semantic} methods to Plot?

class GroupMapping(SemanticMapping):  # TODO only needed for levels...
    """Mapping that does not alter any visual properties of the artists."""
    def setup(self, data: Series, scale: Scale | None = None) -> GroupMapping:
        self.levels = categorical_order(data)
        return self


# Alt name RGBAMapping
class ColorMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""

    # TODO type the important class attributes here

    def _setup_numeric(
        self,
        data: Series,
        palette: PaletteSpec,
        norm: Normalize | None,
    ) -> tuple[list, dict, Normalize | None, Colormap]:
        """Determine colors when the variable is quantitative."""
        cmap: Colormap
        if isinstance(palette, dict):

            # In the function interface, the presence of a norm object overrides
            # a dictionary of colors to specify a numeric mapping, so we need
            # to process it here.
            # TODO this functionality only exists to support the old relplot
            # hack for linking hue orders across facets.  We don't need that any
            # more and should probably remove this, but needs deprecation.
            # (Also what should new behavior be? I think an error probably).
            levels = list(sorted(palette))
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            dictionary = palette.copy()

        else:

            # The levels are the sorted unique values in the data
            levels = list(np.sort(remove_na(data.unique())))

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # TODO do we want to do something complicated to ensure contrast?
            palette = "ch:" if palette is None else palette

            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)

            # Now sort out the data normalization
            # TODO consolidate in ScaleWrapper so we always have a norm here?
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "`norm` must be None, tuple, or Normalize object."
                raise ValueError(err)
            norm.autoscale_None(data.dropna())

            dictionary = dict(zip(levels, cmap(norm(levels))))

        return levels, dictionary, norm, cmap
