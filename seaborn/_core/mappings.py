from __future__ import annotations
from copy import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize, to_rgb

from seaborn._compat import MarkerStyle
from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn.utils import get_color_cycle
from seaborn.palettes import QUAL_PALETTES, color_palette

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Tuple
    from pandas import Series
    from matplotlib.colors import Colormap
    from matplotlib.scale import Scale
    from seaborn._core.typing import PaletteSpec

    DashPattern = Tuple[float, ...]
    DashPatternWithOffset = Tuple[float, Optional[DashPattern]]


class IdentityTransform:

    def __call__(self, x):
        return x


class RangeTransform:

    def __init__(self, lo: float, hi: float):
        self.out_range = lo, hi

    def __call__(self, x: float) -> float:
        lo, hi = self.out_range
        return lo + x * (hi - lo)


class Semantic:

    variable: str

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> SemanticMapping:

        raise NotImplementedError()

    def _check_dict_not_missing_levels(self, levels: list, values: dict) -> None:

        missing = set(levels) - set(values)
        if missing:
            formatted = ", ".join(map(repr, sorted(missing, key=str)))
            err = f"Missing {self.variable} for following value(s): {formatted}"
            raise ValueError(err)

    def _ensure_list_not_too_short(self, levels: list, values: list) -> list:

        if len(levels) > len(values):
            msg = " ".join([
                f"The {self.variable} list has fewer values ({len(values)})",
                f"than needed ({len(levels)}) and will cycle, which may",
                "produce an uninterpretable plot."
            ])
            warnings.warn(msg, UserWarning)

            values = [x for _, x in zip(levels, itertools.cycle(values))]

        return values


class DiscreteSemantic(Semantic):

    _provided: list | dict | None

    def __init__(self, values: list | dict | None = None, variable: str = "value"):

        self._provided = values
        self.variable = variable

    def _default_values(self, n: int) -> list:
        """Return n unique values."""
        raise NotImplementedError

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> LookupMapping:

        values = self._provided
        order = None if scale is None else scale.order
        levels = categorical_order(data, order)

        if values is None:
            mapping = dict(zip(levels, self._default_values(len(levels))))
        elif isinstance(values, dict):
            self._check_dict_not_missing_levels(levels, values)
            mapping = values
        elif isinstance(values, list):
            values = self._ensure_list_not_too_short(levels, values)
            mapping = dict(zip(levels, values))

        return LookupMapping(mapping)


class BooleanSemantic(DiscreteSemantic):

    def _default_values(self, n: int) -> list:
        return [x for x, _ in zip(itertools.cycle([True, False]), range(n))]

    # TODO Should we have some generalied way of doing input checking?


class ContinuousSemantic(Semantic):

    norm: Normalize
    transform: Callable  # TODO sort out argument typing in a way that satisfies mypy
    default_range: tuple[float, float] = 0, 1

    def __init__(
        self,
        values: tuple[float, float] | list[float] | dict[Any, float] | None = None,
        variable: str = "",  # TODO default?
    ):

        self._values = values
        self.variable = variable

    def _infer_map_type(
        self,
        scale: Scale,
        values: tuple[float, float] | list[float] | dict[Any, float] | None,
        data: Series,
    ) -> VarType:
        """Determine how to implement the mapping."""
        map_type: VarType
        if scale is not None:
            return scale.type
        # TODO list/dict values imply categorical
        # TODO presence of norm implies numeric
        else:
            map_type = variable_type(data, boolean_type="categorical")
        return map_type

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> NormedMapping | LookupMapping:

        values = self.default_range if self._values is None else self._values
        order = None if scale is None else scale.order
        levels = categorical_order(data, order)
        norm = Normalize() if scale is None or scale.norm is None else copy(scale.norm)
        map_type = self._infer_map_type(scale, values, data)

        # TODO check inputs ... what if scale.type is numeric but we got a list or dict?

        # TODO how to handle values as tuple? (where they indicate output range?)

        mapping: NormedMapping | LookupMapping

        transform = RangeTransform(*values)

        if map_type == "numeric":

            if isinstance(norm, tuple):
                norm = Normalize(*norm)

            if not norm.scaled():
                # Initialize auto-limits
                norm(np.asarray(data.dropna()))

            mapping = NormedMapping(norm, transform)

        elif map_type == "categorical":

            if values is None:
                # Go from large to small so first category appears most important
                numbers = np.linspace(1, 0, len(levels))
                values = transform(numbers)
                mapping_dict = dict(zip(levels, values))
            elif isinstance(values, tuple):
                # TODO even spacing between these values, large to small?
                raise NotImplementedError()
            elif isinstance(values, dict):
                self._check_dict_not_missing_levels(levels, values)
                mapping_dict = values
            elif isinstance(values, list):
                values = self._ensure_list_not_too_short(levels, values)
                # TODO check list not too long as well?
                mapping_dict = dict(zip(levels, values))

            mapping = LookupMapping(mapping_dict)

        elif map_type == "datetime":

            # TODO; needs implementation
            raise NotImplementedError()

        return mapping


# ==================================================================================== #


class ColorSemantic(Semantic):

    def __init__(self, palette: PaletteSpec = None, variable: str = "color"):

        self._palette = palette
        self.variable = variable

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> LookupMapping | NormedMapping:
        """Infer the type of mapping to use and define it using this vector of data."""
        mapping: LookupMapping | NormedMapping
        palette: PaletteSpec = self._palette

        norm = None if scale is None else scale.norm
        order = None if scale is None else scale.order

        # TODO We need to add some input checks ...
        # e.g. specifying a numeric scale and a qualitative colormap should fail nicely.

        map_type = self._infer_map_type(scale, palette, data)

        if map_type == "categorical":

            mapping = LookupMapping(self._setup_categorical(data, palette, order))

        elif map_type == "numeric":

            data = pd.to_numeric(data)
            lookup, norm, transform = self._setup_numeric(data, palette, norm)
            if lookup:
                # TODO See comments in _setup_numeric about deprecation of this
                mapping = LookupMapping(lookup)
            else:
                mapping = NormedMapping(norm, transform)

        elif map_type == "datetime":
            # TODO this needs actual implementation
            mapping = LookupMapping(self._setup_categorical(data, palette, order))

        return mapping

    def _setup_categorical(
        self,
        data: Series,
        palette: PaletteSpec,
        order: list | None,
    ) -> dict[Any, tuple[float, float, float]]:
        """Determine colors when the mapping is categorical."""
        levels = categorical_order(data, order)
        n_colors = len(levels)

        if isinstance(palette, dict):
            self._check_dict_not_missing_levels(levels, palette)
            mapping = palette
        else:
            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    # None uses current (global) default palette
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                colors = self._ensure_list_not_too_short(levels, palette)
                # TODO check not too long also?
            else:
                colors = color_palette(palette, n_colors)
            mapping = dict(zip(levels, colors))

        return mapping

    def _setup_numeric(
        self,
        data: Series,
        palette: PaletteSpec,
        norm: Normalize | None,
    ) -> tuple[dict[Any, tuple[float, float, float]], Normalize, Callable]:
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
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            mapping = palette.copy()

        else:

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # TODO do we want to do something complicated to ensure contrast?
            palette = "ch:" if palette is None else palette

            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)

            # Now sort out the data normalization
            # TODO consolidate in ScaleWrapper so we always have a Normalize here?
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "`norm` must be None, tuple, or Normalize object."
                raise ValueError(err)
            norm.autoscale_None(data.dropna())
            mapping = {}

        def rgb_transform(x):
            rgba = cmap(x)
            # TODO we should have general vectorized to_rgb/to_rgba
            if isinstance(rgba, tuple):
                return to_rgb(rgba)
            else:
                return rgba[..., :3]

        return mapping, norm, rgb_transform

    def _infer_map_type(
        self,
        scale: Scale,
        palette: PaletteSpec,
        data: Series,
    ) -> VarType:
        """Determine how to implement a color mapping."""
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


class MarkerSemantic(DiscreteSemantic):

    # TODO full types
    def __init__(self, shapes: list | dict | None = None, variable: str = "marker"):

        # TODO fill or filled parameter?
        # allow singletons? e.g. map_marker(shapes="o", filled=[True, False])?
        # allow full matplotlib fillstyle API?

        if isinstance(shapes, list):
            shapes = [MarkerStyle(s) for s in shapes]
        elif isinstance(shapes, dict):
            shapes = {k: MarkerStyle(v) for k, v in shapes.items()}

        self._provided = shapes
        self.variable = variable

    def _default_values(self, n: int) -> list[MarkerStyle]:
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

        # TODO or have this as an infinite generator?
        return markers[:n]


# TODO or linestyle?
class DashSemantic(DiscreteSemantic):

    def __init__(
        self,
        styles: list | dict | None = None,
        variable: str = "dash pattern"
    ):
        # TODO full types

        if isinstance(styles, list):
            styles = [self._get_dash_pattern(s) for s in styles]
        elif isinstance(styles, dict):
            styles = {k: self._get_dash_pattern(v) for k, v in styles.items()}

        self._provided = styles
        self.variable = variable

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


# TODO or pattern?
class HatchSemantic(DiscreteSemantic):
    ...


# TODO allow subclass to define validation function for values?


class AreaSemantic(ContinuousSemantic):
    ...


class WidthSemantic(ContinuousSemantic):
    default_range: tuple[float, float] = .2, .8


class LineWidthSemantic(ContinuousSemantic):
    # TODO scale rcParam by default (.5, 2), use property to do it
    # TODO Plot should probably cache the rcParams at the time it's set up...
    default_range: tuple[float, float] = .5, 2


class EdgeWidthSemantic(ContinuousSemantic):
    ...


# TODO or opacity?
class AlphaSemantic(ContinuousSemantic):
    ...


# ==================================================================================== #

class SemanticMapping:
    ...


class LookupMapping(SemanticMapping):

    def __init__(self, mapping: dict):

        self.mapping = mapping

    def __call__(self, x: Any) -> Any:  # Possible to type output based on lookup_table?

        if isinstance(x, pd.Series):
            if x.dtype.name == "category":
                # https://github.com/pandas-dev/pandas/issues/41669
                x = x.astype(object)
            return x.map(self.mapping)
        else:
            return self.mapping[x]


class NormedMapping(SemanticMapping):

    def __init__(self, norm: Normalize, transform: Callable[[float], Any]):

        self.norm = norm
        self.transform = transform

    def __call__(self, x: Any) -> Any:

        # TODO can we work out whether transform is vectorized and use it that way?
        # (Or ensure that it is, since we control it?)
        # TODO note that matplotlib Normalize is going to return a masked array
        # maybe this is fine since we're handing the output off to matplotlib?
        return self.transform(self.norm(x))
