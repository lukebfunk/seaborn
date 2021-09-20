
import numpy as np
import pandas as pd
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize, to_rgb

import pytest
from numpy.testing import assert_array_equal

# TODO from seaborn._compat import MarkerStyle
from seaborn._compat import MarkerStyle
from seaborn._core.rules import categorical_order
from seaborn._core.scales import ScaleWrapper, CategoricalScale
from seaborn._core.mappings import (
    GroupMapping,
    ColorMapping,
    DashMapping,
    MarkerMapping,
)
from seaborn.palettes import color_palette


class TestGroupMapping:

    def test_levels(self):

        x = pd.Series(["a", "c", "b", "b", "d"])
        m = GroupMapping().setup(x)
        assert m.levels == categorical_order(x)


class TestColorMapping:

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

    @pytest.fixture
    def num_norm(self, num_vector):
        norm = Normalize()
        norm.autoscale(num_vector)
        return norm

    @pytest.fixture
    def cat_vector(self, long_df):
        return long_df["a"]

    @pytest.fixture
    def cat_order(self, cat_vector):
        return categorical_order(cat_vector)

    def test_categorical_default_palette(self, cat_vector, cat_order):

        expected = dict(zip(cat_order, color_palette()))
        m = ColorMapping().setup(cat_vector)

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_default_palette_large(self):

        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        n_colors = len(vector)
        expected = dict(zip(vector, color_palette("husl", n_colors)))
        m = ColorMapping().setup(vector)

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_named_palette(self, cat_vector, cat_order):

        palette = "Blues"
        m = ColorMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette
        assert m.levels == cat_order

        expected = dict(
            zip(cat_order, color_palette(palette, len(cat_order)))
        )

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_list_palette(self, cat_vector, cat_order):

        palette = color_palette("Reds", len(cat_order))
        m = ColorMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette

        expected = dict(zip(cat_order, palette))

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_implied_by_list_palette(self, num_vector, num_order):

        palette = color_palette("Reds", len(num_order))
        m = ColorMapping(palette=palette).setup(num_vector)
        assert m.palette == palette

        expected = dict(zip(num_order, palette))

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_dict_palette(self, cat_vector, cat_order):

        palette = dict(zip(cat_order, color_palette("Greens")))
        m = ColorMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette

        for level, color in palette.items():
            assert m(level) == color

    def test_categorical_implied_by_dict_palette(self, num_vector, num_order):

        palette = dict(zip(num_order, color_palette("Greens")))
        m = ColorMapping(palette=palette).setup(num_vector)
        assert m.palette == palette

        for level, color in palette.items():
            assert m(level) == color

    def test_categorical_dict_with_missing_keys(self, cat_vector, cat_order):

        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        with pytest.raises(ValueError):
            ColorMapping(palette=palette).setup(cat_vector)

    def test_categorical_list_with_wrong_length(self, cat_vector, cat_order):

        palette = color_palette("Oranges", len(cat_order) - 1)
        with pytest.raises(ValueError):
            ColorMapping(palette=palette).setup(cat_vector)

    def test_categorical_with_ordered_scale(self, cat_vector):

        cat_order = list(cat_vector.unique()[::-1])
        scale = ScaleWrapper(CategoricalScale(order=cat_order), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(cat_order))

        m = ColorMapping(palette=palette).setup(cat_vector, scale)
        assert m.levels == cat_order

        expected = dict(zip(cat_order, colors))

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_implied_by_scale(self, num_vector, num_order):

        scale = ScaleWrapper(CategoricalScale(), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(num_order))

        m = ColorMapping(palette=palette).setup(num_vector, scale)
        assert m.levels == num_order

        expected = dict(zip(num_order, colors))

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_implied_by_ordered_scale(self, num_vector):

        order = num_vector.unique()
        if order[0] < order[1]:
            order[[0, 1]] = order[[1, 0]]
        order = list(order)

        scale = ScaleWrapper(CategoricalScale(order=order), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(order))

        m = ColorMapping(palette=palette).setup(num_vector, scale)
        assert m.levels == order

        expected = dict(zip(order, colors))

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_with_ordered_categories(self, cat_vector, cat_order):

        new_order = list(reversed(cat_order))
        new_vector = cat_vector.astype("category").cat.set_categories(new_order)

        expected = dict(zip(new_order, color_palette()))

        m = ColorMapping().setup(new_vector)

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_implied_by_categories(self, num_vector):

        new_vector = num_vector.astype("category")
        new_order = categorical_order(new_vector)

        expected = dict(zip(new_order, color_palette()))

        m = ColorMapping().setup(new_vector)

        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_implied_by_palette(self, num_vector, num_order):

        palette = "bright"
        expected = dict(zip(num_order, color_palette(palette)))
        m = ColorMapping(palette=palette).setup(num_vector)
        for level, color in expected.items():
            assert m(level) == color

    def test_categorical_from_binary_data(self):

        vector = pd.Series([1, 0, 0, 0, 1, 1, 1])
        expected_palette = dict(zip([0, 1], color_palette()))
        m = ColorMapping().setup(vector)

        for level, color in expected_palette.items():
            assert m(level) == color

        first_color, *_ = color_palette()

        for val in [0, 1]:
            m = ColorMapping().setup(pd.Series([val] * 4))
            assert m(val) == first_color

    def test_categorical_multi_lookup(self):

        x = pd.Series(["a", "b", "c"])
        colors = color_palette(n_colors=len(x))
        m = ColorMapping().setup(x)
        assert_array_equal(m(x), np.stack(colors))

    def test_categorical_multi_lookup_categorical(self):

        x = pd.Series(["a", "b", "c"]).astype("category")
        colors = color_palette(n_colors=len(x))
        m = ColorMapping().setup(x)
        assert_array_equal(m(x), np.stack(colors))

    def test_numeric_default_palette(self, num_vector, num_order, num_norm):

        m = ColorMapping().setup(num_vector)
        expected_cmap = color_palette("ch:", as_cmap=True)
        for level in num_order:
            assert m(level) == to_rgb(expected_cmap(num_norm(level)))

    def test_numeric_named_palette(self, num_vector, num_order, num_norm):

        palette = "viridis"
        m = ColorMapping(palette=palette).setup(num_vector)
        expected_cmap = color_palette(palette, as_cmap=True)
        for level in num_order:
            assert m(level) == to_rgb(expected_cmap(num_norm(level)))

    def test_numeric_colormap_palette(self, num_vector, num_order, num_norm):

        cmap = color_palette("rocket", as_cmap=True)
        m = ColorMapping(palette=cmap).setup(num_vector)
        for level in num_order:
            assert m(level) == to_rgb(cmap(num_norm(level)))

    def test_numeric_norm_limits(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        cmap = color_palette("rocket", as_cmap=True)
        scale = ScaleWrapper(LinearScale("color"), "numeric", norm=lims)
        norm = Normalize(*lims)
        m = ColorMapping(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert m(level) == to_rgb(cmap(norm(level)))

    def test_numeric_norm_object(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        norm = Normalize(*lims)
        cmap = color_palette("rocket", as_cmap=True)
        scale = ScaleWrapper(LinearScale("color"), "numeric", norm=norm)
        m = ColorMapping(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert m(level) == to_rgb(cmap(norm(level)))

    def test_numeric_dict_palette_with_norm(self, num_vector, num_order, num_norm):

        palette = dict(zip(num_order, color_palette()))
        scale = ScaleWrapper(LinearScale("color"), "numeric", norm=num_norm)
        m = ColorMapping(palette=palette).setup(num_vector, scale)
        for level, color in palette.items():
            assert m(level) == to_rgb(color)

    def test_numeric_multi_lookup(self, num_vector, num_norm):

        cmap = color_palette("mako", as_cmap=True)
        m = ColorMapping(palette=cmap).setup(num_vector)
        expected_colors = cmap(num_norm(num_vector.to_numpy()))[:, :3]
        assert_array_equal(m(num_vector), expected_colors)

    def test_bad_palette(self, num_vector):

        with pytest.raises(ValueError):
            ColorMapping(palette="not_a_palette").setup(num_vector)

    def test_bad_norm(self, num_vector):

        norm = "not_a_norm"
        scale = ScaleWrapper(LinearScale("color"), "numeric", norm=norm)
        with pytest.raises(ValueError):
            ColorMapping().setup(num_vector, scale)


class TestDashMapping:

    def assert_dashes_equal(self, a, b):

        a = DashMapping._get_dash_pattern(a)
        b = DashMapping._get_dash_pattern(b)
        assert a == b

    def test_unique_dashes(self):

        n = 24
        dashes = DashMapping()._default_values(n)

        assert len(dashes) == n
        assert len(set(dashes)) == n

        assert dashes[0] == (0, None)
        for spec in dashes[1:]:
            assert isinstance(spec, tuple)
            assert spec[0] == 0
            assert not len(spec[1]) % 2

    def test_none_provided(self):

        keys = pd.Series(["a", "b", "c"])
        m = DashMapping().setup(keys)

        defaults = DashMapping()._default_values(len(keys))

        for key, want in zip(keys, defaults):
            self.assert_dashes_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(defaults)
        for have, want in zip(mapped, defaults):
            self.assert_dashes_equal(have, want)

    def test_provided_list(self):

        keys = pd.Series(["a", "b", "c", "d"])
        dashes = ["-", (1, 4), "dashed", (.5, (5, 2))]
        m = DashMapping(dashes).setup(keys)

        for key, want in zip(keys, dashes):
            self.assert_dashes_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(dashes)
        for have, want in zip(mapped, dashes):
            self.assert_dashes_equal(have, want)

    def test_provided_dict(self):

        keys = pd.Series(["a", "b", "c", "d"])
        values = ["-", (1, 4), "dashed", (.5, (5, 2))]
        dashes = dict(zip(keys, values))
        m = DashMapping(dashes).setup(keys)

        for key, want in dashes.items():
            self.assert_dashes_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(values)
        for have, want in zip(mapped, values):
            self.assert_dashes_equal(have, want)

    def test_provided_dict_with_missing(self):

        m = DashMapping({})
        keys = pd.Series(["a", 1])
        err = r"Missing dash pattern for following value\(s\): 1, 'a'"
        with pytest.raises(ValueError, match=err):
            m.setup(keys)

    def test_provided_list_too_short(self):

        m = DashMapping(["-", "dashed"])
        keys = pd.Series(["a", "b", "c"])
        msg = r"The dash pattern list has fewer values \(2\) than needed \(3\)"
        with pytest.warns(UserWarning, match=msg):
            m.setup(keys)


class TestMarkerMapping:

    def assert_markers_equal(self, a, b):

        a = MarkerStyle(a)
        b = MarkerStyle(b)
        assert a.get_path() == b.get_path()
        assert a.get_joinstyle() == b.get_joinstyle()
        assert a.get_transform().to_values() == b.get_transform().to_values()
        assert a.get_fillstyle() == b.get_fillstyle()

    def test_unique_markers(self):

        n = 24
        markers = MarkerMapping()._default_values(n)

        assert len(markers) == n
        assert len(set(
            (m.get_path(), m.get_joinstyle(), m.get_transform().to_values())
            for m in markers
        )) == n

        for m in markers:
            assert MarkerStyle(m).is_filled()

    def test_none_provided(self):

        keys = pd.Series(["a", "b", "c"])
        m = MarkerMapping().setup(keys)

        defaults = MarkerMapping()._default_values(len(keys))

        for key, want in zip(keys, defaults):
            self.assert_markers_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(defaults)
        for have, want in zip(mapped, defaults):
            self.assert_markers_equal(have, want)

    def test_provided_list(self):

        keys = pd.Series(["a", "b", "c"])
        markers = ["o", (5, 2, 0), MarkerStyle("o", fillstyle="none")]
        m = MarkerMapping(markers).setup(keys)

        for key, want in zip(keys, markers):
            self.assert_markers_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(markers)
        for have, want in zip(mapped, markers):
            self.assert_markers_equal(have, want)

    def test_provided_dict(self):

        keys = pd.Series(["a", "b", "c"])
        values = ["o", (5, 2, 0), MarkerStyle("o", fillstyle="none")]
        markers = dict(zip(keys, values))
        m = MarkerMapping(markers).setup(keys)

        for key, want in markers.items():
            self.assert_markers_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(values)
        for have, want in zip(mapped, values):
            self.assert_markers_equal(have, want)

    def test_provided_dict_with_missing(self):

        m = MarkerMapping({})
        keys = pd.Series(["a", 1])
        err = r"Missing marker for following value\(s\): 1, 'a'"
        with pytest.raises(ValueError, match=err):
            m.setup(keys)

    def test_provided_list_too_short(self):

        m = MarkerMapping(["o", "s"])
        keys = pd.Series(["a", "b", "c"])
        msg = r"The marker list has fewer values \(2\) than needed \(3\)"
        with pytest.warns(UserWarning, match=msg):
            m.setup(keys)
