#!/usr/bin/env python3
"""Tests for the pandas-string DataFrame output / input fast paths.

Covers three optimizations:
  - PyUnicode dedup in PandasDataFrameBuilder string output
  - ArrowStringArray output when pandas >= 3
  - ArrowStringArray input zero-copy via Python(__df__) table function
"""

import os
import unittest
import shutil

import numpy as np
import pandas as pd

import chdb


PANDAS_MAJOR = int(pd.__version__.split(".")[0])


class TestStringOutput(unittest.TestCase):
    def setUp(self):
        self.dir = ".tmp_test_string_output_fast"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.session = chdb.session.Session(self.dir)

    def tearDown(self):
        self.session.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_low_cardinality_string_values_correct(self):
        sql = (
            "SELECT ['a','b','c','d'][1 + intDiv(number, 25) % 4] AS s "
            "FROM numbers(100)"
        )
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 100)
        self.assertEqual(df["s"].iloc[0], "a")
        self.assertEqual(df["s"].iloc[25], "b")
        self.assertEqual(df["s"].iloc[75], "d")
        self.assertEqual(sorted(df["s"].unique().tolist()), ["a", "b", "c", "d"])

    def test_unique_string_values_correct(self):
        sql = "SELECT lpad(toString(number), 6, '0') AS s FROM numbers(1000)"
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 1000)
        self.assertEqual(df["s"].iloc[0], "000000")
        self.assertEqual(df["s"].iloc[999], "000999")
        self.assertEqual(df["s"].nunique(), 1000)

    def test_string_with_nulls_values_correct(self):
        sql = (
            "SELECT if(number % 3 = 0, NULL, toString(number)) AS s "
            "FROM numbers(30)"
        )
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 30)
        self.assertTrue(pd.isna(df["s"].iloc[0]))
        self.assertEqual(df["s"].iloc[1], "1")
        self.assertEqual(df["s"].iloc[2], "2")
        self.assertTrue(pd.isna(df["s"].iloc[3]))

    def test_mixed_int_and_string_columns(self):
        sql = (
            "SELECT toInt64(number) AS n, lpad(toString(number), 4, '0') AS s "
            "FROM numbers(50)"
        )
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(list(df.columns), ["n", "s"])
        self.assertEqual(df["n"].iloc[10], 10)
        self.assertEqual(df["s"].iloc[10], "0010")
        self.assertEqual(df["n"].dtype.kind, "i")

    def test_empty_string_column(self):
        sql = "SELECT '' AS s FROM numbers(5)"
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 5)
        for i in range(5):
            self.assertEqual(df["s"].iloc[i], "")

    def test_utf8_multibyte_string(self):
        sql = "SELECT '中文' AS s FROM numbers(3)"
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 3)
        self.assertEqual(df["s"].iloc[0], "中文")

    @unittest.skipIf(PANDAS_MAJOR < 3, "Arrow string default needs pandas >= 3")
    def test_pandas3_string_dtype_is_str(self):
        df = self.session.query(
            "SELECT toString(number) AS s FROM numbers(10)", "DataFrame"
        )
        self.assertEqual(str(df["s"].dtype), "str")

    def test_large_string_column_correct(self):
        sql = "SELECT lpad(toString(number), 50, 'x') AS s FROM numbers(200)"
        df = self.session.query(sql, "DataFrame")
        self.assertEqual(len(df), 200)
        self.assertEqual(df["s"].iloc[0].count("x"), 49)
        self.assertEqual(len(df["s"].iloc[42]), 50)


class TestStringInputViaPythonTableFunction(unittest.TestCase):
    """The Python(__df__) table function path: pandas DataFrame → CH → result."""

    def setUp(self):
        self.dir = ".tmp_test_string_input_fast"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.session = chdb.session.Session(self.dir)

    def tearDown(self):
        self.session.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _query(self, sql, df):
        __df__ = df  # noqa: F841 — accessed by chdb's Python(__df__)
        return self.session.query(sql, "DataFrame")

    def test_pandas_string_column_round_trip(self):
        pdf = pd.DataFrame({"s": ["alpha", "beta", "gamma", "delta"]})
        out = self._query('SELECT "s" FROM Python(__df__) ORDER BY "s"', pdf)
        self.assertEqual(list(out["s"]), ["alpha", "beta", "delta", "gamma"])

    def test_pandas_string_with_filter_returns_correct_rows(self):
        pdf = pd.DataFrame(
            {"s": ["x", "y", "x", "z", "y"], "n": [1, 2, 3, 4, 5]}
        )
        out = self._query(
            'SELECT "s", "n" FROM Python(__df__) WHERE "s" = \'x\' ORDER BY "n"', pdf
        )
        self.assertEqual(list(out["s"]), ["x", "x"])
        self.assertEqual(list(out["n"]), [1, 3])

    @unittest.skipIf(PANDAS_MAJOR < 3, "Arrow string default needs pandas >= 3")
    def test_arrow_backed_pandas3_string_column(self):
        pdf = pd.DataFrame({"s": ["foo", "bar", "baz"] * 10})
        self.assertIn(str(pdf["s"].dtype), ("str", "string"))
        out = self._query(
            'SELECT count(*) AS c FROM Python(__df__) WHERE "s" = \'foo\'', pdf
        )
        self.assertEqual(int(out["c"].iloc[0]), 10)

    def test_pandas_string_aggregation_matches_pandas(self):
        pdf = pd.DataFrame(
            {"g": ["a", "b", "a", "b", "a"], "v": [10, 20, 30, 40, 50]}
        )
        out = self._query(
            'SELECT "g", sum("v") AS s FROM Python(__df__) GROUP BY "g" ORDER BY "g"',
            pdf,
        )
        expected = pdf.groupby("g")["v"].sum().reset_index()
        self.assertEqual(list(out["g"]), list(expected["g"]))
        self.assertEqual(list(out["s"]), list(expected["v"]))

    def test_pandas_string_empty_values(self):
        pdf = pd.DataFrame({"s": ["", "x", "", "y"]})
        out = self._query(
            'SELECT count() AS n FROM Python(__df__) WHERE "s" = \'\'', pdf
        )
        self.assertEqual(int(out["n"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
