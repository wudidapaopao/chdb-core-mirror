#include "PandasDataFrameBuilder.h"
#include "PythonImporter.h"
#include "NumpyType.h"
#include "QueryResult.h"

#include <DataTypes/Serializations/SerializationNullable.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeTime.h>
#include <DataTypes/DataTypeTime64.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <Common/DateLUTImpl.h>
#include <Processors/Chunk.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeDate.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnNullable.h>
#include <base/Decimal.h>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

}

using namespace DB;

namespace CHDB
{

namespace
{

/// pandas >= 3.0 stores str columns as ArrowStringArray by default. When that
/// is the case we hand pandas a pa.StringArray directly to avoid materializing
/// then re-encoding millions of PyUnicode objects.
bool PandasUsesArrowStringDefault()
{
    static const int result = []() -> int {
        try
        {
            py::object pd = py::module_::import("pandas");
            std::string version = pd.attr("__version__").cast<std::string>();
            int major = std::atoi(version.c_str());
            return major >= 3 ? 1 : 0;
        }
        catch (...)
        {
            return 0;
        }
    }();
    return result == 1;
}

bool ColumnIsString(const DataTypePtr & type)
{
    auto actual = removeLowCardinalityAndNullable(type);
    auto id = actual->getTypeId();
    return id == TypeIndex::String || id == TypeIndex::FixedString;
}

}

PandasDataFrameBuilder::PandasDataFrameBuilder(const ChunkQueryResult & chunk_result)
{
    chunks = std::move(const_cast<ChunkQueryResult &>(chunk_result).chunks);

    for (const auto & chunk : chunks)
        total_rows += chunk.getNumRows();

    if (!total_rows && !chunk_result.header)
    {
        finalize();
        return;
    }

    if (!chunk_result.header)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "ChunkQueryResult header is empty");

    const auto & sample = *chunk_result.header;
    column_names.reserve(sample.columns());
    column_types.reserve(sample.columns());

    std::unordered_map<String, size_t> name_map;

    for (const auto & column : sample)
    {
        const auto & col_name = column.name;
        String final_name = col_name;

        if (name_map.contains(col_name))
        {
            auto idx = name_map[col_name];
            final_name = col_name + "_" + std::to_string(idx);
            while (name_map.contains(final_name))
            {
                ++name_map[col_name];
                final_name = col_name + "_" + std::to_string(name_map[col_name]);
            }

            ++name_map[final_name];
        }
        else
        {
            name_map[col_name] = 1;
        }

        column_names.push_back(final_name);
        column_types.push_back(column.type);

        /// Record timezone for timezone-aware types
        auto actual_type = removeNullable(column.type);
        if (const auto * dt = typeid_cast<const DataTypeDateTime *>(actual_type.get()))
        {
            if (dt->hasExplicitTimeZone())
                column_timezones[final_name] = dt->getTimeZone().getTimeZone();
        }
        else if (const auto * dt64 = typeid_cast<const DataTypeDateTime64 *>(actual_type.get()))
        {
            if (dt64->hasExplicitTimeZone())
                column_timezones[final_name] = dt64->getTimeZone().getTimeZone();
        }
    }

    finalize();
}

py::object PandasDataFrameBuilder::genDataFrame(const py::handle & dict)
{
    auto & import_cache = PythonImporter::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas)
    {
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Pandas is not installed");
	}

	py::object items = dict.attr("items")();
	for (const py::handle & item : items) {
		auto key_value = py::cast<py::tuple>(item);
		py::handle key = key_value[0];
		py::handle value = key_value[1];

		if (py::isinstance(value, import_cache.numpy.ma.masked_array()))
        {
		    auto dtype = ConvertNumpyDtype(value);
			auto series = pandas.attr("Series")(value.attr("data"), py::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			dict.attr("__setitem__")(key, series);
		}
	}

	auto df = pandas.attr("DataFrame").attr("from_dict")(dict);

	/// Apply timezone conversion for timezone-aware columns
	changeToTZType(df);

	return df;
}

void PandasDataFrameBuilder::changeToTZType(py::object & df)
{
    if (column_timezones.empty())
        return;

    for (const auto & [column_name, timezone_str] : column_timezones)
    {
        /// Check if column exists in DataFrame
        if (!df.attr("__contains__")(column_name).cast<bool>())
            continue;

        /// Get the column
        auto column = df[column_name.c_str()];

        /// First localize to UTC (assuming the timestamps are in UTC)
        auto utc_localized = column.attr("dt").attr("tz_localize")("UTC");

        /// Then convert to the target timezone
        auto tz_converted = utc_localized.attr("dt").attr("tz_convert")(timezone_str);

        /// Update the column in DataFrame
        df.attr("__setitem__")(column_name.c_str(), tz_converted);
    }
}

py::object PandasDataFrameBuilder::buildArrowStringArray(size_t col_idx, bool nullable)
{
    /// chdb-core's ColumnString packs strings contiguously without null
    /// terminators (see src/Columns/ColumnString.h:41). offsets[i] is the end
    /// of string i in chars[]; size = offsets[i] - offsets[i-1].
    size_t total_bytes = 0;
    for (const auto & chunk : chunks)
    {
        const auto & cols = chunk.getColumns();
        const IColumn * c = cols[col_idx].get();
        if (const auto * n = typeid_cast<const ColumnNullable *>(c))
            c = &n->getNestedColumn();
        const auto * sc = typeid_cast<const ColumnString *>(c);
        if (!sc)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnString in Arrow string output path");
        total_bytes += sc->getChars().size();
    }

    if (total_bytes > std::numeric_limits<int32_t>::max())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "String column exceeds 2GiB; LargeString output not yet supported");

    /// Allocate Python bytes objects to back the Arrow buffers.
    PyObject * offsets_py = PyBytes_FromStringAndSize(nullptr, (total_rows + 1) * sizeof(int32_t));
    PyObject * data_py = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total_bytes));
    PyObject * validity_py = nullptr;
    if (nullable)
        validity_py = PyBytes_FromStringAndSize(nullptr, (total_rows + 7) / 8);

    auto * offsets = reinterpret_cast<int32_t *>(PyBytes_AsString(offsets_py));
    auto * data = reinterpret_cast<char *>(PyBytes_AsString(data_py));
    auto * validity = validity_py ? reinterpret_cast<uint8_t *>(PyBytes_AsString(validity_py)) : nullptr;
    if (validity)
        std::memset(validity, 0xFF, (total_rows + 7) / 8);

    size_t row = 0;
    size_t pos = 0;
    int64_t null_count = 0;
    offsets[0] = 0;
    for (const auto & chunk : chunks)
    {
        const auto & cols = chunk.getColumns();
        const IColumn * c = cols[col_idx].get();
        const ColumnNullable * n = nullable ? typeid_cast<const ColumnNullable *>(c) : nullptr;
        const IColumn * data_col = n ? &n->getNestedColumn() : c;
        const auto * sc = typeid_cast<const ColumnString *>(data_col);
        const auto & ch_chars = sc->getChars();
        const auto & ch_offs = sc->getOffsets();
        size_t n_rows = sc->size();
        for (size_t i = 0; i < n_rows; ++i, ++row)
        {
            if (n && n->isNullAt(i))
            {
                validity[row / 8] &= static_cast<uint8_t>(~(1u << (row % 8)));
                ++null_count;
                offsets[row + 1] = static_cast<int32_t>(pos);
                continue;
            }
            size_t start = (i == 0) ? 0 : ch_offs[i - 1];
            size_t end = ch_offs[i];
            size_t len = end - start;
            if (len)
                std::memcpy(data + pos, ch_chars.raw_data() + start, len);
            pos += len;
            offsets[row + 1] = static_cast<int32_t>(pos);
        }
    }
    chassert(row == total_rows);

    auto & cache = PythonImporter::ImportCache();
    auto pa_buffer = cache.pyarrow.py_buffer();
    auto pa_array = cache.pyarrow.Array();
    auto pa_string = cache.pyarrow.string_type();

    auto offsets_obj = py::reinterpret_steal<py::object>(offsets_py);
    auto data_obj = py::reinterpret_steal<py::object>(data_py);
    py::object validity_obj = py::none();
    if (validity_py)
        validity_obj = py::reinterpret_steal<py::object>(validity_py);

    py::object offsets_buf = pa_buffer(offsets_obj);
    py::object data_buf = pa_buffer(data_obj);
    py::object validity_buf = py::none();
    if (!validity_obj.is_none())
        validity_buf = pa_buffer(validity_obj);

    py::list buffers;
    buffers.append(validity_buf);
    buffers.append(offsets_buf);
    buffers.append(data_buf);

    py::object arr = pa_array.attr("from_buffers")(
        pa_string(),
        total_rows,
        buffers,
        py::arg("null_count") = null_count);

    auto pandas_mod = cache.pandas();
    return pandas_mod.attr("array")(arr, py::arg("dtype") = "str");
}

void PandasDataFrameBuilder::finalize()
{
    const auto types_size = column_types.size();
    const bool use_arrow_strings = PandasUsesArrowStringDefault();

    std::vector<bool> col_is_arrow_string(types_size, false);
    columns_data.reserve(types_size);

    for (size_t i = 0; i < types_size; ++i)
    {
        const auto & type = column_types[i];
        bool actual_string = ColumnIsString(type);
        bool nullable = type->isNullable();
        /// FixedString lacks a direct Arrow equivalent here; keep numpy path.
        auto raw_id = removeLowCardinalityAndNullable(type)->getTypeId();
        if (use_arrow_strings && actual_string && raw_id == TypeIndex::String)
            col_is_arrow_string[i] = true;

        columns_data.emplace_back(type);
        if (!col_is_arrow_string[i])
            columns_data[i].init(total_rows);
        (void)nullable;
    }

    for (const auto & chunk : chunks)
    {
        const auto & columns = chunk.getColumns();

        if (columns.size() != types_size)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Chunk column size not match");

        for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
        {
            if (col_is_arrow_string[col_idx])
                continue;
            columns_data[col_idx].append(columns[col_idx]);
        }
    }

    py::dict res;
    for (size_t col_idx = 0; col_idx < column_names.size(); ++col_idx)
    {
        const auto & name = column_names[col_idx];
        if (col_is_arrow_string[col_idx])
        {
            bool nullable = column_types[col_idx]->isNullable();
            res[name.c_str()] = buildArrowStringArray(col_idx, nullable);
        }
        else
        {
            res[name.c_str()] = columns_data[col_idx].toArray();
        }
    }

    chunks.clear();
    final_dataframe = genDataFrame(res);
}

py::object PandasDataFrameBuilder::getDataFrame()
{
    return std::move(final_dataframe);
}

}
