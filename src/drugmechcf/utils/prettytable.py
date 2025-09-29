"""
For simple pretty-printed tables.
None of the available packages seemed good enough or simple enough, e.g. Tabulate, Pandas, Asciitable.

Not yet supported: Fancy formats like HTML, LaTeX.
"""

from collections import Counter
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class PrettyTable:
    """
    Usage::

        from pubmed.utils.prettytable import PrettyTable

        ptbl = PrettyTable()

        ptbl.add_column('name 1', ['v1', 'v2', ...])
        ptbl.add_column('name 2', [..., ints, ...], format_spec=',d')
        ptbl.add_column('name 2', [..., floats, ...], format_spec='.2f')
        ... or ...
        ptbl.set_colnames(['name 1', 'name 2', 'name 3'], ['s', ',d', '.2f'])
        ptbl.add_row_('v1', 1, 1.1)
        ptbl.add_row_('v2', 2, 2.2)
        ptbl.add_row_('v3', 3, 3.3)

        print(ptbl)
        ... or ...
        ptbl.print(tsv_mode=True, file=file)

    Not an efficient implementation -- not intended for large tables.
    """
    def __init__(self, sep='  ', indent: int = 0, for_md=False):
        """

        :param sep: Separator to use between columns
        :param for_md: Is the output for MarkDown?
            IF True THEN output is formatted for MarkDown. This also changes the separator to " | ".
        """
        self.hdrs: List[str] = []
        self.cols: List[Any] = []
        self.formats: List[str] = []
        self.isnumeric: List[bool] = []

        self.underline_char = '-'

        self.colspace = sep
        self.indent = ' ' * indent

        self.print_for_markdown = for_md
        if self.print_for_markdown:
            self.colspace = " | "
        return

    def add_column(self, name: str, values: Sequence[Any], format_spec: str = None):
        """
        Add a column named `name` with `values`.

        :param str name: Column name.
        :param list, np.ndarray values: Sequence of values, any of which can be None.
        :param str format_spec: Format spec, e.g. '.3f'
        """
        assert isinstance(name, str)
        assert isinstance(values, (tuple, list)) or (isinstance(values, np.ndarray) and values.ndim == 1)

        if isinstance(values, np.ndarray):
            values = values.tolist()

        self.hdrs.append(name)
        self.cols.append(values)
        self.formats.append(format_spec)

        self.isnumeric.append(self._isnumeric(len(self.cols) - 1))
        return

    def set_colnames(self, names: Sequence[str], format_specs: Sequence[str] = None):
        """
        Set col names to `names`.
        :param names: List or Tuple of str.
        :param format_specs: List or Tuple of str.
        """
        assert isinstance(names, (tuple, list))
        assert format_specs is None or isinstance(format_specs, (tuple, list))
        self.hdrs = names
        self.formats = format_specs or list()
        return

    def add_row_(self, *values):
        return self.add_row(values)

    def add_row(self, values: Sequence[Any]):
        assert isinstance(values, (tuple, list))
        self._normalize_shape(len(values))
        for c, v in zip(self.cols, values):
            c.append(v)
        return

    def _normalize_shape(self, ncols):
        curr_ncols = len(self.cols)
        new_ncols = max(len(self.hdrs), curr_ncols, ncols)

        self.hdrs += ['col_' + str(i + 1) for i in range(len(self.hdrs), new_ncols)]

        if self.cols:
            nrows = max(len(c) for c in self.cols)
            self.cols = [c + [None] * (nrows - len(c)) for c in self.cols] + \
                        [[None for _ in range(nrows)] for _ in range(curr_ncols, new_ncols)]
        else:
            nrows = 0
            self.cols = [[] for _ in range(new_ncols)]
        return new_ncols, nrows

    def _normalize_formats(self, ncols=0):
        if not ncols:
            ncols, _ = self._normalize_shape(0)

        self.formats += list([None]) * (ncols - len(self.formats))
        self.isnumeric += [self._isnumeric(ci) for ci in range(len(self.isnumeric), ncols)]
        return

    def _isnumeric(self, ci):
        """
        Any reason to believe this column is numeric?
        """
        if self.formats[ci] is not None:
            return self.formats[ci][-1] in ['d', 'f', '%']
        # Check if any value is non-numeric (np.isreal(None) is True !)
        return any(v is not None and np.isreal(v) for v in self.cols[ci])
        # v = firstnot(lambda v_: v_ is None, self.cols[ci])   # Returns None if none found
        # return np.isreal(v) if v is not None else False     # np.isreal(None) = True !

    def _pp_val(self, ci, v):
        if v is None:
            return ''
        fspec = self.formats[ci]
        if not fspec or (fspec[-1] != "s" and isinstance(v, str)):
            fspec = '{}'
        else:
            fspec = '{:' + fspec + '}'
        return str.format(fspec, v)

    def _pp_valw(self, ci, ri, width=0):
        if ri == -2:
            vstr = self.hdrs[ci]
        elif ri == -1:
            vstr = self.underline_char * width
            if self.print_for_markdown and (self.isnumeric[ci] or self.formats[ci] == ">s"):
                # Ensure min width 3 for numeric cols, o/w table does not display correctly in mark-down
                if self.isnumeric[ci] and len(vstr) < 3:
                    vstr = self.underline_char * 3
                vstr = vstr[:-1] + ":"
        else:
            try:
                vstr = '' if ri >= len(self.cols[ci]) else self._pp_val(ci, self.cols[ci][ri])
            except TypeError as e:
                print(f"fspec = {self.formats[ci]}, v = {self.cols[ci][ri]}")
                raise e
        if self.isnumeric[ci] or self.formats[ci] == ">s":
            return vstr.rjust(width)
        else:
            return vstr.ljust(width)

    def __str__(self):
        if not self.hdrs:
            return ''

        self._normalize_formats()

        width = len(self.hdrs)
        height = max(len(c) for c in self.cols)

        colws = [max(len(self._pp_valw(ci, ri)) for ri in range(-2, len(self.cols[ci]))) for ci in range(width)]

        prefix = f"{self.indent}| " if self.print_for_markdown else self.indent
        suffix = " |" if self.print_for_markdown else ""

        return '\n'.join((prefix + self.colspace.join(self._pp_valw(ci, ri, colws[ci]) for ci in range(width)) + suffix)
                         for ri in range(-2, height)) + '\n'

    def print(self, tsv_mode=False, for_markdown=False, file=None):
        if tsv_mode:
            width = len(self.hdrs)
            height = max(len(c) for c in self.cols)
            print(*['\t'.join(self._pp_valw(ci, ri) for ci in range(width)) for ri in range(-2, height)],
                  sep='\n', file=file)
        elif for_markdown:
            prev_colspace, self.colspace = self.colspace, " | "
            prev_for_markdown, self.print_for_markdown = self.print_for_markdown, True

            print(self, file=file)

            self.colspace, self.print_for_markdown = prev_colspace, prev_for_markdown
        else:
            print(self, file=file)
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def firstnot(predicate, iterable):
    """First element in iterable that does not satisfy predicate."""
    try:
        return next(itertools.dropwhile(predicate, iterable))
    except StopIteration:
        return None


def pp_counts(col_headings: List[str],
              col_formats: List[str],
              *,
              rows: Sequence[Sequence[Any]],
              count_col_idx: int,
              pct_col_hdg: Optional[str] = "%Total",
              pct_format: str = ".1%",
              add_index_col: bool = True,
              total_count: int = 0,
              total_count_name: str = "Total",
              add_totals_row: bool = True,
              for_md: bool = True):
    """

    :param col_headings:
    :param col_formats:
    :param rows: Sequence of rows, each row is a Sequence of cols.
        Sequence could be np.ndarray, list, tuple
    :param count_col_idx: Index of the column that contains the counts
    :param add_index_col: Whether an index is added as the first col.
    :param total_count:
    :param add_totals_row:
    :param total_count_name:
    :param for_md:
    :param pct_col_hdg: Name of added col containing % of total count.
        Set to None for no col added
    :param pct_format:
    """
    col_headings = col_headings.copy()
    col_formats = col_formats.copy()

    try:
        kcol_idx = col_formats.index("s")
    except ValueError:
        kcol_idx = 0

    add_pct_col = pct_col_hdg is not None
    if add_pct_col:
        col_headings.append(pct_col_hdg)
        col_formats.append(pct_format)

    if (add_pct_col or add_totals_row) and total_count == 0:
        total_count = sum(row_[count_col_idx] for row_ in rows)

    if add_index_col:
        kcol_idx += 1
        count_col_idx += 1
        col_headings = ["#"] + col_headings
        col_formats = [",d"] + col_formats

    ptbl = PrettyTable(for_md=for_md)
    ptbl.set_colnames(col_headings, col_formats)

    sum_all = 0
    for i, row_ in enumerate(rows, start=1):
        row_ = list(row_)
        if add_index_col:
            row_ = [i] + row_
        if add_pct_col:
            row_ += [row_[count_col_idx] / total_count]
        sum_all += row_[count_col_idx]
        ptbl.add_row(row_)

    if add_totals_row:
        row_: List[Any] = [None] * len(col_headings)
        row_[kcol_idx] = "SUM"
        row_[count_col_idx] = sum_all
        if add_pct_col:
            row_[-1] = 1.0 if total_count == 0 else sum_all / total_count
        ptbl.add_row(row_)

    if add_pct_col:
        row_: List[Any] = [None] * len(col_headings)
        row_[kcol_idx] = total_count_name
        row_[count_col_idx] = total_count
        if add_pct_col:
            row_[-1] = 1.0
        ptbl.add_row(row_)

    print(ptbl)
    return


def pp_counts_and_pct(counts_dict: Dict[Any, int],
                      title: str,
                      col_headings: List[str],
                      total_count: int = 0,
                      keys_in_sorted_order: Union[bool, List[str]] = True,
                      counts_in_decreasing_order: bool = False,
                      add_totals_row=False,
                      for_md=True,
                      pct_format=".1%"):
    """
    Prints a table with 2 or 3 headings, e.g. "Key", "Count", "Percent of Total"

    :param counts_dict:
    :param title:
    :param col_headings: e.g. ['Key', 'Count'], or ['Key', 'Count', '%Total']
        Presence of 3rd col indicates add a %-of-Total column to table.
    :param total_count: IF Zero THEN total_count is computed as the sum from counts_dict

    :param keys_in_sorted_order: IF True THEN Keys to `counts_dict` will be printed in sorted order.
        Else IF a list of Keys, THEN output keys in provided order.

    :param counts_in_decreasing_order: IF True, and `keys_in_sorted_order` = False
        THEN entries are printed in decreasing Count

    :param add_totals_row: IF True THEN adds a bottom row of computed Totals
    :param for_md: Whether output is for Markdown
    :param pct_format: Number format for the %-age column

    :return:
    """
    assert 2 <= len(col_headings) <= 3

    print(title)
    print()

    first_key = list(counts_dict.keys())[0]
    if isinstance(first_key, int):
        index_fmt = ",d"
    else:
        index_fmt = "s"

    ptbl = PrettyTable(for_md=for_md)
    ptbl.set_colnames(col_headings, [index_fmt, ",d", pct_format][:len(col_headings)])

    if len(col_headings) == 3 and total_count == 0:
        total_count = sum(counts_dict.values())

    index_vals = counts_dict.keys()
    if keys_in_sorted_order:
        if isinstance(keys_in_sorted_order, List):
            index_vals = keys_in_sorted_order + sorted([k for k in index_vals if k not in keys_in_sorted_order])
        else:
            index_vals = sorted(index_vals)
    elif counts_in_decreasing_order:
        counts_dict = Counter(counts_dict)
        index_vals = [k for k, cnt in counts_dict.most_common()]

    total = 0
    for index in index_vals:
        cnt = counts_dict[index]
        total += cnt
        row = [index, cnt]
        if len(col_headings) == 3 and total_count > 0:
            row.append(cnt / total_count)
        ptbl.add_row(row)

    if add_totals_row:
        cnt = total if total_count == 0 else total_count
        row = ["TOTAL", cnt]
        if len(col_headings) == 3 and total_count > 0:
            row.append(cnt / total_count)
        ptbl.add_row(row)

    print(ptbl)
    return


def pp_seq_key_count(key_count_seq: Iterable[Tuple[Any, int]],
                     col_headings: Sequence[str] = ("Key", "Count"),
                     col_types: Sequence[str] = ("s", ",d"),
                     add_index=False,
                     total_count: int = 0,
                     add_total=False,
                     for_md=True):
    """

    :param key_count_seq: e.g. [ ('One', 1), ... ]
    :param col_headings:  e.g. [ 'Key', 'Count' ]
    :param col_types:     e.g. [ 's', ',d' ]
    :param add_index:
    :param add_total:
    :param total_count: IF > 0 THEN a 'pct' col is added
    :param for_md:
    """

    ptbl = PrettyTable(for_md=for_md)

    col_headings = list(col_headings)
    col_types = list(col_types)

    if add_index:
        col_headings = ['', *col_headings]
        col_types = [',d', *col_types]

    if total_count > 0:
        col_headings.append('%Total')
        col_types.append("5.1%")

    ptbl.set_colnames(col_headings, col_types)

    total = 0
    total_pct = 0
    for i, (key, count) in enumerate(key_count_seq, start=1):
        if add_index:
            row = [i, key, count]
        else:
            row = [key, count]

        if total_count > 0:
            pct = count / total_count
            total_pct += pct
            row.append(pct)

        ptbl.add_row(row)
        total += count

    if add_total:
        if add_index:
            row = [None, "ALL", total]
        else:
            row = ["ALL", total]
        if total_count > 0:
            row.append(total_pct)
        ptbl.add_row(row)

        if total_count > 0:
            if add_index:
                row = [None, "Total", total_count]
            else:
                row = ["Total", total_count]
            ptbl.add_row(row)

    print(ptbl)
    return


def pp_seq_key_dict(key_dict_seq: Iterable[Tuple[Any, Dict[str, Any]]],
                    col_headings: List[str],
                    col_types: List[str],
                    add_index=False,
                    for_md=True):
    """

    :param key_dict_seq:  [(Key, Vals-Dict), ...]
        The format is: [(Row-name, { Col-Heading-i => Col-Value-i, ... }), ...]
        e.g. [ ('One', { 'col1' => 1.1, 'col2' => '1b' }), ('Two', {...}), ... ]
    :param col_headings:  e.g. [ 'Key', 'col1', 'col2' ].
        col_headings[1:] are also keys in the Vals-Dict `key_dict[key]`
    :param col_types:     e.g. [ 's',   '.1f',  's' ]
    :param add_index:
    :param for_md:
    :return:
    """

    ptbl = PrettyTable(for_md=for_md)

    dict_cols = col_headings[1:]

    if add_index:
        col_headings = ['', *col_headings]
        col_types = [',d', *col_types]

    ptbl.set_colnames(col_headings, col_types)

    for i, (key, vals_dict) in enumerate(key_dict_seq, start=1):
        if add_index:
            ptbl.add_row_(i, key, *[vals_dict[col] for col in dict_cols])
        else:
            ptbl.add_row_(key, *[vals_dict[col] for col in dict_cols])

    print(ptbl)
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# To test, invoke as: python -m utils.prettytable

if __name__ == '__main__':

    ptbl_ = PrettyTable()
    ptbl_.add_column('Col 1', list(range(5)))
    ptbl_.add_column('Col 2', list(10 + v / 10 for v in range(5)), '.1f')
    print('2 cols x 5 rows:', ptbl_, sep='\n')
    ptbl_.add_row([6.1, 6.2, 6.3, 6.4])
    print('Added 2 empty cols, then 1 row:', ptbl_, sep='\n')
    print()
    print("TSV Mode:")
    ptbl_.print(tsv_mode=True)
