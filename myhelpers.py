"""Helper functions for EDA with pandas and seaborn.
Tools, which are not connected with particular stack, live as just functions.
Tools for Pandas or Seabotn live in corresponding classes as staticmethods.
I am too lazy write documentation, type hinting and naming is enough :)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns

from IPython.display import display

from typing import Dict, Optional, Sequence, Tuple, List
from collections import defaultdict

def sqlite_info(connection):
    
    tables_query = '''
    --sql
    SELECT name, rootpage FROM sqlite_schema
    WHERE type = "table";
    '''

    tables = pd.read_sql(tables_query, connection)

    print('TABLES SUMMARY:')
    display(tables)

    for table in tables.name:

        query = f'''
        --sql
        SELECT COUNT(*) size
        FROM {table};
        '''

        df = pd.read_sql(query, connection)
        print(f'TABLE "{table.upper()}"')
        print(f'SIZE: {int(df.iloc[0])}')

        query = f'''
        --sql
        SELECT * FROM {table}
        LIMIT 5;
        '''

        df = pd.read_sql(query, connection)
        display(df)


def col_to_title(col_name: str) -> str:
    return col_name.replace('_', ' ').capitalize()

class PandasHelpers:

    @staticmethod
    def explore_df(dataframe: pd.DataFrame) -> None:
        display(pd.concat([dataframe.head(), dataframe.tail()]))
        display(dataframe.info())

    @staticmethod
    def full_display_rows(series: pd.Series, n: int = 5) -> None:
        
        count = 0
        for id, row in series.iteritems():
            count += 1
            if count > n:
                break
            print(f'{id}\t{row}')
        # Can I do it more efficiently?

class PandasSeriesAppliers:
    """Functions designed to apply to pd series"""

    @staticmethod
    # TODO Better mapper functionality (inversed etc)
    def multi_replace(
        row: str, mapper: Dict[str, Sequence[str]],
        default: Optional[str] = None
    ) -> str:

        if default is None:
            default = row

        for output_str, inputs in mapper.items():
            if row in inputs:
                return output_str
        
        return default

    def cat_to_num(
        row: str, order: List[str]
    ) -> float:
        return order.index(row) / (len(order) - 1)


class SeabornHelpers:

    @staticmethod
    def kde_boxen(
        dataframe, col_name,
        title: Optional[str] = None,
        figsize: Optional[Tuple[str, str]] = None
        # It's not good practice to "inherit" figsize without actual inheritance or composition
    ) -> None:

        if title is None:
            title = col_to_title(col_name)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(title)

        if figsize:
            x, y = figsize
            fig.set_figwidth(x)
            fig.set_figheight(y)
        
        sns.histplot(data=dataframe, x=col_name, ax=ax1, kde=True)
        sns.boxenplot(data=dataframe, x=col_name, ax=ax2, )

        sns.despine()
        plt.show()

