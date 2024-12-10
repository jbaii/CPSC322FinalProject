from mysklearn import myutils
import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        
        col_index = self.column_names.index(col_identifier) if isinstance(col_identifier, str) else col_identifier
        if include_missing_values:
            return [row[col_index] for row in self.data]
        else:
            return [row[col_index] for row in self.data if row[col_index] != "NA"]
    


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    row[i] = float(row[i])
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        self.data = [row for i, row in enumerate(self.data) if i not in row_indexes_to_drop]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        with open(filename, "r", encoding = "utf8") as infile:
            reader = csv.reader(infile)
            for row in reader:
        #print(row)
                table.append(row)
        self.column_names = table[0]
        self.data = table[1:]
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w", newline = "") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        print(key_column_names)
        key_indices = {col_name: self.column_names.index(col_name) for col_name in key_column_names}

        seen_keys = {}
        duplicates = set()

        for i, row in enumerate(self.data):
            key_values = tuple(row[key_indices[col]] for col in key_column_names)
            if key_values in seen_keys:
                #duplicates.add(seen_keys[key_values])
                duplicates.add(i)
            else:
                seen_keys[key_values] = i

        return sorted(list(duplicates))

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [row for row in self.data if "NA" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col = self.get_column(col_index, False)
        s = 0 
        i = 1
        for n in col:
            if n != '':
                s+=n
                i+=1
        avg = s / i
        for row in self.data:
            if row[col_index] == "NA" or row[col_index] == '':
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_stats = []
        for col_name in col_names:
            col_data = self.get_column(col_name, include_missing_values=False)
            if col_data:
                min_val = min(col_data)
                max_val = max(col_data)
                mid_val = (min_val + max_val) / 2
                avg_val = sum(col_data) / len(col_data)
                sorted_col_data = sorted(col_data)
                n = len(sorted_col_data)
                if n % 2 == 0:
                    median_val = (sorted_col_data[n // 2 - 1] + sorted_col_data[n // 2]) / 2
                else:
                    median_val = sorted_col_data[n // 2]
                summary_stats.append([col_name, min_val, max_val, mid_val, avg_val, median_val])
        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], summary_stats)
    
    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indexes = [self.column_names.index(key) for key in key_column_names]
        other_key_indexes = [other_table.column_names.index(key) for key in key_column_names]
        
        # Prepare the joined table's columns
        combined_columns = list(self.column_names)
        for col in other_table.column_names:
            if col not in key_column_names:
                combined_columns.append(col)
        
        # Prepare the joined data
        joined_data = []
        
        # Perform the inner join by matching rows on the key columns
        for self_row in self.data:
            self_key = tuple(self_row[index] for index in self_key_indexes)
            for other_row in other_table.data:
                other_key = tuple(other_row[index] for index in other_key_indexes)
                if self_key == other_key:
                    # Combine rows and avoid duplicating key columns from the other table
                    combined_row = self_row[:]
                    for i, value in enumerate(other_row):
                        if i not in other_key_indexes:
                            combined_row.append(value)
                    joined_data.append(combined_row)
        
        return MyPyTable(combined_columns, joined_data)
                
    

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        self_key_indexes = [self.column_names.index(key) for key in key_column_names]
        other_key_indexes = [other_table.column_names.index(key) for key in key_column_names]

        # Prepare the joined table's columns
        combined_columns = list(self.column_names)
        for col in other_table.column_names:
            if col not in key_column_names:
                combined_columns.append(col)

        # Prepare the joined data
        joined_data = []

        # Create a set to track which rows from other_table have been joined
        other_table_joined = [False] * len(other_table.data)

        # Perform the outer join by matching rows on the key columns
        for self_row in self.data:
            self_key = tuple(self_row[index] for index in self_key_indexes)
            row_joined = False
            for i, other_row in enumerate(other_table.data):
                other_key = tuple(other_row[index] for index in other_key_indexes)
                if self_key == other_key:
                    # Combine rows and avoid duplicating key columns from the other table
                    combined_row = self_row[:]
                    for j, value in enumerate(other_row):
                        if j not in other_key_indexes:
                            combined_row.append(value)
                    joined_data.append(combined_row)
                    other_table_joined[i] = True
                    row_joined = True
            if not row_joined:
                # If no match, add the self_row with 'NA' for missing columns from the other table
                combined_row = self_row + ['NA'] * (len(combined_columns) - len(self.column_names))
                joined_data.append(combined_row)

        # Add rows from the other table that were not joined
        for i, other_row in enumerate(other_table.data):
            if not other_table_joined[i]:
                # Retain the key columns from other_row and fill the missing columns from self table with 'NA'
                combined_row = []
                for j in range(len(self.column_names)):
                    if j in self_key_indexes:
                        combined_row.append(other_row[other_key_indexes[self_key_indexes.index(j)]])
                    else:
                        combined_row.append('NA')
                for j, value in enumerate(other_row):
                    if j not in other_key_indexes:
                        combined_row.append(value)
                joined_data.append(combined_row)

        return MyPyTable(combined_columns, joined_data)


        
