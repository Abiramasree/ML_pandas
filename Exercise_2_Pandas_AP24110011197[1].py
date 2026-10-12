# Exercise 2 - Basic exercises on Python Machine Learning Packages
# Register No: AP24110011197

import numpy as np
import pandas as pd

# Register No: AP24110011197
# 1. Create a Series from a NumPy array
info = np.array(["S", "I", "V", "A"])
series = pd.Series(info)
print("1. Series from array:")
print(series)

# Register No: AP24110011197
# 2. Create an empty DataFrame
empty_df = pd.DataFrame()
print("\n2. Empty DataFrame:")
print(empty_df)

# Register No: AP24110011197
# 3. Create a DataFrame using a list
names = ["Siva", "Rama", "Krishna"]
list_df = pd.DataFrame(names)
print("\n3. DataFrame from list:")
print(list_df)

# Register No: AP24110011197
# 4. Create a DataFrame from a dictionary of lists
info_dict = {
    "ID": [1001, 1002, 1003],
    "Department": ["B.Tech", "M.Tech", "PhD"]
}
dict_df = pd.DataFrame(info_dict)
print("\n4. DataFrame from dictionary of lists:")
print(dict_df)

# Register No: AP24110011197
# 5. Create a DataFrame from a dictionary of Series
series_info = {
    "one": pd.Series([1, 2, 3, 4, 5, 6], index=list("abcdef")),
    "two": pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list("abcdefgh"))
}
df_series = pd.DataFrame(series_info)
print("\n5. DataFrame from dictionary of Series:")
print(df_series)

# Register No: AP24110011197
# 6. Select columns from a DataFrame
print("\n6. Column selection - one:")
print(df_series["one"])
print("Column selection - two:")
print(df_series["two"])

# Register No: AP24110011197
# 7. Add columns to a DataFrame
column_info = {
    "one": pd.Series([1, 2, 3, 4, 5], index=list("abcde")),
    "two": pd.Series([1, 2, 3, 4, 5, 6], index=list("abcdef"))
}
column_df = pd.DataFrame(column_info)
print("\n7. Original DataFrame:")
print(column_df)
print("Add new column by passing a Series:")
column_df["three"] = pd.Series([20, 40, 60], index=["a", "b", "c"])
print(column_df)
print("Add new column using existing columns:")
column_df["four"] = column_df["one"] + column_df["three"]
print(column_df)

# Register No: AP24110011197
# 8. Delete columns using del and pop
delete_info = {
    "one": pd.Series([1, 2], index=["a", "b"]),
    "two": pd.Series([1, 2, 3], index=["a", "b", "c"])
}
delete_df = pd.DataFrame(delete_info)
print("\n8. DataFrame before deleting columns:")
print(delete_df)
del delete_df["one"]
print("After deleting column 'one':")
print(delete_df)
delete_df.pop("two")
print("After deleting column 'two':")
print(delete_df)

# Register No: AP24110011197
# 9. Select rows by label using loc
row_info = {
    "one": pd.Series([1, 2, 3, 4, 5], index=list("abcde")),
    "two": pd.Series([1, 2, 3, 4, 5, 6], index=list("abcdef"))
}
row_df = pd.DataFrame(row_info)
print("\n9. DataFrame:")
print(row_df)
print("Row with label b:")
print(row_df.loc["b"])
print("Row with label e:")
print(row_df.loc["e"])

# Register No: AP24110011197
# 10. Select a row by integer position using iloc
print("\n10. Row at integer position 3:")
print(row_df.iloc[3])

# Register No: AP24110011197
# 11. Slice rows
print("\n11. Rows from position 2 to 4:")
print(row_df[2:5])

# Register No: AP24110011197
# 12. Add rows using pandas.concat
first_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
second_df = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
combined_df = pd.concat([first_df, second_df], ignore_index=True)
print("\n12. DataFrame after adding rows:")
print(combined_df)

# Register No: AP24110011197
# 13. Delete rows by index label
a_info = pd.DataFrame([[4, 5], [6, 7]], columns=["x", "y"])
b_info = pd.DataFrame([[8, 9], [10, 11]], columns=["x", "y"])
combined_rows = pd.concat([a_info, b_info])
print("\n13. Before dropping rows with label 0:")
print(combined_rows)
combined_rows = combined_rows.drop(index=0)
print("After dropping rows with label 0:")
print(combined_rows)

# Register No: AP24110011197
# 14. Select and display DataFrame columns
employee_data = {
    "Gender": ["F", "M", "M"],
    "Emp_ID": ["E01", "E02", "E03"],
    "Age": [25, 27, 25]
}
employee_df = pd.DataFrame(employee_data, columns=["Emp_ID", "Gender", "Age"])
print("\n14. Employee DataFrame:")
print(employee_df)
print("Employee IDs:")
print(employee_df["Emp_ID"])
print("First employee ID:")
print(employee_df["Emp_ID"].iloc[0])

# Register No: AP24110011197
# 15. Read and inspect a CSV file
# If iris.csv is in the same folder, it will be used. Otherwise use the
# built-in Iris dataset from scikit-learn so this example runs without a fixed path.
try:
    iris_df = pd.read_csv("iris.csv")
except FileNotFoundError:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    iris_df = iris.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species"
    })
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

print("\n15. Iris dataset:")
print(iris_df)
print("Dataset description:")
print(iris_df.describe(include="all"))
print("First five rows:")
print(iris_df.head())
print("Last five rows:")
print(iris_df.tail())

# Register No: AP24110011197
# 16. Remove duplicate rows
employee_data = {"Name": ["Siva", "Rama", "Krishna", "Siva"],
                 "Age": [21, 32, 29, 21]}
duplicate_df = pd.DataFrame(employee_data)
print("\n16. Original DataFrame:")
print(duplicate_df)
duplicate_df = duplicate_df.drop_duplicates()
print("After removing duplicates:")
print(duplicate_df)

# Register No: AP24110011197
# 17. Identify null values
null_df = pd.DataFrame({
    "Person": ["Siva", "Pavan", "Gopal", "Raghu"],
    "Age": [27.0, 29, np.nan, 32]
})
print("\n17. DataFrame with a null value:")
print(null_df)
print("Non-null count:")
print(null_df.count())
print("Null-value mask:")
print(null_df.isnull())
print("Number of null values in each column:")
print(null_df.isnull().sum())

# Register No: AP24110011197
# 18. Sort rows by index
sort_df = pd.DataFrame({
    "Person": ["Siva", "Pavan", "Gopal", "Raghu"],
    "Age": [27.0, 29, np.nan, 32]
})
print("\n18. Original DataFrame:")
print(sort_df)
print("Ascending index order:")
print(sort_df.sort_index())
print("Descending index order:")
print(sort_df.sort_index(ascending=False))

# Register No: AP24110011197
# 19. Aggregate values by species
numeric_iris = iris_df.copy()
numeric_iris["sepal_length"] = pd.to_numeric(numeric_iris["sepal_length"], errors="coerce")
numeric_iris["sepal_width"] = pd.to_numeric(numeric_iris["sepal_width"], errors="coerce")
numeric_iris["petal_length"] = pd.to_numeric(numeric_iris["petal_length"], errors="coerce")
numeric_iris["petal_width"] = pd.to_numeric(numeric_iris["petal_width"], errors="coerce")
mean_df = numeric_iris.groupby("species")[["sepal_length", "sepal_width", "petal_length", "petal_width"]].mean()
print("\n19. Mean values for each species:")
print(mean_df)
sum_df = numeric_iris.groupby("species")[["sepal_length", "sepal_width", "petal_length", "petal_width"]].sum()
print("Sum of values for each species:")
print(sum_df)
