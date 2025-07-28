import pandas as pd

def test_column_names():
    """
    Test to check if the uploaded CSV file contains the required columns.
    """
    df = pd.read_csv("sample_sales_data.csv")
    expected_columns = ["date", "product_category", "region", "units_sold", "revenue"]
    assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, but got {list(df.columns)}"
