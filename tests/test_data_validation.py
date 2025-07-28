def test_column_names():
    import pandas as pd
    df = pd.read_csv("sample_sales_data.csv")
    expected_cols = ["date", "product_category", "region", "units_sold", "revenue"]
    assert list(df.columns) == expected_cols
    