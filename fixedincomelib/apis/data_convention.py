import pandas as pd
from fixedincomelib.conventions.data_conventions import DataConvention

def displayDataConvention(data_convention: DataConvention):
    base_info = [("unique_name", data_convention.unique_name),
                 ("data_type", data_convention.data_type)]
    content_items = list(data_convention.content.items())
    df = pd.DataFrame(base_info + content_items, columns=["Field", "Value"])
    return df
    

