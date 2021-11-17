def parse_paper_data(data, data_row_start):
    meta_data_df = data.head(5)
    meta_data_df = meta_data_df.set_index('Name')

    exam_data_df = data[data_row_start:]
    exam_data_df = exam_data_df.dropna(axis=0, subset=list(meta_data_df.columns)) # drop rows with nan values
    exam_data_df.reset_index(inplace=True)
    exam_data_df = exam_data_df.astype(float)
    exam_data_df = exam_data_df[meta_data_df.columns]
    
    return exam_data_df, meta_data_df
