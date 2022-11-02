pass


class data_integrity:
    def __init__(self, table_name: str, lookback_val: str, lookback_frame: str, lookback_utc_adjustment: str):
        self.table_name = table_name
        self.lookback_val = lookback_val
        self.lookback_frame = lookback_frame
        self.lookback_utc_adjustment = lookback_utc_adjustment


    def missing_index_utcnow(self):
        pass