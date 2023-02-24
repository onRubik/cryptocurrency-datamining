CREATE TABLE btcbusd_4h_historical (
    time   DATETIME PRIMARY KEY
                    NOT NULL,
    open   DECIMAL  NOT NULL,
    high   DECIMAL  NOT NULL,
    low    DECIMAL  NOT NULL,
    close  DECIMAL  NOT NULL,
    volume DECIMAL  NOT NULL
)
WITHOUT ROWID;

CREATE TABLE stage_btcbusd_4h_historical (
    time   DATETIME PRIMARY KEY
                    NOT NULL,
    open   DECIMAL  NOT NULL,
    high   DECIMAL  NOT NULL,
    low    DECIMAL  NOT NULL,
    close  DECIMAL  NOT NULL,
    volume DECIMAL  NOT NULL
)
WITHOUT ROWID;