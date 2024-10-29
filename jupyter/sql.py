import sqlite3

def model_filter(model_name=None, task_type=None, dataset_name=None, seq_len=None, metric=None, orderBy=None):
    conn = sqlite3.connect('results.db')
    print("Connection established to ---> results.db")
    cursor = conn.cursor()
    table = 'Results'
    print("Reading from table name ---> Results")

    show = 'model_name, value'

    sql_cmd = f"SELECT distinct {show} FROM {table} "
    if model_name or task_type or dataset_name or seq_len or metric:
        sql_cmd += f" WHERE 1"
        if model_name:
            sql_cmd += f" AND model_name IN{model_name}"
        if task_type:
            sql_cmd += f" AND task_type='{task_type}'"
        if dataset_name:
            sql_cmd += f" AND dataset_name='{dataset_name}'"
        if seq_len:
            sql_cmd += f" AND seq_len='{seq_len}'"
        if metric:
            sql_cmd += f" AND metric='{metric}'"
        if orderBy:
            sql_cmd+=f" ORDER BY {orderBy}"
    print("sql input command: ", sql_cmd)
    cursor.execute(sql_cmd)
    res = cursor.fetchall()
    cursor.close()
    conn.close()
    return res

conn = sqlite3.connect('results.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Results (
        id INTEGER PRIMARY KEY,
        model_name TEXT,
        task_type TEXT CHECK(task_type IN ('classification', 'long-term-forecast', 'short-term-forecast', 'anomaly', 'imputation')),
        dataset_name TEXT,
        seq_len INTEGER,
        metric TEXT,
        value REAL
    )
''')

long_forecast_data = [
    # TimesNet Model
    ('TimesNet', 'long-term-forecast', 'ECL', 96, 'MSE', 0.168),
    ('TimesNet', 'long-term-forecast', 'ECL', 96, 'MAE', 0.272),
    ('TimesNet', 'long-term-forecast', 'ECL', 192, 'MSE', 0.184),
    ('TimesNet', 'long-term-forecast', 'ECL', 192, 'MAE', 0.289),
    ('TimesNet', 'long-term-forecast', 'ECL', 336, 'MSE', 0.198),
    ('TimesNet', 'long-term-forecast', 'ECL', 336, 'MAE', 0.300),
    ('TimesNet', 'long-term-forecast', 'ECL', 720, 'MSE', 0.220),
    ('TimesNet', 'long-term-forecast', 'ECL', 720, 'MAE', 0.320),
    
    ('TimesNet', 'long-term-forecast', 'Traffic', 96, 'MSE', 0.593),
    ('TimesNet', 'long-term-forecast', 'Traffic', 96, 'MAE', 0.321),
    ('TimesNet', 'long-term-forecast', 'Traffic', 192, 'MSE', 0.617),
    ('TimesNet', 'long-term-forecast', 'Traffic', 192, 'MAE', 0.336),
    ('TimesNet', 'long-term-forecast', 'Traffic', 336, 'MSE', 0.629),
    ('TimesNet', 'long-term-forecast', 'Traffic', 336, 'MAE', 0.336),
    ('TimesNet', 'long-term-forecast', 'Traffic', 720, 'MSE', 0.640),
    ('TimesNet', 'long-term-forecast', 'Traffic', 720, 'MAE', 0.350),
    
    ('TimesNet', 'long-term-forecast', 'Weather', 96, 'MSE', 0.172),
    ('TimesNet', 'long-term-forecast', 'Weather', 96, 'MAE', 0.220),
    ('TimesNet', 'long-term-forecast', 'Weather', 192, 'MSE', 0.219),
    ('TimesNet', 'long-term-forecast', 'Weather', 192, 'MAE', 0.261),
    ('TimesNet', 'long-term-forecast', 'Weather', 336, 'MSE', 0.280),
    ('TimesNet', 'long-term-forecast', 'Weather', 336, 'MAE', 0.306),
    ('TimesNet', 'long-term-forecast', 'Weather', 720, 'MSE', 0.365),
    ('TimesNet', 'long-term-forecast', 'Weather', 720, 'MAE', 0.359),
    
    # DLinear Model
    ('DLinear', 'long-term-forecast', 'ECL', 96, 'MSE', 0.197),
    ('DLinear', 'long-term-forecast', 'ECL', 96, 'MAE', 0.282),
    ('DLinear', 'long-term-forecast', 'ECL', 192, 'MSE', 0.196),
    ('DLinear', 'long-term-forecast', 'ECL', 192, 'MAE', 0.285),
    ('DLinear', 'long-term-forecast', 'ECL', 336, 'MSE', 0.209),
    ('DLinear', 'long-term-forecast', 'ECL', 336, 'MAE', 0.301),
    ('DLinear', 'long-term-forecast', 'ECL', 720, 'MSE', 0.245),
    ('DLinear', 'long-term-forecast', 'ECL', 720, 'MAE', 0.333),
    
    ('DLinear', 'long-term-forecast', 'Traffic', 96, 'MSE', 0.650),
    ('DLinear', 'long-term-forecast', 'Traffic', 96, 'MAE', 0.396),
    ('DLinear', 'long-term-forecast', 'Traffic', 192, 'MSE', 0.598),
    ('DLinear', 'long-term-forecast', 'Traffic', 192, 'MAE', 0.370),
    ('DLinear', 'long-term-forecast', 'Traffic', 336, 'MSE', 0.605),
    ('DLinear', 'long-term-forecast', 'Traffic', 336, 'MAE', 0.373),
    ('DLinear', 'long-term-forecast', 'Traffic', 720, 'MSE', 0.645),
    ('DLinear', 'long-term-forecast', 'Traffic', 720, 'MAE', 0.394),

    ('DLinear', 'long-term-forecast', 'Weather', 96, 'MSE', 0.196),
    ('DLinear', 'long-term-forecast', 'Weather', 96, 'MAE', 0.255),
    ('DLinear', 'long-term-forecast', 'Weather', 192, 'MSE', 0.237),
    ('DLinear', 'long-term-forecast', 'Weather', 192, 'MAE', 0.296),
    ('DLinear', 'long-term-forecast', 'Weather', 336, 'MSE', 0.283),
    ('DLinear', 'long-term-forecast', 'Weather', 336, 'MAE', 0.335),
    ('DLinear', 'long-term-forecast', 'Weather', 720, 'MSE', 0.345),
    ('DLinear', 'long-term-forecast', 'Weather', 720, 'MAE', 0.381),

    # Informer Model
    ('Informer', 'long-term-forecast', 'ECL', 96, 'MSE', 0.274),
    ('Informer', 'long-term-forecast', 'ECL', 96, 'MAE', 0.268),
    ('Informer', 'long-term-forecast', 'ECL', 192, 'MSE', 0.296),
    ('Informer', 'long-term-forecast', 'ECL', 192, 'MAE', 0.386),
    ('Informer', 'long-term-forecast', 'ECL', 336, 'MSE', 0.300),
    ('Informer', 'long-term-forecast', 'ECL', 336, 'MAE', 0.394),
    ('Informer', 'long-term-forecast', 'ECL', 720, 'MSE', 0.373),
    ('Informer', 'long-term-forecast', 'ECL', 720, 'MAE', 0.439),
    
    ('Informer', 'long-term-forecast', 'Traffic', 96, 'MSE', 0.719),
    ('Informer', 'long-term-forecast', 'Traffic', 96, 'MAE', 0.391),
    ('Informer', 'long-term-forecast', 'Traffic', 192, 'MSE', 0.696),
    ('Informer', 'long-term-forecast', 'Traffic', 192, 'MAE', 0.379),
    ('Informer', 'long-term-forecast', 'Traffic', 336, 'MSE', 0.777),
    ('Informer', 'long-term-forecast', 'Traffic', 336, 'MAE', 0.420),
    ('Informer', 'long-term-forecast', 'Traffic', 720, 'MSE', 0.864),
    ('Informer', 'long-term-forecast', 'Traffic', 720, 'MAE', 0.472),

    ('Informer', 'long-term-forecast', 'Weather', 96, 'MSE', 0.300),
    ('Informer', 'long-term-forecast', 'Weather', 96, 'MAE', 0.384),
    ('Informer', 'long-term-forecast', 'Weather', 192, 'MSE', 0.598),
    ('Informer', 'long-term-forecast', 'Weather', 192, 'MAE', 0.544),
    ('Informer', 'long-term-forecast', 'Weather', 336, 'MSE', 0.578),
    ('Informer', 'long-term-forecast', 'Weather', 336, 'MAE', 0.523),
    ('Informer', 'long-term-forecast', 'Weather', 720, 'MSE', 1.059),
    ('Informer', 'long-term-forecast', 'Weather', 720, 'MAE', 0.741),
    
    # Timer Model
    ('Timer', 'long-term-forecast', 'ECL', 96, 'MSE', 0.129),
    ('Timer', 'long-term-forecast', 'ECL', 96, 'MAE', 0.221),
    ('Timer', 'long-term-forecast', 'ECL', 192, 'MSE', 0.148),
    ('Timer', 'long-term-forecast', 'ECL', 192, 'MAE', 0.239),
    ('Timer', 'long-term-forecast', 'ECL', 336, 'MSE', 0.164),
    ('Timer', 'long-term-forecast', 'ECL', 336, 'MAE', 0.256),
    ('Timer', 'long-term-forecast', 'ECL', 720, 'MSE', 0.201),
    ('Timer', 'long-term-forecast', 'ECL', 720, 'MAE', 0.289),
    
    ('Timer', 'long-term-forecast', 'Traffic', 96, 'MSE', 0.348),
    ('Timer', 'long-term-forecast', 'Traffic', 96, 'MAE', 0.240),
    ('Timer', 'long-term-forecast', 'Traffic', 192, 'MSE', 0.369),
    ('Timer', 'long-term-forecast', 'Traffic', 192, 'MAE', 0.250),
    ('Timer', 'long-term-forecast', 'Traffic', 336, 'MSE', 0.388),
    ('Timer', 'long-term-forecast', 'Traffic', 336, 'MAE', 0.260),
    ('Timer', 'long-term-forecast', 'Traffic', 720, 'MSE', 0.431),
    ('Timer', 'long-term-forecast', 'Traffic', 720, 'MAE', 0.285),

    ('Timer', 'long-term-forecast', 'Weather', 96, 'MSE', 0.151),
    ('Timer', 'long-term-forecast', 'Weather', 96, 'MAE', 0.202),
    ('Timer', 'long-term-forecast', 'Weather', 192, 'MSE', 0.196),
    ('Timer', 'long-term-forecast', 'Weather', 192, 'MAE', 0.245),
    ('Timer', 'long-term-forecast', 'Weather', 336, 'MSE', 0.249),
    ('Timer', 'long-term-forecast', 'Weather', 336, 'MAE', 0.288),
    ('Timer', 'long-term-forecast', 'Weather', 720, 'MSE', 0.330),
    ('Timer', 'long-term-forecast', 'Weather', 720, 'MAE', 0.344),

    # Moirai Model
    ('Moirai', 'long-term-forecast', 'ECL', 96, 'MSE', 0.130),
    ('Moirai', 'long-term-forecast', 'ECL', 96, 'MAE', 0.225),
    ('Moirai', 'long-term-forecast', 'ECL', 192, 'MSE', 0.150),
    ('Moirai', 'long-term-forecast', 'ECL', 192, 'MAE', 0.244),
    ('Moirai', 'long-term-forecast', 'ECL', 336, 'MSE', 0.166),
    ('Moirai', 'long-term-forecast', 'ECL', 336, 'MAE', 0.262),
    ('Moirai', 'long-term-forecast', 'ECL', 720, 'MSE', 0.206),
    ('Moirai', 'long-term-forecast', 'ECL', 720, 'MAE', 0.297),
    
    ('Moirai', 'long-term-forecast', 'Traffic', 96, 'MSE', 0.359),
    ('Moirai', 'long-term-forecast', 'Traffic', 96, 'MAE', 0.250),
    ('Moirai', 'long-term-forecast', 'Traffic', 192, 'MSE', 0.373),
    ('Moirai', 'long-term-forecast', 'Traffic', 192, 'MAE', 0.257),
    ('Moirai', 'long-term-forecast', 'Traffic', 336, 'MSE', 0.386),
    ('Moirai', 'long-term-forecast', 'Traffic', 336, 'MAE', 0.265),
    ('Moirai', 'long-term-forecast', 'Traffic', 720, 'MSE', 0.421),
    ('Moirai', 'long-term-forecast', 'Traffic', 720, 'MAE', 0.286),

    ('Moirai', 'long-term-forecast', 'Weather', 96, 'MSE', 0.152),
    ('Moirai', 'long-term-forecast', 'Weather', 96, 'MAE', 0.206),
    ('Moirai', 'long-term-forecast', 'Weather', 192, 'MSE', 0.198),
    ('Moirai', 'long-term-forecast', 'Weather', 192, 'MAE', 0.249),
    ('Moirai', 'long-term-forecast', 'Weather', 336, 'MSE', 0.251),
    ('Moirai', 'long-term-forecast', 'Weather', 336, 'MAE', 0.291),
    ('Moirai', 'long-term-forecast', 'Weather', 720, 'MSE', 0.322),
    ('Moirai', 'long-term-forecast', 'Weather', 720, 'MAE', 0.340),
]

cursor.executemany('''
    INSERT INTO Results (model_name, task_type, dataset_name, seq_len, metric, value)
    VALUES (?, ?, ?, ?, ?, ?)
''', long_forecast_data)

conn.commit()
cursor.close()
conn.close()