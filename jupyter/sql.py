import sqlite3

conn = sqlite3.connect('results.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Results (
        id INTEGER PRIMARY KEY,
        model_name TEXT,
        task_type TEXT CHECK(task_type IN ('classification', 'forecast', 'anomaly')),
        dataset_name TEXT,
        seq_len INTEGER,
        metric TEXT,
        value REAL
    )
''')

data = [
    ('TimesNet', 'forecast', 'ECL', 96, 'MSE', 0.168),
    ('DLinear', 'forecast', 'ECL', 96, 'MSE', 0.197)
]

cursor.executemany('''
    INSERT INTO Results (model_name, task_type, dataset_name, seq_len, metric, value)
    VALUES (?, ?, ?, ?, ?, ?)
''', data)

conn.commit()
cursor.close()
conn.close()