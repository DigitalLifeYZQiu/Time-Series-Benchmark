import sqlite3

conn = sqlite3.connect('results.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Results (
        id INTEGER PRIMARY KEY,
        model_name TEXT,
        task_type TEXT CHECK(task_type IN ('classification', 'long-term-forecast', 'short-term-forecast', 'anomaly')),
        dataset_name TEXT,
        seq_len INTEGER,
        metric TEXT,
        value REAL
    )
''')

data = [
    ('TimesNet', 'long-term-forecast', 'ECL', 96, 'MSE', 0.168),
    ('DLinear', 'long-term-forecast', 'ECL', 96, 'MSE', 0.197),
    ('Informer', 'long-term-forecast', 'ECL', 96, 'MSE', 0.274),
]

cursor.executemany('''
    INSERT INTO Results (model_name, task_type, dataset_name, seq_len, metric, value)
    VALUES (?, ?, ?, ?, ?, ?)
''', data)

conn.commit()
cursor.close()
conn.close()