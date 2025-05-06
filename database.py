import mysql.connector
from mysql.connector import Error

def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='amine3214',
        database='segmentation_db'
    )

def initialize_database():
    connection = create_connection()
    cursor = connection.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(255) NOT NULL,
        model_type ENUM('segmentation', 'classification') NOT NULL,
        model_data LONGBLOB NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            original_image VARCHAR(255) NOT NULL,
            result_image VARCHAR(255) NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    connection.commit()
    cursor.close()
    connection.close()

if __name__ == '__main__':
    initialize_database()