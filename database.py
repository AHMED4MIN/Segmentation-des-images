# database.py (updated)
import mysql.connector
from mysql.connector import Error


def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='amine3214',
        database='segmentation_db',
        
        
    )

def initialize_database():
    connection = create_connection()
    cursor = connection.cursor()
    

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_type ENUM('segmentation', 'classification') NOT NULL,
            model_format ENUM('torchscript', 'state_dict') NOT NULL,
            model_path VARCHAR(255) NOT NULL,  
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )   
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            first_name VARCHAR(255) NOT NULL,   # New
            last_name VARCHAR(255) NOT NULL,    # New
            cin VARCHAR(20) UNIQUE NOT NULL,    # New (CIN as unique string)
            date_of_birth DATE NOT NULL,        # New
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    
    connection.commit()
    cursor.close()
    connection.close()

if __name__ == '__main__':
    initialize_database()