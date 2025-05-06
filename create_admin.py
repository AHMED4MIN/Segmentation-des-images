# create_admin.py
from database import create_connection
from werkzeug.security import generate_password_hash

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"  # Change this!

def create_admin():
    connection = create_connection()
    cursor = connection.cursor()
    
    try:
        # Check if admin exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (ADMIN_USERNAME,))
        if cursor.fetchone():
            print("Admin user already exists")
            return

        # Create hashed password
        hashed_password = generate_password_hash(ADMIN_PASSWORD)
        
        # Insert admin user
        cursor.execute("""
            INSERT INTO users (username, password, is_admin)
            VALUES (%s, %s, TRUE)
        """, (ADMIN_USERNAME, hashed_password))
        
        connection.commit()
        print("Admin user created successfully")
        
    except Exception as e:
        print(f"Error creating admin: {str(e)}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_admin()