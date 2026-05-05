#!/usr/bin/env python3
"""
Test script to validate login flow and session cookie handling
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app
from web.models import db, User

def test_login_flow():
    """Test the complete login flow including session cookie creation"""
    
    with app.app_context():
        # Clean up test database
        db.drop_all()
        db.create_all()
        
        # Create test user
        user = User(username='testuser', email='test@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()
        print("✓ Test user created")
        
    # Test client
    client = app.test_client()
    
    # Test login POST request (this triggers session.set_cookie())
    print("🔍 Testing login flow with POST request...")
    try:
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        }, follow_redirects=True)
        
        if response.status_code == 200:
            print("✓ Login POST request successful (status 200)")
            print(f"✓ Response location: {response.request.path}")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
            
    except TypeError as e:
        if 'partitioned' in str(e):
            print(f"✗ PARTITIONED COOKIE ERROR: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_login_flow()
    if success:
        print("\n" + "="*50)
        print("🎉 LOGIN FLOW TEST PASSED!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("❌ LOGIN FLOW TEST FAILED!")
        print("="*50)
        sys.exit(1)
