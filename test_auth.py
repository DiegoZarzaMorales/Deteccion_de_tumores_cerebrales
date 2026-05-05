#!/usr/bin/env python
"""Script de prueba para validar la instalación de autenticación."""

import os
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test que todos los imports funcionen."""
    print("🔍 Probando imports...")
    try:
        from web.models import db, User, LoginHistory, Note
        print("  ✓ Modelos importados correctamente")
    except Exception as e:
        print(f"  ✗ Error importando modelos: {e}")
        return False
    
    try:
        from web.app import app, login_manager
        print("  ✓ App Flask importada correctamente")
    except Exception as e:
        print(f"  ✗ Error importando app: {e}")
        return False
    
    return True


def test_database():
    """Test que la base de datos se puede crear."""
    print("\n🗄️ Probando base de datos...")
    try:
        from web.app import app, db
        
        with app.app_context():
            db.create_all()
            print("  ✓ Base de datos creada/inicializada correctamente")
            
            # Verificar que las tablas existan
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            required_tables = {'users', 'login_history', 'notes'}
            if required_tables.issubset(set(tables)):
                print("  ✓ Todas las tablas fueron creadas")
                for table in required_tables:
                    print(f"    • {table}")
            else:
                print(f"  ✗ Faltan tablas. Esperadas: {required_tables}, Encontradas: {set(tables)}")
                return False
                
    except Exception as e:
        print(f"  ✗ Error con la base de datos: {e}")
        return False
    
    return True


def test_routes():
    """Test que las rutas estén registradas."""
    print("\n🛣️ Probando rutas...")
    try:
        from web.app import app
        
        routes = {}
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static':
                routes[str(rule)] = rule.endpoint
        
        required_routes = [
            '/login',
            '/register',
            '/logout',
            '/main',
            '/notes',
            '/notes/create',
            '/profile',
            '/predict'
        ]
        
        for route in required_routes:
            found = any(route in str(r) for r in routes.keys())
            status = "✓" if found else "✗"
            print(f"  {status} {route}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error verificando rutas: {e}")
        return False


def main():
    print("=" * 50)
    print("🧠 VALIDACIÓN DEL SISTEMA DE AUTENTICACIÓN")
    print("=" * 50)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Base de Datos", test_database()))
    results.append(("Rutas", test_routes()))
    
    print("\n" + "=" * 50)
    print("RESULTADOS:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 Todos los tests pasaron correctamente!")
        print("\nPara iniciar la aplicación, ejecuta:")
        print("  python web_app.py")
        print("\nLuego accede a: http://localhost:5000")
        return 0
    else:
        print("\n❌ Algunos tests fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
