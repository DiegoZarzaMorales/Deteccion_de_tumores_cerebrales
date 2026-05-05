# 🔐 SISTEMA DE AUTENTICACIÓN Y GESTIÓN DE NOTAS

## ✅ Cambios Realizados

Tu proyecto ahora incluye un **sistema completo de autenticación** con base de datos relacional. Aquí está todo lo que se agregó:

### 📦 Nuevas Dependencias Instaladas
- **Flask-SQLAlchemy**: ORM para gestionar la base de datos
- **Flask-Login**: Sistema de autenticación y sesiones
- **Werkzeug**: Manejo seguro de contraseñas (hasheadas)

### 🗄️ Base de Datos Relacional
Se creó una base de datos SQLite (`tumores_cerebrales.db`) con 3 tablas:

#### 1. **users** - Tabla de Usuarios
```
- id (Clave Primaria)
- username (Único)
- email (Único)
- password_hash (Hasheada con seguridad)
- created_at (Fecha de creación)
```

#### 2. **login_history** - Historial de Inicios de Sesión
```
- id (Clave Primaria)
- user_id (Clave Foránea → users)
- login_date (Fecha y hora del login)
- ip_address (Dirección IP del usuario)
- user_agent (Información del navegador/dispositivo)
```

#### 3. **notes** - Bloc de Notas por Usuario
```
- id (Clave Primaria)
- user_id (Clave Foránea → users)
- title (Título de la nota)
- content (Contenido de la nota)
- created_at (Fecha de creación)
- updated_at (Fecha de última actualización)
```

### 🎨 Nuevas Plantillas HTML
1. **login.html** - Página de inicio de sesión
2. **register.html** - Página de registro de nuevos usuarios
3. **notes.html** - Vista principal del bloc de notas (lista de notas)
4. **create_note.html** - Crear una nueva nota
5. **edit_note.html** - Editar una nota existente
6. **profile.html** - Perfil del usuario + historial de logins

### 🛣️ Nuevas Rutas (Endpoints)
```
GET/POST  /login              - Iniciar sesión
GET/POST  /register           - Crear nueva cuenta
GET       /logout             - Cerrar sesión
GET       /main               - Análisis de tumores (requiere login)
POST      /predict            - Procesar imágenes DICOM (requiere login)
GET       /notes              - Ver todas las notas del usuario
GET/POST  /notes/create       - Crear nueva nota
GET/POST  /notes/<id>/edit    - Editar nota
POST      /notes/<id>/delete  - Eliminar nota
GET       /profile            - Ver perfil y historial de logins
GET       /                   - Redirige a login o main_menu según autenticación
```

### 🔒 Características de Seguridad
- ✅ Contraseñas hasheadas con Werkzeug
- ✅ Sesiones seguras con Flask-Login
- ✅ Protección de rutas con `@login_required`
- ✅ Cada usuario solo puede acceder a sus propias notas
- ✅ Historial de logins registrado (IP, navegador, fecha)

---

## 🚀 Cómo Usar

### 1️⃣ Instalar Dependencias
Si aún no lo has hecho, instala las dependencias:
```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar la Aplicación
```bash
python web_app.py
```

La aplicación se ejecutará en `http://localhost:5000`

### 3️⃣ Flujo de Usuario

#### **Primer Acceso:**
1. Accede a `http://localhost:5000`
2. Serás redirigido a la página de login
3. Haz clic en "Regístrate aquí"
4. Completa el formulario de registro:
   - Nombre de usuario (mínimo 3 caracteres)
   - Correo electrónico
   - Contraseña (mínimo 6 caracteres)
5. Confirma la contraseña
6. Se redirige a login para iniciar sesión

#### **Iniciar Sesión:**
1. Ingresa tu usuario y contraseña
2. Marca "Recuérdame" si deseas mantener la sesión
3. ¡Bienvenido al análisis de tumores!

#### **Analizar Imágenes:**
- Accede a la sección "Análisis"
- Sube una imagen DICOM (.dcm) o un archivo ZIP
- La IA procesa la imagen y marca la región tumoral

#### **Gestionar Notas:**
- Accede a la sección "Notas"
- Crea nuevas notas para documentar observaciones clínicas
- Edita o elimina notas según sea necesario
- Cada nota muestra fecha de creación y última actualización

#### **Ver Perfil:**
- Accede a "Mi Perfil" desde el menú dropdown
- Visualiza tu información personal
- Consulta el historial completo de logins con:
  - Fecha y hora exacta
  - Dirección IP
  - Información del navegador/dispositivo
- Ve estadísticas (total de notas, total de logins)

#### **Cerrar Sesión:**
- Haz clic en "Cerrar Sesión" en el menú dropdown
- Tu sesión se cerrará y volverás a la página de login

---

## 📝 Archivos Creados/Modificados

### Creados:
- `web/models.py` - Modelos de base de datos (User, LoginHistory, Note)
- `templates/login.html` - Página de login
- `templates/register.html` - Página de registro
- `templates/notes.html` - Vista de notas
- `templates/create_note.html` - Crear nota
- `templates/edit_note.html` - Editar nota
- `templates/profile.html` - Perfil del usuario

### Modificados:
- `web/app.py` - Agregadas rutas de autenticación, notas y perfil
- `templates/main_menu.html` - Agregada navbar con opciones de usuario
- `requirements.txt` - Agregadas nuevas dependencias

---

## 🔑 Variables de Entorno

Para mayor seguridad en producción, puedes usar variables de entorno:

```bash
export SECRET_KEY="tu-clave-secreta-muy-segura"
```

Si no establecer `SECRET_KEY`, se usará un valor por defecto para desarrollo (⚠️ NO usar en producción).

---

## 🐛 Notas Importantes

1. **Base de Datos**: Se crea automáticamente la primera vez que ejecutas la app
2. **Datos Persisten**: Todos los usuarios, notas e historial de logins se guardan en `tumores_cerebrales.db`
3. **Contraseñas**: Nunca se almacenan en texto plano, solo hasheadas
4. **Privacidad**: Cada usuario solo ve sus propias notas e historial
5. **Seguridad de Rutas**: Las rutas sensibles requieren autenticación (`@login_required`)

---

## ⚠️ Consideraciones para Producción

Si planeas usar esta app en producción:
1. Cambia `app.run(debug=True)` a `debug=False` en `web_app.py`
2. Establece una `SECRET_KEY` segura como variable de entorno
3. Usa una base de datos robusta (PostgreSQL, MySQL) en lugar de SQLite
4. Configura HTTPS
5. Implementa backup automático de la base de datos
6. Considera agregar autenticación de dos factores (2FA)

---

## 📊 Ejemplo de Consulta SQL

Si necesitas consultar la base de datos directamente:

```sql
-- Ver todos los usuarios
SELECT * FROM users;

-- Ver historial de logins
SELECT u.username, lh.login_date, lh.ip_address 
FROM login_history lh 
JOIN users u ON lh.user_id = u.id 
ORDER BY lh.login_date DESC;

-- Ver notas de un usuario
SELECT * FROM notes WHERE user_id = 1 ORDER BY updated_at DESC;
```

---

¡Tu aplicación de detección de tumores cerebrales ya tiene un sistema completo de autenticación! 🎉
