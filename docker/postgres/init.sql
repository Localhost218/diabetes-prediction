CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- AGE phải được LOAD trong mỗi session: đặt mặc định cho database để backend không phải tự LOAD.
-- public đứng trước ag_catalog để bảng của ứng dụng được tạo trong schema public.
DO $$
BEGIN
    EXECUTE format('ALTER DATABASE %I SET session_preload_libraries = %L', current_database(), 'age');
    EXECUTE format('ALTER DATABASE %I SET search_path = "$user", public, ag_catalog', current_database());
END
$$;

LOAD 'age';
SELECT ag_catalog.create_graph('diabetes_kg');
