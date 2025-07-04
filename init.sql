-- init.sql
-- This will be automatically executed when PostgreSQL starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create initial schema will be handled by your application
-- The database "graphrag_chat" is already created by POSTGRES_DB

-- Optional: Create a test table to verify setup
CREATE TABLE IF NOT EXISTS setup_test (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    message TEXT DEFAULT 'PostgreSQL setup successful!'
);

INSERT INTO setup_test (message) VALUES ('Database initialized successfully!');