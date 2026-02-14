-- Enable pgvector in every database created from this image.
-- Runs once when the data directory is initialized (first run with empty volume).
CREATE EXTENSION IF NOT EXISTS vector;
