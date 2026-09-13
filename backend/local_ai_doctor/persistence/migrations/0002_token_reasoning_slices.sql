ALTER TABLE token_events
ADD COLUMN reasoning_slices_json TEXT NOT NULL DEFAULT '[]';
