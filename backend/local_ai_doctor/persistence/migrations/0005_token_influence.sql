CREATE TABLE token_influence (
    run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    token_index INTEGER NOT NULL CHECK (token_index >= 0),
    method TEXT NOT NULL CHECK (method IN ('attention', 'gradient_x_input')),
    -- Canonical JSON {layers, alternative_token_id, source_limit}.
    parameters_key TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL,
    duration_ms REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, token_index, method, parameters_key)
);
