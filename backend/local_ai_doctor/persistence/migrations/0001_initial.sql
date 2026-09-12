CREATE TABLE models (
    id TEXT PRIMARY KEY,
    canonical_path TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    architecture TEXT,
    task TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    fingerprint_algorithm TEXT NOT NULL,
    parameter_count INTEGER,
    weight_bytes INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_models_task ON models(task);
CREATE INDEX idx_models_fingerprint ON models(fingerprint);

CREATE TABLE model_capabilities (
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    capability TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('full', 'partial', 'unsupported', 'unavailable_on_backend')),
    reason TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (model_id, capability)
);

CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_chats_list ON chats(archived, pinned DESC, updated_at DESC);
CREATE INDEX idx_chats_title ON chats(title COLLATE NOCASE);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'streaming', 'complete', 'cancelled', 'failed')),
    branch_index INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_messages_chat_created ON messages(chat_id, created_at);
CREATE INDEX idx_messages_parent ON messages(parent_id, branch_index);

CREATE TABLE attachments (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    storage_name TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    preprocessing_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE message_attachments (
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    attachment_id TEXT NOT NULL REFERENCES attachments(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (message_id, attachment_id)
);

CREATE TABLE inference_runs (
    id TEXT PRIMARY KEY,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    model_id TEXT REFERENCES models(id) ON DELETE SET NULL,
    parent_run_id TEXT REFERENCES inference_runs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('generation', 'embedding', 'prompt_score', 'benchmark')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'loading', 'running', 'complete', 'cancelled', 'failed', 'disconnected')),
    requested_seed TEXT,
    effective_seed TEXT NOT NULL,
    rng_algorithm TEXT NOT NULL,
    generator_device TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    effective_config_json TEXT NOT NULL,
    reproducibility_json TEXT NOT NULL,
    model_fingerprint TEXT,
    tokenizer_fingerprint TEXT,
    rendered_prompt TEXT,
    prompt_token_count INTEGER,
    generated_token_count INTEGER NOT NULL DEFAULT 0,
    finish_reason TEXT,
    error_code TEXT,
    error_message TEXT,
    received_at TEXT NOT NULL,
    queue_entered_at TEXT,
    queue_exited_at TEXT,
    started_at TEXT,
    first_token_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_runs_message ON inference_runs(message_id, created_at);
CREATE INDEX idx_runs_status ON inference_runs(status, created_at);
CREATE INDEX idx_runs_model ON inference_runs(model_id, created_at);

CREATE TABLE environment_snapshots (
    run_id TEXT PRIMARY KEY REFERENCES inference_runs(id) ON DELETE CASCADE,
    hardware_json TEXT NOT NULL,
    software_json TEXT NOT NULL,
    backend_json TEXT NOT NULL,
    captured_at TEXT NOT NULL
);

CREATE TABLE phase_metrics (
    run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    started_ns INTEGER,
    ended_ns INTEGER,
    duration_ms REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, phase)
);

CREATE TABLE token_events (
    run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    token_index INTEGER NOT NULL CHECK (token_index >= 0),
    token_id INTEGER NOT NULL,
    piece TEXT NOT NULL,
    escaped_bytes TEXT NOT NULL,
    display_text TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    raw_logit REAL,
    raw_logprob REAL,
    raw_probability REAL,
    raw_rank INTEGER,
    processed_logit REAL,
    sample_logprob REAL,
    sample_probability REAL,
    entropy REAL,
    surprise REAL,
    cumulative_logprob REAL,
    running_perplexity REAL,
    decode_ms REAL,
    sample_ms REAL,
    emit_ms REAL,
    inter_token_ms REAL,
    cumulative_ms REAL,
    instantaneous_tps REAL,
    rolling_tps REAL,
    segment TEXT NOT NULL DEFAULT 'unknown' CHECK (segment IN ('reasoning', 'answer', 'unknown')),
    selected_experts_json TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, token_index)
);

CREATE INDEX idx_token_events_run_id ON token_events(run_id, token_id);

CREATE TABLE token_alternatives (
    run_id TEXT NOT NULL,
    token_index INTEGER NOT NULL,
    distribution TEXT NOT NULL CHECK (distribution IN ('raw', 'sampling')),
    rank INTEGER NOT NULL CHECK (rank > 0),
    token_id INTEGER NOT NULL,
    piece TEXT NOT NULL,
    logit REAL,
    log_probability REAL,
    probability REAL,
    survived_filter INTEGER NOT NULL CHECK (survived_filter IN (0, 1)),
    PRIMARY KEY (run_id, token_index, distribution, rank),
    FOREIGN KEY (run_id, token_index) REFERENCES token_events(run_id, token_index) ON DELETE CASCADE
);

CREATE TABLE router_events (
    run_id TEXT NOT NULL,
    token_index INTEGER NOT NULL,
    layer_index INTEGER NOT NULL,
    selected_json TEXT NOT NULL,
    executed_json TEXT NOT NULL,
    router_entropy REAL,
    dropped_assignments INTEGER,
    PRIMARY KEY (run_id, token_index, layer_index),
    FOREIGN KEY (run_id, token_index) REFERENCES token_events(run_id, token_index) ON DELETE CASCADE
);

CREATE TABLE router_aggregates (
    run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    layer_index INTEGER NOT NULL,
    expert_id INTEGER NOT NULL,
    activation_count INTEGER NOT NULL,
    average_weight REAL,
    load_fraction REAL,
    PRIMARY KEY (run_id, layer_index, expert_id)
);

CREATE TABLE embedding_runs (
    id TEXT PRIMARY KEY REFERENCES inference_runs(id) ON DELETE CASCADE,
    pooling TEXT NOT NULL,
    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    requested_dimensions INTEGER,
    output_dimensions INTEGER NOT NULL,
    output_dtype TEXT NOT NULL,
    joint_space INTEGER CHECK (joint_space IN (0, 1)),
    preprocessing_ms REAL,
    forward_ms REAL,
    total_ms REAL,
    items_per_second REAL,
    tokens_per_second REAL,
    peak_memory_bytes INTEGER,
    summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE embedding_inputs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES embedding_runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    input_type TEXT NOT NULL,
    content_preview TEXT,
    attachment_id TEXT REFERENCES attachments(id) ON DELETE SET NULL,
    token_count INTEGER,
    truncated INTEGER NOT NULL DEFAULT 0 CHECK (truncated IN (0, 1)),
    vector_json TEXT,
    vector_blob BLOB,
    l2_norm REAL,
    statistics_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (run_id, ordinal)
);

CREATE TABLE raw_events (
    run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    event_type TEXT NOT NULL,
    protocol_version INTEGER NOT NULL,
    monotonic_ns INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE INDEX idx_raw_events_run_sequence ON raw_events(run_id, sequence);
