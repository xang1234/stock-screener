/**
 * Theme API ingress adapters.
 *
 * Centralizes normalization + validation so UI components can consume
 * stable frontend-facing shapes.
 */

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const asString = (value, fallback = null) => {
  if (typeof value === 'string') return value;
  if (value == null) return fallback;
  return String(value);
};

const asOptionalNumber = (value) => {
  if (value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const requireNumber = (value, fieldName) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid ${fieldName}: expected number`);
  }
  return parsed;
};

const requireString = (value, fieldName) => {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Invalid ${fieldName}: expected non-empty string`);
  }
  return value;
};

const requireBoolean = (value, fieldName) => {
  if (typeof value !== 'boolean') {
    throw new Error(`Invalid ${fieldName}: expected boolean`);
  }
  return value;
};

const asStringArray = (value) => {
  if (!Array.isArray(value)) return [];
  return value.map((item) => asString(item, '')).filter((item) => item.length > 0);
};

export const adaptMergeSuggestion = (raw) => {
  if (!isObject(raw)) {
    throw new Error('Invalid merge suggestion: expected object');
  }

  const sourceThemeId = raw.source_theme_id ?? raw.source_cluster_id;
  const targetThemeId = raw.target_theme_id ?? raw.target_cluster_id;
  const sourceThemeName = raw.source_theme_name ?? raw.source_name;
  const targetThemeName = raw.target_theme_name ?? raw.target_name;

  return {
    id: requireNumber(raw.id, 'merge suggestion id'),
    source_theme_id: requireNumber(sourceThemeId, 'source_theme_id'),
    source_theme_name: requireString(sourceThemeName, 'source_theme_name'),
    source_aliases: asStringArray(raw.source_aliases),
    target_theme_id: requireNumber(targetThemeId, 'target_theme_id'),
    target_theme_name: requireString(targetThemeName, 'target_theme_name'),
    target_aliases: asStringArray(raw.target_aliases),
    similarity_score: asOptionalNumber(raw.similarity_score ?? raw.embedding_similarity),
    llm_confidence: asOptionalNumber(raw.llm_confidence),
    relationship_type: asString(raw.relationship_type ?? raw.llm_relationship),
    reasoning: asString(raw.reasoning ?? raw.llm_reasoning),
    suggested_name: asString(raw.suggested_name ?? raw.suggested_canonical_name),
    status: requireString(raw.status, 'status'),
    created_at: asString(raw.created_at),
  };
};

export const adaptMergeSuggestionsResponse = (raw) => {
  if (Array.isArray(raw)) {
    const suggestions = raw.map(adaptMergeSuggestion);
    return { total: suggestions.length, suggestions };
  }
  if (!isObject(raw)) {
    throw new Error('Invalid merge suggestions response');
  }
  const rawList = Array.isArray(raw.suggestions) ? raw.suggestions : [];
  return {
    total: Number.isFinite(Number(raw.total)) ? Number(raw.total) : rawList.length,
    suggestions: rawList.map(adaptMergeSuggestion),
  };
};

export const adaptMergeActionResponse = (raw) => {
  if (!isObject(raw)) {
    throw new Error('Invalid merge action response');
  }
  return {
    success: requireBoolean(raw.success, 'success'),
    error: asString(raw.error),
    source_name: asString(raw.source_name),
    target_name: asString(raw.target_name),
    constituents_merged: asOptionalNumber(raw.constituents_merged),
    mentions_merged: asOptionalNumber(raw.mentions_merged),
    idempotency_key: asString(raw.idempotency_key),
    idempotent_replay: raw.idempotent_replay == null ? null : Boolean(raw.idempotent_replay),
    warning: asString(raw.warning),
    status: asString(raw.status),
  };
};

const adaptMergeHistoryItem = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid merge history row');
  return {
    id: requireNumber(raw.id, 'merge history id'),
    source_name: requireString(raw.source_name, 'source_name'),
    target_name: requireString(raw.target_name, 'target_name'),
    merge_type: requireString(raw.merge_type, 'merge_type'),
    embedding_similarity: asOptionalNumber(raw.embedding_similarity),
    llm_confidence: asOptionalNumber(raw.llm_confidence),
    llm_reasoning: asString(raw.llm_reasoning),
    constituents_merged: asOptionalNumber(raw.constituents_merged),
    mentions_merged: asOptionalNumber(raw.mentions_merged),
    merged_at: asString(raw.merged_at),
    merged_by: asString(raw.merged_by),
  };
};

export const adaptMergeHistoryResponse = (raw) => {
  if (Array.isArray(raw)) {
    const history = raw.map(adaptMergeHistoryItem);
    return { total: history.length, history };
  }
  if (!isObject(raw)) {
    throw new Error('Invalid merge history response');
  }
  const rawHistory = Array.isArray(raw.history) ? raw.history : [];
  return {
    total: Number.isFinite(Number(raw.total)) ? Number(raw.total) : rawHistory.length,
    history: rawHistory.map(adaptMergeHistoryItem),
  };
};

const adaptLifecycleTransition = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid lifecycle transition row');
  return {
    id: requireNumber(raw.id, 'lifecycle transition id'),
    theme_cluster_id: requireNumber(raw.theme_cluster_id, 'theme_cluster_id'),
    theme_name: requireString(raw.theme_name, 'theme_name'),
    pipeline: requireString(raw.pipeline, 'pipeline'),
    from_state: requireString(raw.from_state, 'from_state'),
    to_state: requireString(raw.to_state, 'to_state'),
    actor: requireString(raw.actor, 'actor'),
    job_name: asString(raw.job_name),
    rule_version: asString(raw.rule_version),
    reason: asString(raw.reason),
    transition_metadata: isObject(raw.transition_metadata) ? raw.transition_metadata : {},
    transitioned_at: asString(raw.transitioned_at),
    transition_history_path: requireString(raw.transition_history_path, 'transition_history_path'),
    runbook_url: requireString(raw.runbook_url, 'runbook_url'),
  };
};

export const adaptLifecycleTransitionsResponse = (raw) => {
  if (!isObject(raw)) {
    throw new Error('Invalid lifecycle transitions response');
  }
  const rawTransitions = Array.isArray(raw.transitions) ? raw.transitions : [];
  return {
    total: Number.isFinite(Number(raw.total)) ? Number(raw.total) : rawTransitions.length,
    transitions: rawTransitions.map(adaptLifecycleTransition),
  };
};

const adaptCandidateQueueItem = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid candidate queue row');
  return {
    theme_cluster_id: requireNumber(raw.theme_cluster_id, 'theme_cluster_id'),
    theme_name: requireString(raw.theme_name, 'theme_name'),
    theme_display_name: requireString(raw.theme_display_name ?? raw.theme_name, 'theme_display_name'),
    candidate_since_at: asString(raw.candidate_since_at),
    avg_confidence_30d: asOptionalNumber(raw.avg_confidence_30d) ?? 0,
    confidence_band: requireString(raw.confidence_band, 'confidence_band'),
    mentions_7d: asOptionalNumber(raw.mentions_7d) ?? 0,
    source_diversity_7d: asOptionalNumber(raw.source_diversity_7d) ?? 0,
    persistence_days_7d: asOptionalNumber(raw.persistence_days_7d) ?? 0,
    momentum_score: asOptionalNumber(raw.momentum_score),
    queue_reason: asString(raw.queue_reason, 'candidate_review_pending'),
    evidence: isObject(raw.evidence) ? raw.evidence : {},
  };
};

export const adaptCandidateQueueResponse = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid candidate queue response');
  const itemsRaw = Array.isArray(raw.items) ? raw.items : [];
  const bandsRaw = Array.isArray(raw.confidence_bands) ? raw.confidence_bands : [];
  return {
    total: Number.isFinite(Number(raw.total)) ? Number(raw.total) : itemsRaw.length,
    items: itemsRaw.map(adaptCandidateQueueItem),
    confidence_bands: bandsRaw.map((band) => ({
      band: requireString(band?.band, 'band'),
      count: asOptionalNumber(band?.count) ?? 0,
    })),
  };
};

export const adaptCandidateReviewResponse = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid candidate review response');
  return {
    success: requireBoolean(raw.success, 'success'),
    action: requireString(raw.action, 'action'),
    updated: asOptionalNumber(raw.updated) ?? 0,
    skipped: asOptionalNumber(raw.skipped) ?? 0,
    results: Array.isArray(raw.results) ? raw.results : [],
    error: asString(raw.error),
  };
};

const adaptRelationshipGraphNode = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid relationship graph node');
  return {
    theme_cluster_id: requireNumber(raw.theme_cluster_id, 'theme_cluster_id'),
    theme_name: requireString(raw.theme_name, 'theme_name'),
    theme_display_name: requireString(raw.theme_display_name ?? raw.theme_name, 'theme_display_name'),
    lifecycle_state: requireString(raw.lifecycle_state, 'lifecycle_state'),
    is_root: requireBoolean(raw.is_root, 'is_root'),
  };
};

const adaptRelationshipGraphEdge = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid relationship graph edge');
  return {
    relation_id: requireNumber(raw.relation_id, 'relation_id'),
    source_theme_id: requireNumber(raw.source_theme_id, 'source_theme_id'),
    source_theme_name: asString(raw.source_theme_name),
    target_theme_id: requireNumber(raw.target_theme_id, 'target_theme_id'),
    target_theme_name: asString(raw.target_theme_name),
    relationship_type: requireString(raw.relationship_type, 'relationship_type'),
    confidence: asOptionalNumber(raw.confidence) ?? 0,
    provenance: asString(raw.provenance),
    evidence: isObject(raw.evidence) ? raw.evidence : {},
  };
};

export const adaptRelationshipGraphResponse = (raw) => {
  if (!isObject(raw)) throw new Error('Invalid relationship graph response');
  const nodesRaw = Array.isArray(raw.nodes) ? raw.nodes : [];
  const edgesRaw = Array.isArray(raw.edges) ? raw.edges : [];
  return {
    theme_cluster_id: requireNumber(raw.theme_cluster_id, 'theme_cluster_id'),
    total_nodes: asOptionalNumber(raw.total_nodes) ?? nodesRaw.length,
    total_edges: asOptionalNumber(raw.total_edges) ?? edgesRaw.length,
    nodes: nodesRaw.map(adaptRelationshipGraphNode),
    edges: edgesRaw.map(adaptRelationshipGraphEdge),
  };
};
