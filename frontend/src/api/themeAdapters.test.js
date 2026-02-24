import { describe, expect, it } from 'vitest';
import {
  adaptCandidateQueueResponse,
  adaptCandidateReviewResponse,
  adaptLifecycleTransitionsResponse,
  adaptMergeActionResponse,
  adaptMergeHistoryResponse,
  adaptMergeSuggestionsResponse,
  adaptRelationshipGraphResponse,
} from './themeAdapters';

describe('theme adapters', () => {
  it('normalizes merge suggestions from canonical fields', () => {
    const result = adaptMergeSuggestionsResponse({
      total: 1,
      suggestions: [
        {
          id: 10,
          source_theme_id: 1,
          source_theme_name: 'AI Infra',
          source_aliases: ['AI Infrastructure'],
          target_theme_id: 2,
          target_theme_name: 'AI Datacenter',
          target_aliases: [],
          similarity_score: 0.94,
          llm_confidence: 0.91,
          relationship_type: 'identical',
          reasoning: 'Same theme',
          suggested_name: 'AI Infrastructure',
          status: 'pending',
          created_at: '2026-02-24T10:00:00Z',
        },
      ],
    });

    expect(result.total).toBe(1);
    expect(result.suggestions[0].source_theme_name).toBe('AI Infra');
    expect(result.suggestions[0].similarity_score).toBe(0.94);
  });

  it('normalizes merge suggestions from legacy fields', () => {
    const result = adaptMergeSuggestionsResponse({
      suggestions: [
        {
          id: 11,
          source_cluster_id: 3,
          source_name: 'Legacy Source',
          target_cluster_id: 4,
          target_name: 'Legacy Target',
          embedding_similarity: 0.88,
          llm_relationship: 'related',
          llm_reasoning: 'Close but distinct',
          suggested_canonical_name: 'Legacy Target',
          status: 'pending',
        },
      ],
    });

    expect(result.total).toBe(1);
    expect(result.suggestions[0].source_theme_id).toBe(3);
    expect(result.suggestions[0].target_theme_name).toBe('Legacy Target');
    expect(result.suggestions[0].relationship_type).toBe('related');
  });

  it('throws on invalid merge suggestion id', () => {
    expect(() =>
      adaptMergeSuggestionsResponse({
        suggestions: [{ id: 'bad', source_theme_id: 1, source_theme_name: 'A', target_theme_id: 2, target_theme_name: 'B', status: 'pending' }],
      })
    ).toThrow(/merge suggestion id/i);
  });

  it('normalizes merge history list', () => {
    const result = adaptMergeHistoryResponse({
      total: 1,
      history: [
        {
          id: 1,
          source_name: 'Source',
          target_name: 'Target',
          merge_type: 'manual',
          merged_at: '2026-02-24T11:00:00Z',
          merged_by: 'system',
        },
      ],
    });

    expect(result.total).toBe(1);
    expect(result.history[0].merge_type).toBe('manual');
  });

  it('normalizes lifecycle transition history and metadata', () => {
    const result = adaptLifecycleTransitionsResponse({
      total: 1,
      transitions: [
        {
          id: 5,
          theme_cluster_id: 8,
          theme_name: 'Power Grid',
          pipeline: 'technical',
          from_state: 'active',
          to_state: 'dormant',
          actor: 'system',
          transition_metadata: null,
          transition_history_path: '/api/v1/themes/lifecycle-transitions?theme_cluster_id=8',
          runbook_url: 'https://example.com/runbook',
        },
      ],
    });

    expect(result.total).toBe(1);
    expect(result.transitions[0].transition_metadata).toEqual({});
  });

  it('normalizes merge action responses', () => {
    const result = adaptMergeActionResponse({
      success: true,
      source_name: 'Source',
      target_name: 'Target',
      constituents_merged: 5,
      mentions_merged: 10,
    });
    expect(result.success).toBe(true);
    expect(result.constituents_merged).toBe(5);
  });

  it('throws when merge action success is not a boolean', () => {
    expect(() =>
      adaptMergeActionResponse({
        success: 'false',
      })
    ).toThrow(/expected boolean/i);
  });

  it('normalizes candidate queue response', () => {
    const result = adaptCandidateQueueResponse({
      total: 1,
      items: [
        {
          theme_cluster_id: 9,
          theme_name: 'AI Datacenter',
          theme_display_name: 'AI Datacenter',
          candidate_since_at: '2026-02-24T10:00:00Z',
          avg_confidence_30d: 0.81,
          confidence_band: '0.70-0.84',
          mentions_7d: 12,
          source_diversity_7d: 4,
          persistence_days_7d: 6,
          momentum_score: 74.5,
          queue_reason: 'candidate_review_pending',
          evidence: { mentions_7d: 12 },
        },
      ],
      confidence_bands: [{ band: '0.70-0.84', count: 1 }],
    });

    expect(result.total).toBe(1);
    expect(result.items[0].theme_cluster_id).toBe(9);
    expect(result.confidence_bands[0].count).toBe(1);
  });

  it('normalizes candidate review response', () => {
    const result = adaptCandidateReviewResponse({
      success: true,
      action: 'promote',
      updated: 2,
      skipped: 1,
      results: [],
    });
    expect(result.success).toBe(true);
    expect(result.updated).toBe(2);
  });

  it('normalizes relationship graph response', () => {
    const result = adaptRelationshipGraphResponse({
      theme_cluster_id: 7,
      total_nodes: 2,
      total_edges: 1,
      nodes: [
        {
          theme_cluster_id: 7,
          theme_name: 'Power Grid',
          theme_display_name: 'Power Grid',
          lifecycle_state: 'active',
          is_root: true,
        },
        {
          theme_cluster_id: 11,
          theme_name: 'Utilities',
          theme_display_name: 'Utilities',
          lifecycle_state: 'active',
          is_root: false,
        },
      ],
      edges: [
        {
          relation_id: 5,
          source_theme_id: 7,
          source_theme_name: 'Power Grid',
          target_theme_id: 11,
          target_theme_name: 'Utilities',
          relationship_type: 'related',
          confidence: 0.8,
        },
      ],
    });

    expect(result.theme_cluster_id).toBe(7);
    expect(result.nodes).toHaveLength(2);
    expect(result.edges[0].relationship_type).toBe('related');
  });
});
