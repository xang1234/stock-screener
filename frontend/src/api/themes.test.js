import apiClient from './client';
import {
  getThemePolicyConfig,
  getCandidateThemeQueue,
  promoteStagedThemePolicy,
  revertThemePolicy,
  getThemeRelationshipGraph,
  updateThemePolicy,
  reviewCandidateThemes,
} from './themes';

vi.mock('./client', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

describe('theme api helpers', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches and adapts candidate queue payload', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        total: 1,
        items: [
          {
            theme_cluster_id: 31,
            theme_name: 'Power Demand',
            theme_display_name: 'Power Demand',
            avg_confidence_30d: 0.79,
            confidence_band: '0.70-0.84',
            mentions_7d: 8,
            source_diversity_7d: 3,
            persistence_days_7d: 5,
            queue_reason: 'candidate_review_pending',
          },
        ],
        confidence_bands: [{ band: '0.70-0.84', count: 1 }],
      },
    });

    const payload = await getCandidateThemeQueue({ limit: 50, offset: 0, pipeline: 'technical' });

    expect(apiClient.get).toHaveBeenCalledWith('/v1/themes/candidates/queue', {
      params: { limit: 50, offset: 0, pipeline: 'technical' },
    });
    expect(payload.total).toBe(1);
    expect(payload.items[0].theme_cluster_id).toBe(31);
  });

  it('posts candidate review action and adapts response', async () => {
    apiClient.post.mockResolvedValueOnce({
      data: {
        success: true,
        action: 'promote',
        updated: 2,
        skipped: 0,
        results: [],
      },
    });

    const result = await reviewCandidateThemes(
      { theme_cluster_ids: [1, 2], action: 'promote', actor: 'analyst' },
      'fundamental',
    );

    expect(apiClient.post).toHaveBeenCalledWith(
      '/v1/themes/candidates/review',
      { theme_cluster_ids: [1, 2], action: 'promote', actor: 'analyst' },
      { params: { pipeline: 'fundamental' } },
    );
    expect(result.success).toBe(true);
    expect(result.updated).toBe(2);
  });

  it('fetches relationship graph endpoint with root id', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: {
        theme_cluster_id: 7,
        total_nodes: 1,
        total_edges: 0,
        nodes: [
          {
            theme_cluster_id: 7,
            theme_name: 'AI Compute',
            theme_display_name: 'AI Compute',
            lifecycle_state: 'active',
            is_root: true,
          },
        ],
        edges: [],
      },
    });

    const graph = await getThemeRelationshipGraph(7, { pipeline: 'technical', limit: 80 });

    expect(apiClient.get).toHaveBeenCalledWith('/v1/themes/relationship-graph', {
      params: { theme_cluster_id: 7, pipeline: 'technical', limit: 80 },
    });
    expect(graph.theme_cluster_id).toBe(7);
    expect(graph.nodes[0].is_root).toBe(true);
  });

  it('gets theme policy config with admin key header', async () => {
    apiClient.get.mockResolvedValueOnce({ data: { pipeline: 'technical', defaults: {}, overrides: {}, effective: {}, history: [] } });
    const payload = await getThemePolicyConfig('technical', 'k123');
    expect(apiClient.get).toHaveBeenCalledWith('/v1/config/theme-policies', {
      params: { pipeline: 'technical' },
      headers: { 'X-Admin-Key': 'k123' },
    });
    expect(payload.pipeline).toBe('technical');
  });

  it('updates theme policy in preview mode', async () => {
    apiClient.post.mockResolvedValueOnce({ data: { status: 'preview', mode: 'preview' } });
    const payload = await updateThemePolicy(
      { pipeline: 'technical', mode: 'preview', matcher: { fuzzy_attach_threshold: 0.91 } },
      'k123',
      'tester',
    );
    expect(apiClient.post).toHaveBeenCalledWith('/v1/config/theme-policies', expect.any(Object), {
      headers: { 'X-Admin-Key': 'k123', 'X-Admin-Actor': 'tester' },
    });
    expect(payload.status).toBe('preview');
  });

  it('promotes staged policy and supports revert', async () => {
    apiClient.post
      .mockResolvedValueOnce({ data: { status: 'applied', pipeline: 'technical' } })
      .mockResolvedValueOnce({ data: { status: 'applied', pipeline: 'technical' } });

    const promoted = await promoteStagedThemePolicy('technical', 'ship', 'k123', 'tester');
    const reverted = await revertThemePolicy({ pipeline: 'technical', version_id: 'v1' }, 'k123', 'tester');

    expect(promoted.status).toBe('applied');
    expect(reverted.status).toBe('applied');
  });
});
