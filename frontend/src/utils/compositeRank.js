// A young IPO's composite is its IPO score plus an uplift, so best-first
// composite sorts rank rows Minervini or CANSLIM scored ahead of it, as the
// backend's composite_rank_order does.
const coreScoreTier = (row) => (row?.minervini_score == null && row?.canslim_score == null ? 1 : 0);

export const compareCoreScoredFirst = (left, right) => coreScoreTier(left) - coreScoreTier(right);
