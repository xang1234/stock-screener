export const RRG_SCOPE_LABELS = {
  groups: 'Groups',
  sectors: 'Sectors',
};

export const RRG_SCOPE_ORDER = Object.keys(RRG_SCOPE_LABELS);

export const normalizeRrgScopes = (scopes, fallback = RRG_SCOPE_ORDER) => {
  const source = Array.isArray(scopes) ? scopes : fallback;
  const seen = new Set();
  return source.filter((scope) => {
    if (!RRG_SCOPE_LABELS[scope] || seen.has(scope)) {
      return false;
    }
    seen.add(scope);
    return true;
  });
};

export const availableRrgScopesFromBundle = (bundle) => {
  if (!bundle) {
    return null;
  }

  if (Array.isArray(bundle.available_scopes)) {
    return normalizeRrgScopes(bundle.available_scopes, []);
  }

  return normalizeRrgScopes(
    RRG_SCOPE_ORDER.filter((scope) => (bundle.payload?.[scope]?.groups ?? []).length > 0),
    [],
  );
};
