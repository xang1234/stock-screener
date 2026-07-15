from app.infra.query.like_pattern import literal_contains_pattern


def test_literal_contains_pattern_escapes_like_metacharacters():
    assert literal_contains_pattern(r"Fund_100%\Growth") == r"%Fund\_100\%\\Growth%"
