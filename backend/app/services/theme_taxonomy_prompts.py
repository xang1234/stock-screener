"""LLM prompts for L1 taxonomy naming and grouping validation."""

L1_CATEGORIES = {
    "technology",
    "healthcare",
    "energy",
    "defense",
    "financials",
    "materials",
    "consumer",
    "industrials",
    "macro",
    "crypto",
    "real_estate",
    "other",
}


def build_l1_naming_prompt(theme_names: list[str]) -> str:
    """Build a prompt for LLM to name an L1 cluster from its L2 theme names."""
    names_list = "\n".join(f"- {name}" for name in theme_names)
    categories_list = ", ".join(sorted(L1_CATEGORIES))

    return f"""You are a financial analyst organizing market investment themes into groups.

Given the following cluster of related market themes, provide:
1. A concise L1 parent theme name (2-4 words, broad investable category)
2. A sector category from this list: {categories_list}
3. A one-sentence description of the investment thesis

Themes in this cluster:
{names_list}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "l1_name": "Broad Theme Name",
  "category": "category_from_list",
  "description": "One sentence investment thesis description"
}}"""
