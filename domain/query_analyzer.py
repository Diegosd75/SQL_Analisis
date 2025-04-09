def analyze_query(query: str) -> dict:
    """Analyzes an SQL query and identifies potential issues."""
    issues = []
    suggestions = []
    score = 5

    if "SELECT *" in query.upper():
        issues.append("Using SELECT * can be inefficient.")
        suggestions.append("Specify only the necessary columns to improve performance.")
        score -= 1

    if "JOIN" in query.upper() and "WHERE" not in query.upper():
        issues.append("JOIN without a WHERE clause may create a Cartesian product.")
        suggestions.append("Ensure a proper ON condition to optimize performance.")
        score -= 1

    return {"score": max(0, score), "issues": issues, "suggestions": suggestions}
