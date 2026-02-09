"""PR Impact Bot — automatic blast radius analysis for pull requests.

Runs as a GitHub Action that comments on PRs with:
  - Risk score (color-coded badge)
  - Changed symbols and their blast radius
  - Affected files tree
  - Suggested reviewers based on ownership
"""
