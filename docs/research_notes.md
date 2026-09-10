# Post-Processing Research Notes

These ideas were retained from exploratory work but are not implemented as
validated conclusions:

- Preserve more coauthors per paper, especially for biomedical publications
  with long author lists.
- Avoid truncating the coauthor candidate set too aggressively because repeated
  collaborators are strong identity evidence.
- Normalize organization strings to a coarser institution level before
  comparison.
- Avoid asking an LLM to merge hundreds of clusters in one request. Candidate
  clusters could first be partitioned by organization family, frequent
  coauthors, or broad topic, then compared in smaller groups.

These directions belong to the broader goal of combining deterministic graph
signals with LLM judgment before and after the main clustering call.
