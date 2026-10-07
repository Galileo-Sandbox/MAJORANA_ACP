# Artifact tools

`verify_artifact.py` validates the portable paper evidence against
`manifests/artifact_release_v1.json`. With `--server`, it also verifies every
saved prediction used by the presentation export and every mechanism-control
prediction/checkpoint listed in the run registries.

`restore_resum_flex.py` validates and safely extracts a user-supplied archive
of the exact historical RESUM_FLEX source snapshot. It does not download or
redistribute that source.

Run both tools from any working directory; repository paths are resolved from
the script location. Neither tool modifies tracked files. The restore tool
only writes to the explicitly selected ignored destination and refuses to
overwrite it.
