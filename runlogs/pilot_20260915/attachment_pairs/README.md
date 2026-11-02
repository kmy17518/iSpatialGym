# Attachment first-demo pairs (2026-09-19/20)

The `configuration_matching-attachment-lid-teapot-breakfast_table` variant entries that produced the valid demos, one file per
lid / teapot pair (the shared `ispatialgym/assets/task_specs/isg_customs.json` whitelists only tnwisp / mvrhya; the sampler's
model draw does not consult `roles/constraints_compatible.csv`, so the second pair runs from an experiment copy of the customs).

To reproduce: copy `isg_customs.json`, replace the variant entry with the file's, and run
`python scripts/isg/first_demo.py run --variant configuration_matching-attachment-lid-teapot-breakfast_table --customs <copy>`
(or `sample_isg --customs <copy>`).

| pair | variant entry | valid demo | generation knobs beyond the shared entry |
|---|---|---|---|
| lid tnwisp / teapot mvrhya | `tnwisp_mvrhya.variant.json` | `first_demo/configuration_matching-attachment/0919-231626` | `grasp_height_extra_m 0.008` |
| lid pwgbsb / teapot jlalfc | `pwgbsb_jlalfc.variant.json` | `first_demo/configuration_matching-attachment/0919-231624` | whitelist pwgbsb / jlalfc, `grasp_keys lid/pwgbsb`, `partner_gap_min_m 0.08`, `partner_centre_distance_min_m 0.18`, `grasp_height_extra_m 0.006` |

Both entries carry the shared `brace_search: false` (the owner's one-arm sequence) and `witness_residual_pre_m 0.015`.
Open question for the owner: a second grounded variant for the pwgbsb pair, or a pair-aware model draw (`iter_template_choices`
honouring `partner_of` rows), so the pair lives in the shared customs.
