# Numerical Homogenization (WIP)

Numerical Homogenization of Partial Differential Equations

### Example usage:

Solve the same problem at two scales (coarse and fine) and draw the solution:

`make numhom`

`./numhom -coarse_dm_plex_box_faces 8 -fine_dm_plex_box_faces 32 -uf_view -uc_view -coarse_dm_plex_dim 1 -fine_dm_plex_dim 1 -petscspace_degree 1 -draw_pause -1 -draw_size 1000,1000 -coarse_dm_plex_hash_location -fine_dm_plex_hash_location -papt_view -ac_view -pf_view `
