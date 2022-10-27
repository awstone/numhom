# Numerical Homogenization (WIP)

Numerical Homogenization of Partial Differential Equations

### Example usage:

Solve the same problem at two scales (coarse and fine) and view the solution:

`make numhom`

`./numhom -coarse_dm_plex_box_faces 10 -fine_dm_plex_box_faces 100 -uf_view -uc_view -coarse_dm_plex_dim 1 -fine_dm_plex_dim 1 -petscspace_degree 1 -draw_pause -1`

