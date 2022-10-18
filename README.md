# numhom
Numerical Homogenization of Partial Differential Equations

### Example usage:

Solve the same problem at two scales (coarse and fine) and view the solution:

`make numhom`
`./numhom -coarse_dm_plex_dim 1 -fine_dm_plex_dim 1 -coarse_dm_plex_box_faces 8 -fine_dm_plex_box_faces 36 -uc_view -uf_view`

