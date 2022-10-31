# Numerical Homogenization (WIP)

Numerical Homogenization of Partial Differential Equations

### Example usage:

Solve the same problem at two scales (coarse and fine) and draw the solution:

To compile:
`make numhom`

To draw the coarse and fine solution vectors: 
`./numhom -coarse_dm_plex_box_faces 8 -fine_dm_plex_box_faces 32 -uf_view -uc_view -coarse_dm_plex_dim 1 -fine_dm_plex_dim 1 -petscspace_degree 1 -draw_pause -1 -draw_size 1000,1000 -coarse_dm_plex_hash_location -fine_dm_plex_hash_location`

To dump the coarse, fine scale system matrix and the projection matrix to binary:
`./numhom -coarse_dm_plex_box_faces 8 -fine_dm_plex_box_faces 32 -coarse_dm_plex_dim 1 -fine_dm_plex_dim 1 -petscspace_degree 1 -coarse_dm_plex_hash_location -fine_dm_plex_hash_location -ac_mat_view binary:ac.mat -af_mat_view binary:af.mat -pf_mat_view binary:pf.mat`

