#!/bin/bash
set -e

# -------------------------
# Paths
# -------------------------
IMAGES=images
DB=database.db
SPARSE=sparse
DENSE=dense

# -------------------------
# Create required folders
# -------------------------
mkdir -p $IMAGES
mkdir -p $SPARSE
mkdir -p $DENSE

if [ -f "$DB" ]; then
  echo "Removing existing database.db"
  rm $DB
fi

# -------------------------
# Feature extraction
# -------------------------
colmap feature_extractor \
  --database_path $DB \
  --image_path $IMAGES \
  --ImageReader.single_camera 1 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.num_octaves 5 \
  --SiftExtraction.octave_resolution 4 \
  --SiftExtraction.peak_threshold 0.006 \
  --SiftExtraction.edge_threshold 10

# -------------------------
# Matching
# -------------------------
colmap exhaustive_matcher \
  --database_path $DB \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.max_num_matches 32768

# -------------------------
# Sparse reconstruction
# -------------------------
colmap mapper \
  --database_path $DB \
  --image_path $IMAGES \
  --output_path $SPARSE \
  --Mapper.init_min_num_inliers 50 \
  --Mapper.min_num_matches 15 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 1 \
  --Mapper.ba_refine_extra_params 1

# -------------------------
# Sparse model must exist
# -------------------------
if [ ! -d "$SPARSE/0" ]; then
  echo "ERROR: sparse/0 was not created. Mapper failed."
  exit 1
fi

# -------------------------
# Image undistortion
# -------------------------
mkdir -p $DENSE/0

colmap image_undistorter \
  --image_path $IMAGES \
  --input_path $SPARSE/0 \
  --output_path $DENSE/0 \
  --output_type COLMAP

# -------------------------
# Dense stereo
# -------------------------
colmap patch_match_stereo \
  --workspace_path $DENSE/0 \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 3200 \
  --PatchMatchStereo.num_samples 15 \
  --PatchMatchStereo.window_radius 7

# -------------------------
# Depth fusion
# -------------------------
colmap stereo_fusion \
  --workspace_path $DENSE/0 \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DENSE/0/fused.ply \
  --StereoFusion.min_num_pixels 3

# -------------------------
# Meshing
# -------------------------
colmap poisson_mesher \
  --input_path $DENSE/0/fused.ply \
  --output_path $DENSE/0/poisson.ply \
  --PoissonMeshing.depth 10

echo "Reconstruction finished successfully"
