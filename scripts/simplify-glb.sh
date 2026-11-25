#!/bin/sh
# Reduce the triangle count of a GLB with gltfpack (meshoptimizer), keeping
# plain float attributes, named nodes (the wheels turn in the game) and
# materials, so the game's loader reads the result as it is.
#
# usage: simplify-glb.sh in.glb out.glb [ratio=0.3]
#   e.g. scripts/optimize-glb.py source.glb /tmp/small-textures.glb 1024
#        scripts/simplify-glb.sh /tmp/small-textures.glb src/assets/fire_truck_osaka_st_game.glb
set -e
npx --yes gltfpack@1.3 -i "$1" -o "$2" -si "${3:-0.3}" -se 0.004 -kn -km -noq
